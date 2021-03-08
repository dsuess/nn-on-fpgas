#ifndef NNONFPGA_UTILS
#define NNONFPGA_UTILS

#include <tuple>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <CL/cl2.hpp>
#include <nonstd/optional.hpp>
#include "libnpy.hpp"
#include "utils.hpp"

typedef unsigned int uint;

static const uint DEFAULT_ALIGNMENT = 4096;

#ifdef HW_EMU_MODE
// Hardware emulation doesn't work with any other bank
static const int DEFAULT_MEMORY_BANK = XCL_MEM_DDR_BANK1;
#else
static const int DEFAULT_MEMORY_BANK = XCL_MEM_DDR_BANK0;
#endif

// Memory alignment
template <typename T>
T *aligned_alloc(std::size_t num, std::size_t alignment = DEFAULT_ALIGNMENT)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, alignment, num * sizeof(T)))
    {
        throw std::bad_alloc();
    }
    return reinterpret_cast<T *>(ptr);
}

class Matrix
{
private:
    inline uint flatten_idx(const uint row, const uint col) const
    {
        assert(row < rows);
        assert(col < cols);
        return cols * row + col;
    }

protected:
    float *data;
    nonstd::optional<cl::Buffer> device_buffer;

public:
    uint cols, rows;
    uint alignment;

    Matrix() : cols(0), rows(0), alignment(DEFAULT_ALIGNMENT) { data = NULL; };
    Matrix(const uint rows, const uint cols, const uint alignment = DEFAULT_ALIGNMENT) : cols(cols), rows(rows), alignment(alignment)
    {
        data = aligned_alloc<float>(cols * rows, alignment);
    }
    Matrix(const Matrix &src) : rows(src.rows), cols(src.cols), alignment(src.alignment)
    {
        data = aligned_alloc<float>(cols * rows, alignment);
        device_buffer = src.device_buffer;
        memcpy(data, src.data, rows * cols * sizeof(float));
    }
    Matrix(Matrix &&src) : rows(src.rows), cols(src.cols), alignment(src.alignment)
    {
        data = src.data;
        device_buffer = src.device_buffer;
        src.data = NULL;
    }
    Matrix &operator=(const Matrix &src)
    {
        if (&src != this)
        {
            rows = src.rows;
            cols = src.cols;
            alignment = src.alignment;
            device_buffer = src.device_buffer;
            if (data != NULL)
            {
                free(data);
            }
            data = aligned_alloc<float>(cols * rows, alignment);
            memcpy(data, src.data, rows * cols * sizeof(float));
        }
    }
    Matrix &operator=(Matrix &&src)
    {
        if (&src != this)
        {
            rows = src.rows;
            cols = src.cols;
            alignment = src.alignment;
            device_buffer = src.device_buffer;
            if (data != NULL)
            {
                free(data);
            }
            data = src.data;
            src.data = NULL;
        }
        return *this;
    }

    ~Matrix()
    {
        if (data != NULL)
        {

            free(data);
        }
    }

    float &operator()(const uint row, const uint col)
    {
        const auto idx = flatten_idx(row, col);
        return data[idx];
    }

    std::string to_string()
    {
        std::ostringstream res;
        for (uint i = 0; i < rows; i++)
        {
            for (uint j = 0; j < cols; j++)
            {
                const auto fidx = flatten_idx(i, j);
                res << data[fidx] << " ";
            }
            res << std::endl;
        }
        return res.str();
    }

    static Matrix constant(const uint rows, const uint cols, const float val, const uint alignment = DEFAULT_ALIGNMENT)
    {
        Matrix mat(rows, cols, alignment);
        for (uint i = 0; i < rows * cols; i++)
        {
            mat.data[i] = val;
        }
        return mat;
    }

    static Matrix from_npy(const std::string &path)
    {
        int rows, cols;
        std::vector<float> data;
        aoba::LoadArrayFromNumpy(path, rows, cols, data);
        Matrix mat(rows, cols);
        memcpy(mat.data, data.data(), cols * rows * sizeof(float));
        return mat;
    }

    Matrix &clear_device_buffer()
    {
        if (device_buffer.has_value())
        {
            device_buffer.reset();
        }
        return *this;
    }

    cl::Buffer &get_buffer()
    {
        if (device_buffer.has_value())
        {
            return device_buffer.value();
        }
        else
        {
            std::cerr << "Put data on device first";
            throw -1;
        }
    }

    Matrix &to_device(DeviceHandle &handle = HANDLE, const int bank = DEFAULT_MEMORY_BANK)
    {
        clear_device_buffer();
        cl_mem_ext_ptr_t mext_io;
        mext_io.flags = bank;
        mext_io.obj = data;
        mext_io.param = 0;

        // Since memory is page-aligned, we don't need to copy and cal use CL_MEM_USE_HOST_PTR
        assert(alignment == 4096);
        device_buffer = nonstd::optional<cl::Buffer>{cl::Buffer(handle.context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                                sizeof(float) * rows * cols, &mext_io)};
        std::vector<cl::Memory> ob_io;
        ob_io.push_back(device_buffer.value());
        handle.q.enqueueMigrateMemObjects(ob_io, 0, nullptr, nullptr);
        return *this;
    }

    Matrix &to_cpu(DeviceHandle &handle = HANDLE)
    {
        std::vector<cl::Memory> ob_io;
        if (!device_buffer.has_value())
        {
            std::cerr << "Trying to copy values that don't exist" << std::endl;
            throw 21;
        }
        ob_io.push_back(device_buffer.value());
        handle.q.enqueueMigrateMemObjects(ob_io, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, nullptr);
        return *this;
    }
};

std::pair<Matrix, cl::Event> apply_matmul(Matrix &matrixA, Matrix &matrixB, cl::Kernel &kernel, std::vector<cl::Event> *wait_on = NULL, DeviceHandle &handle = HANDLE)
{
    Matrix result = Matrix::constant(matrixA.rows, matrixB.cols, 0.0, 4096);
    result.to_device(handle);
    kernel.setArg(0, matrixA.get_buffer());
    kernel.setArg(1, matrixB.get_buffer());
    kernel.setArg(2, matrixA.rows);
    kernel.setArg(3, matrixA.cols);
    kernel.setArg(4, matrixB.cols);
    kernel.setArg(5, result.get_buffer());

    cl::Event event;
    handle.q.enqueueTask(kernel, wait_on, &event);
    return std::make_pair(std::move(result), event);
}

cl::Event apply_bias(Matrix &input, Matrix &bias, cl::Kernel &kernel, std::vector<cl::Event> *wait_on = NULL, DeviceHandle &handle = HANDLE)
{
    kernel.setArg(0, input.get_buffer());
    kernel.setArg(1, bias.get_buffer());
    kernel.setArg(2, input.rows);
    kernel.setArg(3, input.cols);

    cl::Event event;
    handle.q.enqueueTask(kernel, wait_on, &event);
    return std::move(event);
}

#endif /* end of include guard: NNONFPGA_UTILS */