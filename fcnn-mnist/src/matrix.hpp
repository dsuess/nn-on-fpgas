#ifndef NNONFPGA_UTILS
#define NNONFPGA_UTILS

#include <tuple>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <CL/cl2.hpp>
#include <nonstd/optional.hpp>

typedef unsigned int uint;

static const uint DEFAULT_ALIGNMENT = 4096;

typedef struct DeviceHandle
{
    cl::Device device;
    cl::CommandQueue q;
    cl::Context context;
} DeviceHandle;

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
    const uint cols, rows;
    const uint alignment;

    Matrix(const uint rows, const uint cols, const uint alignment = DEFAULT_ALIGNMENT) : cols(cols), rows(rows), alignment(alignment)
    {
        data = aligned_alloc<float>(cols * rows, alignment);
    }
    ~Matrix()
    {
        free(data);
    }
    float &operator[](const std::tuple<uint, uint> &idx) const
    {
        const auto fidx = flatten_idx(std::get<0>(idx), std::get<1>(idx));
        return data[fidx];
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
            throw "Put data on device first";
        }
    }

    Matrix &to_device(DeviceHandle &handle, const int bank = XCL_MEM_DDR_BANK1)
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

        // we need to keep track of event so that queue can wait on this
        cl::Event kernel_evt;
        handle.q.enqueueMigrateMemObjects(ob_io, 0, nullptr, &kernel_evt);
        return *this;
    }

    Matrix &to_cpu(DeviceHandle &handle)
    {
        std::vector<cl::Memory> ob_io;
        ob_io.push_back(device_buffer.value());
        // we need to keep track of event so that queue can wait on this
        handle.q.enqueueMigrateMemObjects(ob_io, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, nullptr);
        return *this;
    }
};

#endif /* end of include guard: NNONFPGA_UTILS */