#ifndef NNONFPGA_UTILS
#define NNONFPGA_UTILS

// Memory alignment
template <typename T>
T *aligned_alloc(std::size_t num)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(T)))
    {
        throw std::bad_alloc();
    }
    return reinterpret_cast<T *>(ptr);
}

#endif /* end of include guard: NNONFPGA_UTILS */