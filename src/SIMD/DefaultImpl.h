#include "Default.h"

namespace simd {

template <typename T>
inline void Default<T>::copy(T *y, const T *x, const ptrdiff_t n) {
    ptrdiff_t i = 0;

    for(; i <n-4; i+=4)
    {
        x[i] = y[i];
        x[i+1] = y[i+1];
        x[i+2] = y[i+2];
        x[i+3] = y[i+3];
    }

    for(; i < n; i++)
        x[i] = y[i];
}

template <typename T>
inline void Default<T>::fill(T *x, const T c, const ptrdiff_t n) {
    ptrdiff_t i = 0;

    for(; i <n-4; i+=4)
    {
        x[i] = c;
        x[i+1] = c;
        x[i+2] = c;
        x[i+3] = c;
    }

    for(; i < n; i++)
        x[i] = c;
}

template <typename T>
inline void Default<T>::cadd(T *z, const T *x, const T *y, const T c, const ptrdiff_t n) {
    ptrdiff_t i = 0;

    for(; i<n-4; i+=4)
    {
        z[i] = x[i] + c * y[i];
        z[i+1] = x[i+1] + c * y[i+1];
        z[i+2] = x[i+2] + c * y[i+2];
        z[i+3] = x[i+3] + c * y[i+3];
    }

    for(; i<n; i++)
        z[i] = x[i] + c * y[i];
}

template <typename T>
inline void Default<T>::adds(T *y, const T *x, const T c, const ptrdiff_t n) {
    ptrdiff_t i = 0;

    for(; i<n-4; i+=4)
    {
        y[i] = x[i] + c;
        y[i+1] = x[i+1] + c;
        y[i+2] = x[i+2] + c;
        y[i+3] = x[i+3] + c;
    }

    for(; i<n; i++)
        y[i] = x[i] + c;
}

template <typename T>
inline void Default<T>::cmul(T *z, const T *x, const T*y, const ptrdiff_t n) {
    ptrdiff_t i = 0;

    for(; i <n-4; i+=4)
    {
        z[i] = x[i] * y[i];
        z[i+1] = x[i+1] * y[i+1];
        z[i+2] = x[i+2] * y[i+2];
        z[i+3] = x[i+3] * y[i+3];
    }

    for(; i < n; i++)
        z[i] = x[i] * y[i];
}

template <typename T>
inline void Default<T>::muls(T *y, const T *x, const T c, const ptrdiff_t n)
{
    ptrdiff_t i = 0;

    for(; i <n-4; i+=4)
    {
        y[i] = x[i] * c;
        y[i+1] = x[i+1] * c;
        y[i+2] = x[i+2] * c;
        y[i+3] = x[i+3] * c;
    }

    for(; i < n; i++)
        y[i] = x[i] * c;
}

template <typename T>
inline void Default<T>::cdiv(T *z, const T *x, const T *y, const ptrdiff_t n)
{
    ptrdiff_t i = 0;

    for(; i<n-4; i+=4)
    {
        z[i] = x[i] / y[i];
        z[i+1] = x[i+1] / y[i+1];
        z[i+2] = x[i+2] / y[i+2];
        z[i+3] = x[i+3] / y[i+3];
    }

    for(; i < n; i++)
        z[i] = x[i] / y[i];
}

template <typename T>
inline void Default<T>::divs(T *y, const T *x, const T c, const ptrdiff_t n)
{
    ptrdiff_t i = 0;

    for(; i<n-4; i+=4)
    {
        y[i] = x[i] / c;
        y[i+1] = x[i+1] / c;
        y[i+2] = x[i+2] / c;
        y[i+3] = x[i+3] / c;
    }

    for(; i < n; i++)
        y[i] = x[i] / c;
}

} // simd