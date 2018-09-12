#ifndef AVX2_H
#define AVX2_H

#if defined(__AVX2__)
#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

#include <complex>

namespace simd {

template <typename T>
struct AVX2 {
    static inline void copy(T *y, const T *x, const ptrdiff_t n);
    static inline void fill(T *z, const T c, const ptrdiff_t n);
    static inline void cdiv(T *z, const T *x, const T *y, const ptrdiff_t n);
    static inline void divs(T *z, const T *x, const T c, const ptrdiff_t n);
    static inline void cmul(T *z, const T *x, const T *y, const ptrdiff_t n);
    static inline void muls(T *z, const T *x, const T c, const ptrdiff_t n);
    static inline void cadd(T *z, const T *x, const T c, const ptrdiff_t n);
};

} // simd

#endif // defined(__AVX__)
#endif // AVX2_H