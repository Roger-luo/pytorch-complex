#ifndef DEFAULT_H
#define DEFAULT_H

#include <complex>

namespace simd {

template <typename T>
struct Default {
    static inline void copy(T *y, const T *x, const ptrdiff_t n);
    static inline void fill(T *z, const T c, const ptrdiff_t n);
    static inline void cdiv(T *z, const T *x, const T *y, const ptrdiff_t n);
    static inline void divs(T *z, const T *x, const T c, const ptrdiff_t n);
    static inline void cmul(T *z, const T *x, const T *y, const ptrdiff_t n);
    static inline void muls(T *y, const T *x, const T c, const ptrdiff_t n);
    static inline void cadd(T *z, const T *x, const T *y, const T c, const ptrdiff_t n);
    static inline void adds(T *y, const T *x, const T c, const ptrdiff_t n);
};

} // simd

#include "DefaultImpl.h"

#endif // DEFAULT_H
