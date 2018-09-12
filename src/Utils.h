#ifndef UTILS_H
#define UTILS_H

#include "General.h"

namespace at {


std::vector<int64_t> calculate_contiguous_stride(IntList sizes) {
    std::vector<int64_t> strides(sizes.size());
    int ndim = sizes.size();

    for (int d = ndim - 1; d >= 0; d--)
    {
        if (d == ndim - 1) {
            strides[d] = 1;
        }
        else {
            strides[d] = std::max<int64_t>(sizes[d+1], 1) * strides[d+1];
        }
    }
    return strides;
}

// Maybe someone wants to move this in Tensor/TensorImpl?
bool is_transposed(const TensorImpl *self) {
    int64_t max_stride = 1;
    int64_t size_max_stride = 1;
    int64_t z = 1;
    int d;
    for (d = 0; d < self->dim(); ++d) {
      if (self->stride(d) == 0 && self->size(d) != 1)
        return false;
      if (self->stride(d) > max_stride) {
        max_stride = self->stride(d);
        size_max_stride = self->size(d);
      }
      z *= self->size(d);
    }
    if (z == max_stride * size_max_stride) {
      return true;
    }
    return false;
}

}

#endif // UTILS_H