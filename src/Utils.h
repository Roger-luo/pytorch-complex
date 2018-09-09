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
}

#endif // UTILS_H