#ifndef CPU_COMPLEX_COPY_H
#define CPU_COMPLEX_COPY_H

#include "General.h"
#include "ComplexTypeInfo.h"
#include "CPUComplexType.h"

namespace at {

template <typename DST, typename SRC>
struct CPUCopy;

// template arguments is mixed with C macros, e.g
// TH_TENSOR_APPLY2(CPUTypeInfo<double, double>, ...)
// will not be correct...

// template <typename Type>
// struct CPUCopy<Type, Type> {

//     inline static void eval(TensorImpl *dst, TensorImpl *src) {
//         CPUTypeInfo<Type>::scalar_t *dst_data = NULL;

//         TH_TENSOR_APPLY2(
//             CPUTypeInfo<Type>::scalar_t, dst,
//             CPUTypeInfo<Type>::scalar_t, src,
//             *dst_data = static_cast<CPUTypeInfo<Type>::scalar_t>(static_cast<inter_copy_type_t<CPUTypeInfo<Type>::scalar_t>>(*src_data));
//         )
//     }
// };

// Copy from THTensorCopy
// 
// C and C++ have a lovely set of implicit conversion rules, where casting
// signed integral values to unsigned integral values is always valid
// (it basically treats the value as if using modulo arithmetic), however
// converting negative floating point values to unsigned integral types
// is UB! This means that: (double)-1 -> (int64_t)-1 -> (uint8_t)255 is
// guaranteed to look like this, but we have (double)-1 -> (uint8_t)<ANYTHING>
// because it's UB. This also makes UBSan really angry.
//
// I think those rules are stupid and we really shouldn't conform to them.
// The structs below ensure that for all unsigned types we use (currently
// only uint8_t), we will do an intermediate convertion via int64_t,
// to ensure that any negative values are wrapped around correctly.
//
// Note that conversions from doubles to signed integral types that can't
// represent a particular value after truncating the fracitonal part are UB as well,
// but fixing them is not as simple as adding an int64_t intermediate, beacuse the
// int64_t -> <smaller signed type> conversion is UB for those large values anyway.
// I guess in that case we just have to live with that, but it's definitely less
// surprising than the thing above.
//
// For the curious:
//   https://en.cppreference.com/w/cpp/language/implicit_conversion
//   The relevant paragraph is "Floating–integral conversions".

template<typename T>
struct inter_copy_type {
  using type = T;
};

template<>
struct inter_copy_type<uint8_t> {
  using type = int64_t;
};

template<typename T>
using inter_copy_type_t = typename inter_copy_type<T>::type;


template <typename DST, typename SRC>
struct CPUCopy {
    inline static void eval(TensorImpl *dst, TensorImpl *src) {
        TH_TENSOR_APPLY2(
            DST, dst,
            SRC, src,
            *dst_data = static_cast<DST>(static_cast<inter_copy_type_t<SRC>>(*src_data));
        )
    }
};

// copy from complex to real
template <typename DST, typename SRC>
struct CPUCopy<DST, std::complex<SRC>> {
    inline static void eval(TensorImpl *dst, TensorImpl *src) {
        TH_TENSOR_APPLY2(
            DST, dst,
            std::complex<SRC>, src,
            *dst_data = static_cast<DST>(static_cast<inter_copy_type_t<SRC>>((*src_data).real()));
        )
    }
};


template <typename PT>
Tensor & CPUComplexType<PT>::s_copy_(Tensor & dst, const Tensor & src, bool non_blocking) const {
    checked_tensor_unwrap(dst, "dst", 0, false, Backend::CPU, CPUComplexTypeInfo<PT>::scalar_type);

    switch (src.type().ID()) {
        case TypeID::CPUByte:
            CPUCopy<std::complex<PT>, int8_t>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUChar:
            CPUCopy<std::complex<PT>, int8_t>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUDouble:
            std::cout << "double is copied to complex" << std::endl;
            CPUCopy<std::complex<PT>, double>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUFloat:
            CPUCopy<std::complex<PT>, float>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUComplexFloat:
            CPUCopy<std::complex<PT>, std::complex<float>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUComplexDouble:
            CPUCopy<std::complex<PT>, std::complex<double>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUInt:
            CPUCopy<std::complex<PT>, int32_t>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPULong:
            CPUCopy<std::complex<PT>, int64_t>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUShort:
            CPUCopy<std::complex<PT>, int16_t>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUHalf:
            std::cout << "copy half" << std::endl;
            break;
        default:
            return src.type()._s_copy_from(src, dst, non_blocking);
    }

    dst.unsafeGetTensorImpl()->maybe_zero_dim(src.dim() == 0);
    return dst;
}

template <typename PT>
Tensor & CPUComplexType<PT>::_s_copy_from(const Tensor & src, Tensor & dst, bool non_blocking) const {
    // This handles the copy from other types

    switch (dst.type().ID()) {
        case TypeID::CPUByte:
            CPUCopy<int8_t, std::complex<PT>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUChar:
            CPUCopy<int8_t, std::complex<PT>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUDouble:
            CPUCopy<double, std::complex<PT>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUFloat:
            CPUCopy<float, std::complex<PT>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUInt:
            CPUCopy<int32_t, std::complex<PT>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPULong:
            CPUCopy<int64_t, std::complex<PT>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        case TypeID::CPUShort:
            CPUCopy<int16_t, std::complex<PT>>::eval(dst.unsafeGetTensorImpl(), src.unsafeGetTensorImpl());
            break;
        default:
            AT_ERROR("copy does not support ", src.type().toString(), " to ", dst.type().toString(), " copy (s_copy_from case).");
    }
    dst.unsafeGetTensorImpl()->maybe_zero_dim(src.dim() == 0);
    return dst;
}

} // at

#endif // CPU_COMPLEX_COPY_H