#ifndef COMPLEX_TYPE_INFO_H
#define COMPLEX_TYPE_INFO_H

#include "General.h"

namespace at {

template <typename T, Backend device>
struct TypeInfo;

template <>
struct TypeInfo<float, Backend::CPU> {
    using scalar_t = float;
    using precision_t = float;

    static const auto scalar_type = ScalarType::Float;
    static const auto type_id = TypeID::CPUFloat;
};

template <>
struct TypeInfo<double, Backend::CPU> {
    using scalar_t = double;
    using precision_t = double;

    static const auto scalar_type = ScalarType::Double;
    static const auto type_id = TypeID::CPUDouble;
};

template <>
struct TypeInfo<std::complex<float>, Backend::CPU> {
    using scalar_t = std::complex<float>;
    using precision_t = float;

    static const auto scalar_type = ScalarType::ComplexFloat;
    static const auto type_id = TypeID::CPUComplexFloat;
};

template <>
struct TypeInfo<std::complex<double>, Backend::CPU> {
    using scalar_t = std::complex<double>;
    using precision_t = double;

    static const auto scalar_type = ScalarType::ComplexDouble;
    static const auto type_id = TypeID::CPUComplexDouble;
};


template <typename T>
using CPUTypeInfo = TypeInfo<T, Backend::CPU>;

template <typename T, Backend device>
using ComplexTypeInfo = TypeInfo<std::complex<T>, device>;

template <typename PrecisionType>
using CPUComplexTypeInfo = ComplexTypeInfo<PrecisionType, Backend::CPU>;

} // at

#endif // COMPLEX_TYPE_INFO_H