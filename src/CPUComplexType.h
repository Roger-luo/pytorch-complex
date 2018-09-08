#ifndef CPUComplexType_H
#define CPUComplexType_H

#include <ATen/detail/ComplexHooksInterface.h>
#include <ATen/detail/VariableHooksInterface.h>
#include <ATen/Type.h>
#include <ATen/CPUFloatType.h>

#include "ATen/TensorImpl.h"
#include "ATen/CPUGenerator.h"
#include "ATen/TensorImpl.h"
#include "ATen/Allocator.h"
#include "ATen/DeviceGuard.h"
#include "ATen/NativeFunctions.h"
#include "ATen/UndefinedTensor.h"
#include "ATen/Utils.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/core/Half.h"
#include "ATen/core/optional.h"

#include "TH/THTensor.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "ATen/Config.h"

namespace at {

template <typename T, Backend device>
struct ComplexTypeInfo;

template <>
struct ComplexTypeInfo<float, Backend::CPU> {
    static const auto scalar_type = ScalarType::ComplexFloat;
    static const auto type_id = TypeID::CPUComplexFloat;
};

template <>
struct ComplexTypeInfo<double, Backend::CPU> {
    static const auto scalar_type = ScalarType::ComplexDouble;
    static const auto type_id = TypeID::CPUComplexDouble;
};

template <typename PrecisionType>
using CPUComplexTypeInfo = ComplexTypeInfo<PrecisionType, Backend::CPU>;

template <typename PrecisionType>
struct CPUComplexType: public at::CPUTypeDefault {

    CPUComplexType()
    : CPUTypeDefault(CPUTensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}

    ScalarType scalarType() const override;
    Backend backend() const override;
    const char * toString() const override;
    size_t elementSizeInBytes() const override;
    TypeID ID() const override;
    Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
    Tensor & _s_copy_from(const Tensor & self, Tensor & dst, bool non_blocking) const override;

    Tensor tensor(Storage storage, int64_t storageOffset, IntList sizes, IntList strides) const override;
    Tensor tensor(IntList sizes, IntList strides) const override;
    Tensor tensor(IntList size) const override;
};

} // namespace at

#include "CPUComplexTypeImpl.h"

#endif // CPUComplexType_H