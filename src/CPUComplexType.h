#ifndef CPUComplexType_H
#define CPUComplexType_H

#include "General.h"
#include "ComplexTypeInfo.h"

namespace at {

template <typename PrecisionType>
struct CPUComplexType: public at::CPUTypeDefault {

    CPUComplexType()
    : CPUTypeDefault(CPUTensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}

    virtual ScalarType scalarType() const override;
    virtual caffe2::TypeMeta typeMeta() const override;
    Backend backend() const override;
    const char * toString() const override;
    size_t elementSizeInBytes() const override;
    TypeID ID() const override;
    Tensor & s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const override;
    Tensor _s_copy_from(const Tensor & self, const Tensor & dst, bool non_blocking) const override;

    // Tensor & resize_(Tensor & self, IntList size) const override;

    /*
    Tensor _th_tensor(Storage storage, int64_t storageOffset, IntList sizes, IntList strides) const override;
    Tensor _th_tensor(IntList sizes, IntList strides) const override;
    */
    Tensor empty(IntList size, const TensorOptions & options) const override;
    /*
    Tensor tensor() const override;
    */

    Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntList size, IntList stride) const override;
    // Tensor & set_(Tensor & self, Storage source) const override;
    // Tensor & set_(Tensor & self, const Tensor & source) const override;
    // Tensor & set_(Tensor & self) const override;

    Tensor & cat_out(Tensor & self, TensorList tensors, int64_t dim) const override;
    Tensor cat(TensorList tensors, int64_t dim) const override;

    Tensor & fill_(Tensor & self, Scalar value) const override;
    Tensor & fill_(Tensor & self, const Tensor & value) const override;

    Tensor & zero_(Tensor & self) const override;
    Tensor & native_zero_(Tensor & self) const override;
    void* data_ptr(const Tensor & self) const override;
    Scalar _local_scalar_dense(const Tensor & self) const override;

    // LinearAlgebra
    Tensor & mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const override;
    Tensor mv(const Tensor & self, const Tensor & vec) const override;
    Tensor & mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const override;
    Tensor mm(const Tensor & self, const Tensor & mat2) const override;
};

} // namespace at

#include "CPUComplexTypeImpl.h"
#include "CPUComplexCopy.h"

#endif // CPUComplexType_H
