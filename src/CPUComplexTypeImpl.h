#include "CPUComplexType.h"
#include "Utils.h"
#include "ComplexTensorApply.h"
#include "SIMD/SIMD.h"

namespace at {

template <typename PT>
ScalarType CPUComplexType<PT>::scalarType() const {
    return CPUComplexTypeInfo<PT>::scalar_type;
}

template <typename PT>
caffe2::TypeMeta CPUComplexType<PT>::typeMeta() const {
    return scalarTypeToTypeMeta(CPUComplexTypeInfo<PT>::scalar_type);
}

template <typename PT>
Backend CPUComplexType<PT>::backend() const {
    return Backend::CPU;
}

template <typename PT>
TypeID CPUComplexType<PT>::ID() const {
    return CPUComplexTypeInfo<PT>::type_id;
}

template <typename PT>
size_t CPUComplexType<PT>::elementSizeInBytes() const {
    return 2 * sizeof(PT);
}

template <typename PT>
int64_t CPUComplexType<PT>::storage_offset(const Tensor & self) const {
    // DeviceGuard omitted
    auto self_ = checked_tensor_unwrap(self, "self", 1, false, Backend::CPU, CPUComplexTypeInfo<PT>::scalar_type);
    return static_cast<int64_t>(self_->storage_offset());
}

template <typename PT>
Tensor CPUComplexType<PT>::tensor(Storage storage, int64_t storageOffset, IntList sizes, IntList strides) const {
    // DeviceGuard omitted

    // checks
    if (strides.data()) {AT_CHECK(sizes.size() == strides.size(), "number of sizes and strides must match");}
    auto storage_ = checked_storage(storage, "storage", 1, DeviceType::CPU, at::scalarTypeToDataType(CPUComplexTypeInfo<PT>::scalar_type));

    // make tensor
    auto self = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
        /* storage */ std::move(storage_),
        /* tensor type id */ at::CPUTensorId(),
        /* is_variable */ false);

    /* storageOffset */
    if(storageOffset < 0)
        THError("Tensor: invalid storage offset");
    self->set_storage_offset(storageOffset);

    // set size
    self->set_sizes_and_strides(sizes, strides);
    return Tensor(self);
}

template <typename PT>
Tensor CPUComplexType<PT>::tensor(IntList sizes, IntList strides) const {
    // DeviceGuard omitted
    int64_t numel = 1;
    for (auto s : sizes) {
        numel *= s;
    }

    Storage s{c10::make_intrusive<StorageImpl>(
        scalarTypeToTypeMeta(CPUComplexTypeInfo<PT>::scalar_type),
        numel,
        getCPUAllocator(),
        /* resizable */ true)};

    return tensor(s, 0, sizes, strides);
}

template <typename PT>
Tensor CPUComplexType<PT>::tensor(IntList size) const {
    auto _strides = calculate_contiguous_stride(size);
    IntList strides{_strides};
    return tensor(size, strides);
}

template <typename PT>
Tensor CPUComplexType<PT>::tensor() const {
    Storage s{c10::make_intrusive<StorageImpl>(
        scalarTypeToTypeMeta(CPUComplexTypeInfo<PT>::scalar_type),
        0,
        getCPUAllocator(),
        /* resizable */ true)};

    // make tensor
    Tensor t{c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
        /* storage */ std::move(s),
        /* tensor type id */ at::CPUTensorId(),
        /* is_variable */ false)};

    return t;
}

template <typename PT>
Tensor & CPUComplexType<PT>::set_(Tensor & self, Storage source, int64_t storage_offset, IntList sizes, IntList strides) const {
    // DeviceGuard omitted
    auto self_ = checked_tensor_unwrap(self,"self",1, false, Backend::CPU, CPUComplexTypeInfo<PT>::scalar_type);
    auto source_ = checked_storage(source,"source",2, DeviceType::CPU, at::scalarTypeToDataType(CPUComplexTypeInfo<PT>::scalar_type));

    StorageImpl *storage_ptr = source.unsafeGetStorageImpl();
    StorageImpl *self_storage_ptr = self_->storage_.unsafeGetStorageImpl();

    if (self_storage_ptr != storage_ptr)
    {
        if (!self_storage_ptr) {
            AT_ERROR("Tensor: invalid null storage");
        }

        // steal storage
        c10::raw::intrusive_ptr::incref(storage_ptr);
        self_->storage_ = Storage(storage_ptr);
    }

    if (storage_offset < 0)
        AT_ERROR("Tensor: invalid storage offset");

    self_->set_storage_offset(storage_offset);
    // set size
    self_->set_sizes_and_strides(sizes, strides);
    self_->maybe_zero_dim(false);
    return self;
}

/* NOTE: This C macro here mainly because ISO C++03 14.2/4 
 * 
 * When the name of a member template specialization appears after . or -> in a postfix-expression,
 * or after nested-name-specifier in a qualified-id, and the postfix-expression or qualified-id 
 * explicitly depends on a template-parameter (14.6.2), the member template name must be prefixed 
 * by the keyword template. Otherwise the name is assumed to name a non-template.
 * 
 * We have TENSOR->data inside the TH_TENSOR_APPLY macro without template, but our implementation via
 * C++ templates for generic complex number requires a template keyword for data.
 * 
 * This is just a workaround, when everything moves to ATen/native, we can use the new protocals.
 */
#define IMPLEMENT_FILL(PrecisionType) \
template <> \
Tensor & CPUComplexType<PrecisionType>::_fill_(Tensor & self, Scalar value) const { \
    const DeviceGuard device_guard(self); \
    auto self_ = checked_tensor_unwrap(self,"self",1, false, Backend::CPU, CPUComplexTypeInfo<PrecisionType>::scalar_type); \
    auto value_ = value.to<std::complex<PrecisionType>>(); \
\
    if(self_->is_contiguous() || is_transposed(self_)) { \
        TH_TENSOR_APPLY_CONTIG(std::complex<PrecisionType>, self_, simd::Default<std::complex<PrecisionType>>::fill(self__data, value_, self__len); ); \
    } else { \
        TH_TENSOR_APPLY(std::complex<PrecisionType>, self_, \
            if (self__stride == 1) { \
                simd::Default<std::complex<PrecisionType>>::fill(self__data, value_, self__size); \
	            self__i = self__size; \
	            self__data += self__stride * self__size; \
	            break; \
            } else { \
                *self__data = value_; \
            } \
        ); \
    } \
\
    return self; \
}

IMPLEMENT_FILL(double)
IMPLEMENT_FILL(float)

template <typename PT>
Tensor &CPUComplexType<PT>::_fill_(Tensor &self, const Tensor & value) const {
    const DeviceGuard device_guard(self);
    if (value.dim() == 0) {
        return static_cast<const TypeExtendedInterface*>(this)->_fill_(self, at::_local_scalar(value));
    }
    AT_ERROR("_fill_ only supports a 0-dimensional value tensor, but got tensor "
        "with ", value.dim(), " dimension(s).");
}

template <typename PT>
Tensor & CPUComplexType<PT>::th_zero_(Tensor & self) const {
    return _fill_(self, Scalar(0.0));
}

template <typename PT>
Tensor &CPUComplexType<PT>::native_zero_(Tensor & self) const {
    return _fill_(self, Scalar(0.0));
}

template <typename PT>
void *CPUComplexType<PT>::data_ptr(const Tensor & self) const {
    auto self_ = checked_tensor_unwrap(self,"self",1, false, Backend::CPU, CPUComplexTypeInfo<PT>::scalar_type);
    return self_->template data<std::complex<PT>>();
}

template <typename PT>
Scalar CPUComplexType<PT>::_local_scalar_dense(const Tensor & self) const {
    const DeviceGuard device_guard(self);
    const auto& self_ty = *this;
    (void)self_ty;
    return at::native::_local_scalar_dense_cpu(/* actuals */ self);
}

template <>
inline const char * CPUComplexType<float>::toString() const {
    return "CPUComplexTensor<float>";
}

template <>
inline const char * CPUComplexType<double>::toString() const {
    return "CPUComplexType<double>";
}

// Linear Algebra

template <typename PT>
Tensor & CPUComplexType<PT>::_mv_out(Tensor & result, const Tensor & self, const Tensor & vec) const {
    AT_ERROR("_mv_out not implemented");
}

template <typename PT>
Tensor CPUComplexType<PT>::_mv(const Tensor & self, const Tensor & vec) const {
    AT_ERROR("_mv_out not implemented");
}

template <typename PT>
Tensor CPUComplexType<PT>::_mm(const Tensor &self, const Tensor &mat2) const {
    AT_ERROR("_mm not implemented");
}

template <typename PT>
Tensor CPUComplexType<PT>::_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) const {
    AT_ERROR_("_mm_out not implemented");
}

} // at