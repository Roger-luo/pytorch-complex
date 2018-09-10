#include "CPUComplexType.h"
#include "Utils.h"

namespace at {

template <typename PT>
ScalarType CPUComplexType<PT>::scalarType() const {
    return CPUComplexTypeInfo<PT>::scalar_type;
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
    auto self = c10::make_intrusive<TensorImpl, UndefinedTensor>(
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
        scalarTypeToDataType(CPUComplexTypeInfo<PT>::scalar_type),
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
        scalarTypeToDataType(CPUComplexTypeInfo<PT>::scalar_type),
        0,
        getCPUAllocator(),
        /* resizable */ true)};

    // make tensor
    Tensor t{c10::make_intrusive<TensorImpl, UndefinedTensor>(
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

// template <typename PT>
// Tensor & CPUComplexType<PT>::set_(Tensor & self, )

template <>
inline const char * CPUComplexType<float>::toString() const {
    return "CPUComplexTensor<float>";
}

template <>
inline const char * CPUComplexType<double>::toString() const {
    return "CPUComplexType<double>";
}

} // at