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

// template <typename PT>
// Tensor & CPUComplexType<PT>::resize_(Tensor & self, IntList size) const {

// }

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



template <>
inline const char * CPUComplexType<float>::toString() const {
    return "CPUComplexTensor<float>";
}

template <>
inline const char * CPUComplexType<double>::toString() const {
    return "CPUComplexType<double>";
}

} // at