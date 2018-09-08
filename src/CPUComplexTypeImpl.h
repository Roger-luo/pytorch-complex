#include "CPUComplexType.h"

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
    return sizeof(PT);
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
    // TODO: Upstream this
    int64_t numel = 1;
    for (auto s : size) {
        numel *= s;
    }

    Storage s{c10::make_intrusive<StorageImpl>(
        scalarTypeToDataType(CPUComplexTypeInfo<PT>::scalar_type),
        numel,
        getCPUAllocator(),
        /* resizable */ true)};
    Tensor t{c10::make_intrusive<TensorImpl, UndefinedTensor>(
        std::move(s),
        at::CPUTensorId(),
        /* is_variable */ false)};
    return t;
}


template <typename PT>
Tensor & CPUComplexType<PT>::s_copy_(Tensor & dst, const Tensor & src, bool non_blocking) const {
    AT_ERROR("not yet supported");
}

template <typename PT>
Tensor & CPUComplexType<PT>::_s_copy_from(const Tensor & src, Tensor & dst, bool non_blocking) const {
    AT_ERROR("not yet supported");
}

template <>
inline const char * CPUComplexType<float>::toString() const {
    return "CPUComplexType<float>";
}

template <>
inline const char * CPUComplexType<double>::toString() const {
    return "CPUComplexType<double>";
}

} // at