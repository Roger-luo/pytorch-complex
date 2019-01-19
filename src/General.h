#ifndef GENERAL_H
#define GENERAL_H

#include <ATen/detail/ComplexHooksInterface.h>
#include <ATen/Type.h>
#include <ATen/CPUFloatType.h>

#include <TH/THTensor.hpp>
// #include "THTensorApply.h"
#include <TH/THTensorApply.h>

#include <c10/core/TensorImpl.h>
#include <ATen/CPUGenerator.h>
#include <ATen/Allocator.h>
#include <c10/core/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "ATen/Config.h"

#endif // GENERAL_H
