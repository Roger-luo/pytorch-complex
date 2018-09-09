#ifndef GENERAL_H
#define GENERAL_H

#include <ATen/detail/ComplexHooksInterface.h>
#include <ATen/Type.h>
#include <ATen/CPUFloatType.h>

#include <TH/THTensor.hpp>
#include <TH/THTensorApply.h>

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

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "ATen/Config.h"

#endif // GENERAL_H