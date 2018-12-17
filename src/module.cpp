#include <torch/extension.h>
#include "CPUComplexType.h"

namespace at {

struct ComplexHooks : public at::ComplexHooksInterface {
    ComplexHooks(ComplexHooksArgs) {};
    void registerComplexTypes(Context* context) const override {
        context->registerType(Backend::CPU, CPUComplexTypeInfo<float>::scalar_type, new CPUComplexType<float>());
        context->registerType(Backend::CPU, CPUComplexTypeInfo<double>::scalar_type, new CPUComplexType<double>());
    }
};


REGISTER_COMPLEX_HOOKS(ComplexHooks);

}

// create the extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("cpptest", &cpptest, "cpp test");
}
