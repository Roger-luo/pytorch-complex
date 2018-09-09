#include <iostream>

template <typename T>
struct TypeInfo;

template <>
struct TypeInfo<double> {
    using scalar_t = double;
};


int main(int argc, char const *argv[])
{
    TypeInfo<double>::scalar_t *a = NULL;
    std::cout << a << std::endl;
    return 0;
}
