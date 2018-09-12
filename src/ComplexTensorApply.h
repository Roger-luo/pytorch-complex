#include "ComplexTypeInfo.h"

#ifndef NAN
  #define NAN (nan(NULL))
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define HYPER_TH_OMP_OVERHEAD_THRESHOLD 2000
#define ORDIN_TH_OMP_OVERHEAD_THRESHOLD 20000
#define UNCERTAIN_TH_OMP_OVERHEAD_THRESHOLD 50000
#define TH_OMP_OVERHEAD_THRESHOLD 100000

#ifdef _OPENMP

#ifndef _WIN32
#define PRAGMA(P) _Pragma(#P)
#else
#define PRAGMA(P) __pragma(P)
#endif

#define TH_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  int inOmp = omp_in_parallel(); \
  ptrdiff_t TH_TENSOR_size = TENSOR->numel(); \
  PRAGMA(omp parallel if ((TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD) && (!inOmp))) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE *TENSOR##_data = TENSOR->template data<TYPE>() + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY_CONTIG(TYPE, TENSOR, CODE) \
{ \
  TYPE *TENSOR##_data = TENSOR->template data<TYPE>(); \
  ptrdiff_t TENSOR##_len = TENSOR->numel(); \
  CODE \
}
#endif

#ifdef _OPENMP
#define TH_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  int inOmp = omp_in_parallel(); \
  ptrdiff_t TH_TENSOR_size = TENSOR->numel(); \
  PRAGMA(omp parallel if ((TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD) && (!inOmp))) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = TENSOR1->template data<TYPE1>() + TH_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = TENSOR2->template data<TYPE2>() + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY2_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, CODE) \
{ \
  TYPE1 *TENSOR1##_data = TENSOR1->template data<TYPE1>(); \
  TYPE2 *TENSOR2##_data = TENSOR2->template data<TYPE2>(); \
  ptrdiff_t TENSOR1##_len = TENSOR1->numel(); \
  CODE \
}
#endif

#ifdef _OPENMP
#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  int inOmp = omp_in_parallel(); \
  ptrdiff_t TH_TENSOR_size = TENSOR1->numel(); \
  PRAGMA(omp parallel if ((TH_TENSOR_size > TH_OMP_OVERHEAD_THRESHOLD) && (!inOmp))) \
  { \
    size_t num_threads = omp_get_num_threads(); \
    size_t tid = omp_get_thread_num(); \
    ptrdiff_t TH_TENSOR_offset = tid * (TH_TENSOR_size / num_threads); \
    ptrdiff_t TH_TENSOR_end = tid == num_threads - 1 ? TH_TENSOR_size : \
      TH_TENSOR_offset + TH_TENSOR_size / num_threads; \
    ptrdiff_t TENSOR1##_len = TH_TENSOR_end - TH_TENSOR_offset; \
    TYPE1 *TENSOR1##_data = TENSOR1->template data<TYPE1>() + TH_TENSOR_offset; \
    TYPE2 *TENSOR2##_data = TENSOR2->template data<TYPE2>() + TH_TENSOR_offset; \
    TYPE3 *TENSOR3##_data = TENSOR3->template data<TYPE3>() + TH_TENSOR_offset; \
    CODE \
  } \
}
#else
#define TH_TENSOR_APPLY3_CONTIG(TYPE1, TENSOR1, TYPE2, TENSOR2, TYPE3, TENSOR3, CODE) \
{ \
  TYPE1 *TENSOR1##_data = TENSOR1->template data<TYPE1>(); \
  TYPE2 *TENSOR2##_data = TENSOR2->template data<TYPE2>(); \
  TYPE3 *TENSOR3##_data = TENSOR3->template data<TYPE3>(); \
  ptrdiff_t TENSOR1##_len = TENSOR1->numel(); \
  CODE \
}
#endif
