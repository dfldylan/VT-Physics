#pragma once

// 为空定义 CUDA 装饰符，保证非 NVCC 情况下可编译
#include "Core/Math/cuda_compat.h"

// 仅在非 NVCC 编译路径下生效（NVCC 下直接走 CUDA 自带类型）
#ifndef __CUDACC__

// 若系统已安装 CUDA 且可找到 <vector_types.h>，优先使用官方定义，避免后续再次包含引发重定义
#if defined(__has_include)
#  if __has_include(<vector_types.h>)
#    include <vector_types.h>
#    define VT_USING_CUDA_VECTOR_TYPES 1
#  endif
#endif

// 如果没有 CUDA SDK 或未找到 vector_types.h，则提供最小替代定义
#ifndef VT_USING_CUDA_VECTOR_TYPES
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };

struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct int4 { int x, y, z, w; };

inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

inline int2 make_int2(int x, int y) { return int2{x, y}; }
inline int3 make_int3(int x, int y, int z) { return int3{x, y, z}; }
inline int4 make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }
#endif // !VT_USING_CUDA_VECTOR_TYPES

#endif // __CUDACC__