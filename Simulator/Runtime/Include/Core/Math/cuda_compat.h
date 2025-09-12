#pragma once

// 仅在非 NVCC/非 CUDA 设备编译时提供空宏，避免 MSVC 语法错误
#ifndef __CUDACC__
    #ifndef __host__
    #define __host__
    #endif
    #ifndef __device__
    #define __device__
    #endif
    #ifndef __forceinline__
    #define __forceinline__ inline
    #endif
    #ifndef __constant__
    #define __constant__
    #endif
    #ifndef __shared__
    #define __shared__
    #endif
    #ifndef __align__
    #define __align__(x)
    #endif
#endif