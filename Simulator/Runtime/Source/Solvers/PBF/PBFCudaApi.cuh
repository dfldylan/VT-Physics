/**
 * @brief Todo
 * @date 2024/11/7
 */

#ifndef VT_PHYSICS_PBFCUDAAPI_CUH
#define VT_PHYSICS_PBFCUDAAPI_CUH

#include "Solvers/PBF/PBFrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"
#include "Core/Math/helper_math_cu11.6.h"

namespace VT_Physics::pbf {

    __host__ void
    init_data(Data *h_data,
              Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    compute_sph_density_and_error(Data *h_data,
                                  Data *d_data,
                                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    update_lamb(Data *h_data,
                Data *d_data,
                UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    compute_dx(Data *h_data,
               Data *d_data,
               UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    apply_ext_force(Data *h_data,
                    Data *d_data,
                    UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                    UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    apply_dx(Data *h_data,
             Data *d_data,
             UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
             UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    post_correct(Data *h_data,
                 Data *d_data,
                 UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams);

    __host__ void
    update_rigid_body_positions(Data *d_data, RigidObjectData* d_rigid_data, int num_rigid_objects);

    /**
     * @brief 在 GPU 上计算所有粒子速度平方的最大值
     * @param data PBF 求解器的设备端数据指针
     * @param max_vel_sq_out 输出参数，用于存放计算出的最大速度平方
     */
    void compute_max_velocity_sq(Data* data, float* max_vel_sq_out);

}

#endif //VT_PHYSICS_PBFCUDAAPI_CUH
