#include "PBFCudaApi.cuh"

#include "Modules/SPH/KernelFunctions.cuh"

namespace VT_Physics::pbf {

#define CHECK_THREAD() \
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= d_data->particle_num)                     \
        return;                                         \
    auto p_i = d_nsParams->particleIndices_cuData[i];

#define FOR_EACH_NEIGHBOR_Pj() \
       auto neib_ind = p_i * d_nsConfig->maxNeighborNum;                        \
       for (unsigned int p_j = d_nsParams->neighbors_cuData[neib_ind], t = 0;   \
            p_j != UINT_MAX && t < d_nsConfig->maxNeighborNum;                  \
            ++t, p_j = d_nsParams->neighbors_cuData[neib_ind + t])

#define CONST_VALUE(name) \
        d_data->name

#define DATA_VALUE(name, index) \
        d_data->name[index]

#define CUBIC_KERNEL_VALUE() \
        sph::cubic_value(pos_i - pos_j, d_data->h)

#define CUBIC_KERNEL_GRAD() \
        sph::cubic_gradient(pos_i - pos_j, d_data->h)

}

namespace VT_Physics::pbf { // cuda kernels
    __global__ void
    init_cuda(Data *d_data) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= d_data->particle_num)
            return;

        DATA_VALUE(volume, i) = CONST_VALUE(fPart_rest_volume);
        DATA_VALUE(dx, i) *= 0;
        DATA_VALUE(mass, i) = CONST_VALUE(fPart_rest_density) * CONST_VALUE(fPart_rest_volume);
        DATA_VALUE(error, i) = 0;
        DATA_VALUE(error_grad, i) *= 0;
    }

    __global__ void
    compute_rigid_particle_volume_cuda(Data *d_data,
                                       UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_BOUNDARY)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        float delta = 0;
        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);

            if (DATA_VALUE(mat, p_j) == DATA_VALUE(mat, p_i))
                delta += CUBIC_KERNEL_VALUE();
        }

        DATA_VALUE(volume, p_i) = 1.f / delta;

        DATA_VALUE(mass, p_i) = CONST_VALUE(bPart_rest_density) * DATA_VALUE(volume, p_i);
    }

    __global__ void
    compute_sph_density_and_error_cuda(Data *d_data,
                                       UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                       UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        DATA_VALUE(density_sph, p_i) *= 0;

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto m_j = DATA_VALUE(mass, p_j);

            DATA_VALUE(density_sph, p_i) += m_j * CUBIC_KERNEL_VALUE();
        }

        if (DATA_VALUE(density_sph, p_i) < CONST_VALUE(fPart_rest_density))
            DATA_VALUE(density_sph, p_i) = CONST_VALUE(fPart_rest_density);

        DATA_VALUE(error, p_i) = DATA_VALUE(density_sph, p_i) / CONST_VALUE(fPart_rest_density) - 1.f;
    }

    __global__ void
    update_lamb_cuda(Data *d_data,
                     UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                     UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        DATA_VALUE(error_grad, p_i) *= 0;
        DATA_VALUE(lamb, p_i) = 0;

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            auto error_grad_j = -1 / CONST_VALUE(fPart_rest_density) * wGrad;
            DATA_VALUE(error_grad, p_i) += wGrad;
            DATA_VALUE(lamb, p_i) += dot(error_grad_j, error_grad_j);
        }

        DATA_VALUE(error_grad, p_i) /= CONST_VALUE(fPart_rest_density);

        DATA_VALUE(lamb, p_i) += dot(DATA_VALUE(error_grad, p_i), DATA_VALUE(error_grad, p_i));

        DATA_VALUE(lamb, p_i) = -DATA_VALUE(error, p_i) / (DATA_VALUE(lamb, p_i) + 1e-6f);
    }

    __global__ void
    compute_dx_cuda(Data *d_data,
                    UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                    UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        DATA_VALUE(dx, p_i) *= 0;

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto wGrad = CUBIC_KERNEL_GRAD();

            DATA_VALUE(dx, p_i) += (DATA_VALUE(lamb, p_i) + DATA_VALUE(lamb, p_j)) * wGrad;
        }

        DATA_VALUE(dx, p_i) /= CONST_VALUE(fPart_rest_density);
    }

    __global__ void
    apply_ext_force_cuda(Data *d_data,
                         UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                         UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(vel, p_i) += CONST_VALUE(gravity) * CONST_VALUE(dt);
        DATA_VALUE(pos, p_i) += DATA_VALUE(vel, p_i) * CONST_VALUE(dt);
    }

    __global__ void
    apply_dx_cuda(Data *d_data,
                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        DATA_VALUE(pos, p_i) += DATA_VALUE(dx, p_i);
        DATA_VALUE(vel, p_i) += DATA_VALUE(dx, p_i) / CONST_VALUE(dt);
    }

    __global__ void
    XSPH_cuda(Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        CHECK_THREAD();

        if (DATA_VALUE(mat, p_i) != EPM_FLUID)
            return;

        auto pos_i = DATA_VALUE(pos, p_i);
        auto vel_i = DATA_VALUE(vel, p_i);
        DATA_VALUE(dx, p_i) *= 0;
        float3 dv = {0, 0, 0};

        FOR_EACH_NEIGHBOR_Pj() {
            auto pos_j = DATA_VALUE(pos, p_j);
            auto vel_j = DATA_VALUE(vel, p_j);

            dv += CONST_VALUE(XSPH_k) * DATA_VALUE(volume, p_j) * (vel_j - vel_i) * CUBIC_KERNEL_VALUE();
        }

        DATA_VALUE(dx, p_i) += dv * CONST_VALUE(dt);
    }

    __device__ float3 rotate_vector_by_quaternion(const float3& v, const float4& q) {
        // q = (x,y,z,w)
        float4 q_norm = normalize(q);
        float3 u = make_float3(q_norm.x, q_norm.y, q_norm.z);
        float s = q_norm.w;

        // 右手系:  + 2*s*cross(u,v)
        // 左手系:  - 2*s*cross(u,v)   <- 使用左手系以和你的数据/客户端一致
        return 2.0f * dot(u, v) * u
             + (s*s - dot(u, u)) * v
             - 2.0f * s * cross(u, v);  // 改这里的符号
    }

    // 新的内核：根据局部坐标和世界变换来计算粒子当前的世界坐标和速度
    __global__ void update_rigid_body_positions_kernel(Data* d_data, RigidObjectData* rigid_obj) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= rigid_obj->particle_count) return;

        int p_i = rigid_obj->start_idx + i;

        // 1. 读取局部坐标
        float3 local_pos = rigid_obj->d_local_pos[i];

        // 2. 应用旋转和平移，计算新的世界坐标
        // new_pos = R * local_pos + T
        float3 new_pos = rotate_vector_by_quaternion(local_pos, rigid_obj->current_q) + rigid_obj->current_t;

        // 3. 读取旧的世界坐标以计算速度
        float3 old_pos = DATA_VALUE(pos, p_i);

        // 4. 计算速度 v = (P_new - P_old) / dt
        float3 vel = (new_pos - old_pos) * CONST_VALUE(inv_dt);

        // 5. 更新GPU上的世界坐标和速度
        DATA_VALUE(pos, p_i) = new_pos;
        DATA_VALUE(vel, p_i) = vel;
    }


    // 添加这个辅助函数
    __device__ static inline void atomicMax_f(float* address, float val) {
        int* address_as_int = (int*)address;
        int old = *address_as_int, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }

    __global__ void reduce_max_velocity_sq_kernel_part1(const float3* vel, float* partial_max_sq, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float vel_sq = vel[idx].x * vel[idx].x + vel[idx].y * vel[idx].y + vel[idx].z * vel[idx].z;
        // 使用我们自己实现的 atomicMax_f
        atomicMax_f(&partial_max_sq[blockIdx.x], vel_sq);
    }

    __global__ void reduce_max_velocity_sq_kernel_part2(float* partial_max_sq, float* final_max_sq, int num_blocks) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            float max_val = 0.0f;
            for (int i = 0; i < num_blocks; ++i) {
                max_val = fmaxf(max_val, partial_max_sq[i]);
            }
            *final_max_sq = max_val;
        }
    }

    void compute_max_velocity_sq(Data* data, float* d_max_vel_sq_out) {
        // 临时存储每个 block 的最大值
        float* d_partial_max_sq;
        cudaMalloc(&d_partial_max_sq, data->block_num * sizeof(float));
        cudaMemset(d_partial_max_sq, 0, data->block_num * sizeof(float));

        // 第一阶段: 每个 block 内部计算最大值
        reduce_max_velocity_sq_kernel_part1<<<data->block_num, data->thread_num>>>(data->vel, d_partial_max_sq, data->particle_num);

        // 第二阶段: 从各 block 的结果中找到全局最大值
        reduce_max_velocity_sq_kernel_part2<<<1, 1>>>(d_partial_max_sq, d_max_vel_sq_out, data->block_num);

        cudaFree(d_partial_max_sq);
    }
}

namespace VT_Physics::pbf { // host invoke api
    __host__ void
    init_data(Data *h_data,
              Data *d_data,
              UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
              UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // init_data_cuda
        init_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data);

        // compute_rigid_particle_volume_cuda
        compute_rigid_particle_volume_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                      d_nsConfig,
                                                                                      d_nsParams);
    }

    __host__ void
    compute_sph_density_and_error(Data *h_data,
                                  Data *d_data,
                                  UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                                  UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // compute_sph_density_and_lamb_cuda
        compute_sph_density_and_error_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                                      d_nsConfig,
                                                                                      d_nsParams);
    }

    __host__ void
    update_lamb(Data *h_data,
                Data *d_data,
                UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // update_lamb_cuda
        update_lamb_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                    d_nsConfig,
                                                                    d_nsParams);
    }

    __host__ void
    compute_dx(Data *h_data,
               Data *d_data,
               UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
               UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // compute_dx_cuda
        compute_dx_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                   d_nsConfig,
                                                                   d_nsParams);
    }

    __host__ void
    apply_ext_force(Data *h_data,
                    Data *d_data,
                    UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                    UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // apply_ext_force_cuda
        apply_ext_force_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                        d_nsConfig,
                                                                        d_nsParams);
    }

    __host__ void
    apply_dx(Data *h_data,
             Data *d_data,
             UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
             UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // apply_dx_cuda
        apply_dx_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                 d_nsConfig,
                                                                 d_nsParams);
    }

    __host__ void
    post_correct(Data *h_data,
                 Data *d_data,
                 UGNS::UniformGirdNeighborSearcherConfig *d_nsConfig,
                 UGNS::UniformGirdNeighborSearcherParams *d_nsParams) {
        // XSPH_cuda
        XSPH_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                             d_nsConfig,
                                                             d_nsParams);

        // apply_dx_cuda
        apply_dx_cuda<<<h_data->block_num, h_data->thread_num>>>(d_data,
                                                                 d_nsConfig,
                                                                 d_nsParams);
    }

    __host__ void
    update_rigid_body_positions(Data *d_data, RigidObjectData* d_rigid_data, int num_rigid_objects) {
        // 为每个刚体对象启动一个内核
        for (int i = 0; i < num_rigid_objects; ++i) {
            // 从设备端数组中获取单个对象的信息
            RigidObjectData rigid_obj;
            cudaMemcpy(&rigid_obj, &d_rigid_data[i], sizeof(RigidObjectData), cudaMemcpyDeviceToHost);

            if (rigid_obj.particle_count > 0) {
                unsigned int block_num = (rigid_obj.particle_count + 1024 - 1) / 1024; // 使用固定的线程数
                update_rigid_body_positions_kernel<<<block_num, 1024>>>(d_data, &d_rigid_data[i]);
            }
        }
    }
}