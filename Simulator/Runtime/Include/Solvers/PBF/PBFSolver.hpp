/**
 * @brief Todo
 * @date 2024/10/28
 */

#ifndef VT_PHYSICS_PBFSOLVER_HPP
#define VT_PHYSICS_PBFSOLVER_HPP

#include <vector>
#include <set>

#include "Framework/Solver.hpp"
#include "PBFrtData.hpp"
#include "Modules/NeighborSearch/UGNS/UniformGridNeighborSearch.hpp"

// 移除对 CUDA 头的直接依赖，避免服务器侧 MSVC 解析 helper_math
// #include "PBFCudaApi.cuh"

// 为非 CUDA 编译提供 float3 等最小类型
#include "Core/Math/cuda_vector_compat.h"

// 如需用到 __host__/__device__ 等，也可包含上一轮加的空宏头
#include "Core/Math/cuda_compat.h"

// 在命名空间内前置声明 CUDA Data（在 .cpp 中完整定义）
namespace VT_Physics { namespace pbf { struct Data; } }

namespace VT_Physics::pbf {

    inline const std::vector<std::string> PBFConfigRequiredKeys = {
            "animationTime",
            "timeStep",
            "particleRadius",
            "simSpaceLB",
            "simSpaceSize",
            "maxNeighborNum",
            "iterationNum",
            "XSPH_k",
            "fPartRestDensity",
            "bPartRestDensity"
    };

    inline const std::vector<std::string> PBFConfigOptionalKeys = {
            "enable",
            "gravity",
    };

    inline const std::vector<std::string> PBFSolverObjectComponentConfigRequiredKeys = {
            "solverType",
            "exportFlag",
            "velocityStart",
            "colorStart",
    };

    inline const std::set<uint8_t> PBFSolverSupportedMaterials = {
            EPM_FLUID,
            EPM_BOUNDARY
    };

    class PBFSolver : public Solver {
    public:
        PBFSolver() = delete;

        PBFSolver(uint32_t cudaThreadSize);

        virtual ~PBFSolver() override = default;

        virtual json getSolverConfigTemplate() const override;

        virtual bool setConfig(json config) override;

        virtual bool setConfigByFile(std::string config_file) override;

        virtual json getSolverObjectComponentConfigTemplate() override;

        virtual bool initialize() override;

        virtual bool run() override;

        virtual bool tickNsteps(uint32_t n) override;

        virtual bool attachObject(Object *obj) override;

        virtual bool attachObjects(std::vector<Object *> objs) override;

        virtual bool reset() override;

        virtual void destroy() override;

        // --- Lightweight read-back APIs for service ---
        void fetchAllParticles(std::vector<float3>& outPos, std::vector<float3>& outVel);
        void getAttachedObjectRanges(std::vector<int>& start, std::vector<int>& end);
        void setTimeStep(float dt);

    protected:
        virtual bool tick() override;

        virtual bool checkConfig() const override;

    private:
        void exportData();

    private:
        json m_configData;
        Data *m_host_data{nullptr};
        Data *m_device_data{nullptr};
        bool m_isInitialized{false};
        bool m_isCrashed{false};
        uint32_t m_frameCount{0};
        uint32_t m_outputFrameCount{0};
        bool m_doExportFlag{false};
        std::vector<Object *> m_attached_objs;
        UGNS::UniformGirdNeighborSearcher m_neighborSearcher;

        std::vector<float3> m_host_pos;
        std::vector<float3> m_host_vel;
        std::vector<int> m_host_mat;
        std::vector<float3> m_host_color;

        bool m_enableCFL{false}; // 添加一个开关来控制是否启用CFL
        float* m_d_max_vel_sq{nullptr}; // 用于在设备端存储最大速度平方
        float m_cfl_number{0.4f}; // CFL 安全系数
        float m_min_dt{0.0001f}; // 最小时间步，防止除以零或dt过小
        float m_max_dt{0.04f}; // 最大时间步，例如对应60FPS
    };
}

#endif //VT_PHYSICS_PBFSOLVER_HPP
