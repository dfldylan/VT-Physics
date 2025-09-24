#include "Solvers/PBF/PBFSolver.hpp"

#include <string>
#include <cmath>

#include "Logger/Logger.hpp"
#include "Model/ExportUtil.hpp"
#include "JSON/JSONHandler.hpp"
#include "PBFCudaApi.cuh"

// 声明新的API函数
void update_rigid_body_positions(VT_Physics::pbf::Data* d_data, VT_Physics::pbf::RigidObjectData* d_rigid_data, int num_rigid_objects);
void compute_max_velocity_sq(VT_Physics::pbf::Data* data, float* max_vel_sq_out);

namespace VT_Physics::pbf {

    PBFSolver::PBFSolver(uint32_t cudaThreadSize) {
        m_configData = JsonHandler::loadPBFConfigTemplateJson();
        m_host_data = new Data();
        m_host_data->thread_num = cudaThreadSize;
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("PBFSolver fail to allocate memory for device data.")
            delete m_host_data;
            return;
        }
        cudaMalloc((void**)&m_d_max_vel_sq, sizeof(float)); // 分配内存

        LOG_INFO("PBFSolver Created.");
    }

    json PBFSolver::getSolverConfigTemplate() const {
        return JsonHandler::loadPBFConfigTemplateJson();
    }

    bool PBFSolver::setConfig(json config) {
        m_configData = config;
        if (m_configData.empty()) {
            LOG_ERROR("PBF config json is empty.");
            return false;
        }

        if (!checkConfig()) {
            LOG_ERROR("Invalid PBF config file.");
            return false;
        }

        auto pbf_config = m_configData["PBF"];
        m_host_data->dt = pbf_config["Required"]["timeStep"].get<float>();
        m_host_data->inv_dt = 1 / m_host_data->dt;
        m_host_data->inv_dt2 = m_host_data->inv_dt * m_host_data->inv_dt;
        m_host_data->cur_simTime = 0.f;
        m_host_data->particle_radius = pbf_config["Required"]["particleRadius"].get<float>();
        m_host_data->fPart_rest_density = pbf_config["Required"]["fPartRestDensity"].get<float>();
        m_host_data->bPart_rest_density = pbf_config["Required"]["bPartRestDensity"].get<float>();
        m_host_data->fPart_rest_volume = std::pow(2 * m_host_data->particle_radius, 3);
        m_host_data->h = m_host_data->particle_radius * 4;
        m_host_data->XSPH_k = pbf_config["Required"]["XSPH_k"].get<float>();
        m_host_data->gravity = make_float3(0.f, -9.8f, 0.f);

        if (pbf_config["Optional"]["enable"]) {
            auto g = pbf_config["Optional"]["gravity"].get<std::vector<float>>();
            m_host_data->gravity = make_float3(g[0], g[1], g[2]);
        }

        if (pbf_config["Optional"].contains("enableCFL")) {
            m_enableCFL = pbf_config["Optional"]["enableCFL"].get<bool>();
            if (m_enableCFL) {
                LOG_INFO("CFL condition enabled.");
            }
        }

        LOG_INFO("PBFSolver Configured.");
        return true;
    }

    bool PBFSolver::setConfigByFile(std::string config_file) {
        setConfig(JsonHandler::loadJson(config_file));
        return true;
    }

    json PBFSolver::getSolverObjectComponentConfigTemplate() {
        return JsonHandler::loadPBFObjectComponentConfigTemplateJson();
    }

    bool PBFSolver::initialize() {
        m_host_data->particle_num = m_host_pos.size();
        m_host_data->block_num = (m_host_data->particle_num + m_host_data->thread_num - 1) / m_host_data->thread_num;

        if (!ExportUtil::checkConfig(m_configData["EXPORT"])) {
            LOG_ERROR("PBFSolver export config not available.");
            return false;
        }

        auto ugns_config = JsonHandler::loadUGNSConfigTemplateJson();
        ugns_config["simSpaceLB"] = m_configData["PBF"]["Required"]["simSpaceLB"];
        ugns_config["simSpaceSize"] = m_configData["PBF"]["Required"]["simSpaceSize"];
        ugns_config["totalParticleNum"] = m_host_data->particle_num;
        ugns_config["maxNeighborNum"] = m_configData["PBF"]["Required"]["maxNeighborNum"];
        ugns_config["cuKernelBlockNum"] = m_host_data->block_num;
        ugns_config["cuKernelThreadNum"] = m_host_data->thread_num;
        ugns_config["gridCellSize"] = m_host_data->h;
        m_neighborSearcher.setConfig(ugns_config);

        // cudaMalloc and memcpy
        if (!m_host_data->malloc()) {
            LOG_ERROR("PBFSolver fail to allocate memory for rt data.");
            return false;
        }
        cudaMalloc((void **) &m_device_data, sizeof(Data));
        if (!m_neighborSearcher.malloc()) {
            return false;
        }

        cudaMemcpy(m_host_data->pos, m_host_pos.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->vel, m_host_vel.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->color, m_host_color.data(), m_host_data->particle_num * sizeof(float3),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_host_data->mat, m_host_mat.data(), m_host_data->particle_num * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(m_device_data, m_host_data, sizeof(Data), cudaMemcpyHostToDevice);

        m_neighborSearcher.update(m_host_data->pos);
        init_data(m_host_data,
                  m_device_data,
                  m_neighborSearcher.m_config_cuData,
                  m_neighborSearcher.m_params_cuData);

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("PBFSolver fail to initialize.")
            return false;
        }

        m_isInitialized = true;
        return true;
    }

    bool PBFSolver::run() {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        while (m_host_data->cur_simTime < m_configData["PBF"]["Required"]["animationTime"].get<float>()) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("PBFSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("PBFSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool PBFSolver::tickNsteps(uint32_t n) {
        if (!m_isInitialized) {
            if (!initialize()) {
                return false;
            }
        }

        for (uint32_t i = 0; i < n; ++i) {
            if (!m_isCrashed) {
                tick();
                exportData();
                LOG_WARNING("PBFSolver current frame: " + std::to_string(m_frameCount));
            } else {
                LOG_ERROR("PBFSolver is crashed.");
                return false;
            }
        }

        return true;
    }

    bool PBFSolver::attachObject(Object *obj) {
        if (static_cast<uint8_t>(obj->getObjectComponent()->getType()) > 20) {
            LOG_ERROR("PBFSolver only support particle geometry yet.");
            return false;
        }

        if (PBFSolverSupportedMaterials.count(obj->getObjectComponentConfig()["epmMaterial"]) == 0) {
            LOG_ERROR("PBFSolver unsupported material.");
            return false;
        }

        for (const auto &key: PBFSolverObjectComponentConfigRequiredKeys) {
            if (!obj->getSolverObjectComponentConfig().contains(key)) {
                LOG_ERROR(
                        "Object: " + std::to_string(obj->getID()) + " has no PBF object component config key: " + key);
                return false;
            }
        }

        if (obj->getSolverObjectComponentConfig()["solverType"] != static_cast<uint8_t>(SolverType::PBF)) {
            LOG_ERROR("Object: " + std::to_string(obj->getID()) + " has no PBF object component config.");
            return false;
        }

        if (!m_configData["EXPORT"]["SolverRequired"].contains("exportObjectStartIndex")) {
            m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"] = json::array();
            m_configData["EXPORT"]["SolverRequired"]["exportFlags"] = json::array();
        }

        m_configData["EXPORT"]["SolverRequired"]["exportObjectMaterials"].push_back(
                obj->getObjectComponentConfig()["epmMaterial"]);
        m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].push_back(m_host_pos.size());
        m_configData["EXPORT"]["SolverRequired"]["exportFlags"].push_back(
                obj->getSolverObjectComponentConfig()["exportFlag"].get<bool>());

        auto pos = obj->getObjectComponent()->getElements();
        auto part_num = pos.size() / 3;
        auto pos_f3ptr = reinterpret_cast<float3 *>(pos.data());

        auto vel_start = obj->getSolverObjectComponentConfig()["velocityStart"].get<std::vector<float>>();
        auto color_start = obj->getSolverObjectComponentConfig()["colorStart"].get<std::vector<float>>();

        m_host_pos.insert(m_host_pos.end(), pos_f3ptr, pos_f3ptr + part_num);
        m_host_vel.insert(m_host_vel.end(), part_num, make_float3(vel_start[0], vel_start[1], vel_start[2]));
        m_host_mat.insert(m_host_mat.end(), part_num, obj->getObjectComponentConfig()["epmMaterial"].get<int>());
        m_host_color.insert(m_host_color.end(), part_num, make_float3(color_start[0], color_start[1], color_start[2]));

        m_configData["EXPORT"]["SolverRequired"]["exportObjectEndIndex"].push_back(m_host_pos.size());
        m_attached_objs.push_back(obj);

        // 如果是刚体，则为其创建并存储局部坐标信息
        if (obj->getObjectComponentConfig()["epmMaterial"].get<int>() == EPM_BOUNDARY) {
            RigidObjectData rod;
            rod.start_idx = m_configData["EXPORT"]["SolverRequired"]["exportObjectStartIndex"].back().get<int>();
            rod.particle_count = part_num;
            
            // 分配并拷贝局部坐标到GPU
            cudaMalloc(&rod.d_local_pos, part_num * sizeof(float3));
            cudaMemcpy(rod.d_local_pos, pos_f3ptr, part_num * sizeof(float3), cudaMemcpyHostToDevice);
            
            // 添加到刚体列表中
            m_rigid_objects.push_back(rod);
        }

        return true;
    }

    bool PBFSolver::attachObjects(std::vector<Object *> objs) {
        for (auto obj: objs) {
            if (!attachObject(obj)) {
                return false;
            }
        }

        return true;
    }

    bool PBFSolver::reset() {
        m_configData = JsonHandler::loadPBFConfigTemplateJson();
        return true;
    }

    void PBFSolver::destroy() {
        m_host_data->free();
        delete m_host_data;
        cudaFree(m_device_data);
        if (m_d_max_vel_sq) { // 释放内存
            cudaFree(m_d_max_vel_sq);
            m_d_max_vel_sq = nullptr;
        }
        // 释放刚体相关的GPU内存
        for (auto& rod : m_rigid_objects) {
            if (rod.d_local_pos) {
                cudaFree(rod.d_local_pos);
                rod.d_local_pos = nullptr;
            }
        }
        m_rigid_objects.clear();
        m_attached_objs.clear();
        m_neighborSearcher.freeMemory();

        LOG_INFO("PBFSolver Destroyed.");
    }

    void PBFSolver::exportData() {
        static auto &exportConfig = m_configData["EXPORT"];
        if (exportConfig["SolverRequired"]["enable"] && m_doExportFlag) {
            if (exportConfig["SolverRequired"]["exportGroupPolicy"] == "MERGE") {
                std::set<int> mat_exported;
                for (int i = 0; i < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++i) {
                    if (exportConfig["SolverRequired"]["exportFlags"][i].get<bool>()) {
                        auto cur_mat = exportConfig["SolverRequired"]["exportObjectMaterials"][i].get<int>();
                        if (mat_exported.count(cur_mat) != 0) {
                            continue;
                        } else {
                            mat_exported.insert(cur_mat);
                            std::vector<float3> pos_merged;
                            std::vector<float3> color_merged;
                            for (int j = 0; j < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++j) {
                                if (exportConfig["SolverRequired"]["exportObjectMaterials"][j].get<int>() != cur_mat)
                                    continue;
                                int data_size = exportConfig["SolverRequired"]["exportObjectEndIndex"][j].get<int>() -
                                                exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>();
                                std::vector<float3> pos_tmp(data_size);
                                std::vector<float3> color_tmp(data_size);
                                cudaMemcpy(pos_tmp.data(),
                                           m_host_data->pos +
                                           exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>(),
                                           data_size * sizeof(float3),
                                           cudaMemcpyDeviceToHost);
                                cudaMemcpy(color_tmp.data(),
                                           m_host_data->color +
                                           exportConfig["SolverRequired"]["exportObjectStartIndex"][j].get<int>(),
                                           data_size * sizeof(float3),
                                           cudaMemcpyDeviceToHost);
                                pos_merged.insert(pos_merged.end(), pos_tmp.begin(), pos_tmp.end());
                                color_merged.insert(color_merged.end(), color_tmp.begin(), color_tmp.end());
                            }
                            auto exportConfig_tmp = exportConfig;
                            exportConfig_tmp["Common"]["exportTargetDir"] =
                                    exportConfig_tmp["Common"]["exportTargetDir"].get<std::string>() + "/" +
                                    EPMString.at(cur_mat);
                            exportConfig_tmp["Common"]["exportFilePrefix"] = std::to_string(m_outputFrameCount);
                            ExportUtil::exportData(exportConfig_tmp,
                                                   pos_merged);
                        }
                    }
                }
            } else if (exportConfig["SolverRequired"]["exportGroupPolicy"] == "SPLIT") {
                for (int i = 0; i < exportConfig["SolverRequired"]["exportObjectMaterials"].size(); ++i) {
                    if (exportConfig["SolverRequired"]["exportFlags"][i].get<bool>()) {
                        int data_size = exportConfig["SolverRequired"]["exportObjectEndIndex"][i].get<int>() -
                                        exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>();
                        std::vector<float3> pos_tmp(data_size);
                        std::vector<float3> color_tmp(data_size);
                        cudaMemcpy(pos_tmp.data(),
                                   m_host_data->pos +
                                   exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>(),
                                   data_size * sizeof(float3),
                                   cudaMemcpyDeviceToHost);
                        cudaMemcpy(color_tmp.data(),
                                   m_host_data->color +
                                   exportConfig["SolverRequired"]["exportObjectStartIndex"][i].get<int>(),
                                   data_size * sizeof(float3),
                                   cudaMemcpyDeviceToHost);
                        auto exportConfig_tmp = exportConfig;
                        exportConfig_tmp["Common"]["exportTargetDir"] += "/" + m_attached_objs[i]->getName();
                        exportConfig_tmp["Common"]["exportFilePrefix"] += std::to_string(m_outputFrameCount);
                        ExportUtil::exportData(exportConfig_tmp,
                                               pos_tmp,
                                               color_tmp);
                    }
                }
            } else {
                LOG_ERROR("PBFSolver exportGroupPolicy" +
                          exportConfig["SolverRequired"]["exportGroupPolicy"].get<std::string>() + " not supported.");
            }

            m_doExportFlag = false;
        }
    }

    void PBFSolver::setTimeStep(float dt) {
        if (dt <= 0) return;
        m_host_data->dt = dt;
        m_host_data->inv_dt = 1.f / dt;
        m_host_data->inv_dt2 = m_host_data->inv_dt * m_host_data->inv_dt;
        // keep json consistent for potential users
        m_configData["PBF"]["Required"]["timeStep"] = dt;
    }
    
    void PBFSolver::applyRigidBodyTransform(int start_idx, int end_idx, const float q[4], const float t[3]) {
        // 移除 m_isInitialized 检查，允许在任何时候更新变换状态
        // if (!m_isInitialized) return;

        // 查找对应的刚体对象并更新其变换状态
        for (auto& rod : m_rigid_objects) {
            if (rod.start_idx == start_idx && (rod.start_idx + rod.particle_count) == end_idx) {
                // q from python is [w, x, y, z]. C++ float4 is (x, y, z, w)
                rod.current_q = make_float4(q[1], q[2], q[3], q[0]);
                rod.current_t = make_float3(t[0], t[1], t[2]);
                break; // 找到并更新后即可退出
            }
        }
    }

    bool PBFSolver::tick() {
        // 如果启用了CFL，在tick开始时动态计算并更新dt
        if (m_enableCFL) {
            // 1. 计算最大速度平方
            compute_max_velocity_sq(m_host_data, m_d_max_vel_sq);

            // 2. 将结果从GPU拷贝回CPU
            float h_max_vel_sq = 0.0f;
            cudaMemcpy(&h_max_vel_sq, m_d_max_vel_sq, sizeof(float), cudaMemcpyDeviceToHost);

            // 3. 计算新的dt
            float new_dt = m_host_data->dt;
            if (h_max_vel_sq > 1e-9) { // 避免除以零
                float max_vel = std::sqrt(h_max_vel_sq);
                new_dt = m_cfl_number * (m_host_data->particle_radius * 2.0f) / max_vel; // 使用直径作为特征长度更稳定
            }

            // 4. 限制dt在合理范围内并更新
            new_dt = std::max(m_min_dt, std::min(m_max_dt, new_dt));
            setTimeStep(new_dt);
            LOG_INFO("PBFSolver adjusted dt to: " + std::to_string(new_dt) + " based on CFL condition.");
        }

        // 在所有物理计算之前，根据最新的transform更新刚体粒子位置
        if (!m_rigid_objects.empty()) {
            RigidObjectData* d_rigid_data;
            cudaMalloc(&d_rigid_data, m_rigid_objects.size() * sizeof(RigidObjectData));
            cudaMemcpy(d_rigid_data, m_rigid_objects.data(), m_rigid_objects.size() * sizeof(RigidObjectData), cudaMemcpyHostToDevice);
            
            update_rigid_body_positions(m_device_data, d_rigid_data, m_rigid_objects.size());

            cudaFree(d_rigid_data);
        }

        static const float export_gap = 1 / m_configData["EXPORT"]["SolverRequired"]["exportFps"].get<float>();

        apply_ext_force(m_host_data,
                        m_device_data,
                        m_neighborSearcher.m_config_cuData,
                        m_neighborSearcher.m_params_cuData);

        m_neighborSearcher.update(m_host_data->pos);

        for (int i = 0; i < m_configData["PBF"]["Required"]["iterationNum"].get<int>(); ++i) {
            compute_sph_density_and_error(m_host_data,
                                          m_device_data,
                                          m_neighborSearcher.m_config_cuData,
                                          m_neighborSearcher.m_params_cuData);

            update_lamb(m_host_data,
                        m_device_data,
                        m_neighborSearcher.m_config_cuData,
                        m_neighborSearcher.m_params_cuData);

            compute_dx(m_host_data,
                       m_device_data,
                       m_neighborSearcher.m_config_cuData,
                       m_neighborSearcher.m_params_cuData);

            apply_dx(m_host_data,
                     m_device_data,
                     m_neighborSearcher.m_config_cuData,
                     m_neighborSearcher.m_params_cuData);
        }

        post_correct(m_host_data,
                     m_device_data,
                     m_neighborSearcher.m_config_cuData,
                     m_neighborSearcher.m_params_cuData);

        m_host_data->cur_simTime += m_host_data->dt;
        m_frameCount++;
        if (m_host_data->cur_simTime >= export_gap * static_cast<float>(m_outputFrameCount)) {
            m_outputFrameCount++;
            m_doExportFlag = true;
        }

        if (cudaGetLastError() != cudaSuccess) {
            LOG_ERROR("PBFSolver tick failed.");
            m_isCrashed = true;
            return false;
        }
        return true;
    }

    bool PBFSolver::checkConfig() const {
        if (!m_configData.contains("PBF")) {
            LOG_ERROR("PBF config missing main domain: PBF")
            return false;
        }

        auto pbf_config = m_configData["PBF"];
        if (!pbf_config.contains("Required")) {
            LOG_ERROR("PBF config missing necessary domain: Required")
            return false;
        }

        for (const auto &key: PBFConfigRequiredKeys) {
            if (!pbf_config["Required"].contains(key)) {
                LOG_ERROR("PBF config missing Required key: " + key)
                return false;
            }
        }

        for (const auto &key: PBFConfigOptionalKeys) {
            if (!pbf_config["Optional"].contains(key)) {
                LOG_ERROR("PBF config missing Optional key: " + key)
                return false;
            }
        }

        return true;
    }

    // ------------------- Lightweight read-back APIs for service -------------------
    void PBFSolver::fetchAllParticles(std::vector<float3>& outPos, std::vector<float3>& outVel) {
        outPos.resize(m_host_data->particle_num);
        outVel.resize(m_host_data->particle_num);
        if (m_host_data->particle_num == 0) return;
        cudaMemcpy(outPos.data(), m_host_data->pos, m_host_data->particle_num * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(outVel.data(), m_host_data->vel, m_host_data->particle_num * sizeof(float3), cudaMemcpyDeviceToHost);
    }

    void PBFSolver::getAttachedObjectRanges(std::vector<int>& start, std::vector<int>& end) {
        start.clear(); end.clear();
        if (!m_configData.contains("EXPORT") || !m_configData["EXPORT"].contains("SolverRequired")) return;
        auto &sr = m_configData["EXPORT"]["SolverRequired"];
        if (sr.contains("exportObjectStartIndex")) {
            for (auto &v : sr["exportObjectStartIndex"]) start.push_back(v.get<int>());
        }
        if (sr.contains("exportObjectEndIndex")) {
            for (auto &v : sr["exportObjectEndIndex"]) end.push_back(v.get<int>());
        }
        // Fallback build end if missing
        if (!start.empty() && end.size() != start.size()) {
            end.resize(start.size());
            for (size_t i=0;i<start.size();++i) {
                if (i+1<start.size()) end[i] = start[i+1]; else end[i] = (int)m_host_data->particle_num;
            }
        }
    }

}