#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>

// 必须在任何 Windows 头之前定义，防止 Windows 定义 min/max 宏
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifdef _WIN32
// 确保 winsock2 在 windows.h 之前被包含（如已有包含也无妨）
#include <winsock2.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#define closesocket close
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR   (-1)
#endif

#include "Manager/Simulator.hpp"
#include "Framework/Material.hpp"
#include "JSON/JSONHandler.hpp"
#include "Logger/Logger.hpp"
#include "Solvers/PBF/PBFSolver.hpp"

using namespace VT_Physics;

// Simple binary protocol
// [int32 type][payload...]
// Types:
//   1 Reset: payload { float particleRadius }
//   2 Add Fluid cloud: payload { int32 id, int32 N, float pos[N*3], float vel[N*3] }
//   3 Add Solid cloud: payload { int32 id, int32 B, float pos[B*3], float normal[B*3] }  // not simulated yet
//   4 Next frame: payload { float dt } -> response: { int32 ok, int32 objCount, [int32 id, int32 N, float pos[N*3], float vel[N*3]]* }
//   5 Clear objects: payload {}
//   9 Shutdown

namespace {
    struct Cloud {
        int id{0};
        bool isFluid{true};
        std::vector<float3> pos;
        std::vector<float3> vel;  // for fluid. for solid, store normals in vel for now
        int material{EPM_FLUID};
    };

    class PBFService {
    public:
        explicit PBFService(int port): m_port(port) {}

        bool init() {
#ifdef _WIN32
            WSADATA wsaData;
            if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
                LOG_ERROR("WSAStartup failed");
                return false;
            }
#endif
            m_listen = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (m_listen == INVALID_SOCKET) {
                LOG_ERROR("socket() failed");
                return false;
            }
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = htonl(INADDR_ANY);
            addr.sin_port = htons((uint16_t)m_port);
            
#ifdef _WIN32
            // 在 Windows 上，尝试设置独占地址使用，可以避免一些复杂的端口复用问题
            int exclusiveOpt = 1;
            if (setsockopt(m_listen, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*)&exclusiveOpt, sizeof(exclusiveOpt)) == SOCKET_ERROR) {
                LOG_WARNING("setsockopt(SO_EXCLUSIVEADDRUSE) failed, continuing anyway...");
            }
#endif

            int opt=1;
#ifdef _WIN32
            setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
#else
            setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
            if (bind(m_listen, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
#ifdef _WIN32
                LOG_ERROR("bind() failed with error: " + std::to_string(WSAGetLastError()));
#else
                LOG_ERROR("bind() failed");
#endif
                return false;
            }
            if (listen(m_listen, 1) == SOCKET_ERROR) {
                LOG_ERROR("listen() failed");
                return false;
            }
            LOG_INFO("PBFServer listening on port: " + std::to_string(m_port));
            return true;
        }

        void serve() {
            SOCKET client = accept(m_listen, nullptr, nullptr);
            if (client == INVALID_SOCKET) {
                LOG_ERROR("accept() failed");
                return;
            }
            LOG_INFO("Client connected");
            for (;;) {
                int32_t type = 0;
                if (!recvAll(client, &type, sizeof(type))) break;
                switch (type) {
                    case 1: onReset(client); break;
                    case 2: onAddFluid(client); break;
                    case 3: onAddSolid(client); break;
                    case 4: onNextFrame(client); break;
                    case 5: onClear(client); break;
                    case 9: { LOG_INFO("Shutdown signal received"); closesocket(client); return; }
                    default: LOG_WARNING("Unknown msg type: " + std::to_string(type)); break;
                }
            }
            closesocket(client);
        }

    private:
        // minimal scene state
        Simulator &sim = Simulator::getInstance();
        ObjectManager *objMgr = sim.getObjectManager();
        SolverManager *solverMgr = sim.getSolverManager();
        Solver *solver{nullptr};
        std::unordered_map<int, Object*> id2obj;
        std::vector<int> attachOrder; // keep the same order as solver->attachObject()
        float lastDt{0.005f};

        // net helpers
        static bool recvAll(SOCKET s, void* buf, size_t len) {
            char* p = static_cast<char*>(buf);
            size_t got = 0;
            while (got < len) {
                int r = ::recv(s, p+got, (int)(len-got), 0);
                if (r <= 0) return false;
                got += (size_t)r;
            }
            return true;
        }
        static bool sendAll(SOCKET s, const void* buf, size_t len) {
            const char* p = static_cast<const char*>(buf);
            size_t sent = 0;
            while (sent < len) {
                int r = ::send(s, p+sent, (int)(len-sent), 0);
                if (r <= 0) return false;
                sent += (size_t)r;
            }
            return true;
        }

        void ensureSolver() {
            if (!solver) {
                solver = solverMgr->createSolver(eSolverType::PBF);
                // default config from template
                auto cfg = solver->getSolverConfigTemplate();
                solver->setConfig(cfg);
            }
        }

        void onReset(SOCKET s) {
            float radius = 0.f;
            if (!recvAll(s, &radius, sizeof(radius))) return;
            // clear all
            objMgr->clear();
            id2obj.clear();
            attachOrder.clear();
            if (solver) { solver->destroy(); delete solver; solver = nullptr; }
            // ensureSolver();
            // update radius in config
            solver = solverMgr->createSolver(PBF);
            auto pbf_config = solver->getSolverConfigTemplate();
            pbf_config["PBF"]["Required"]["animationTime"] = 5.0f;
            pbf_config["PBF"]["Required"]["timeStep"] = 0.01f;
            pbf_config["PBF"]["Required"]["particleRadius"] = radius > 0 ? radius : 0.05f;
            pbf_config["PBF"]["Required"]["simSpaceLB"] = {-10, -10, -10};
            pbf_config["PBF"]["Required"]["simSpaceSize"] = {20, 20, 20};
            pbf_config["PBF"]["Required"]["iterationNum"] = 25;
            pbf_config["PBF"]["Required"]["XSPH_k"] = 0.01;
            pbf_config["PBF"]["Required"]["fPartRestDensity"] = 1000.f;
            pbf_config["PBF"]["Required"]["bPartRestDensity"] = 1500.f;
            pbf_config["EXPORT"]["Common"]["exportTargetDir"] = ".\\VP-Examples\\PBF";
            pbf_config["EXPORT"]["SolverRequired"]["enable"] = true;
            pbf_config["EXPORT"]["SolverRequired"]["exportFps"] = 35;
            solver->setConfig(pbf_config);
            int32_t ok = 1; sendAll(s, &ok, sizeof(ok));
        }

        void onAddFluid(SOCKET s) {
            // ensureSolver();
            int32_t id=0, N=0; 
            if (!recvAll(s, &id, sizeof(id))) return;
            if (!recvAll(s, &N, sizeof(N))) return;
            if (N<=0) { int32_t ok=0; sendAll(s,&ok,sizeof(ok)); return; }
            std::vector<float> pos( (size_t)N*3 );
            std::vector<float> vel( (size_t)N*3 );
            if (!recvAll(s, pos.data(), pos.size()*sizeof(float))) return;
            if (!recvAll(s, vel.data(), vel.size()*sizeof(float))) return;

            // create object with Particle_Common
            Object* obj = objMgr->createObject(ObjectType::Particle_Common);
            auto &ocfg = obj->getObjectComponentConfig();
            auto solverObjCfg = solver->getSolverObjectComponentConfigTemplate();
            // fill basic config
            auto pbfCfg = solver->getSolverConfigTemplate();
            float r = pbfCfg["PBF"]["Required"]["particleRadius"].get<float>();
            ocfg["particleRadius"] = r;
            ocfg["epmMaterial"] = EPM_FLUID;
            // Minimal change: raw points injection key
            ocfg["__rawPoints__"] = pos; // flatten ok
            // solver object component
            solverObjCfg["solverType"] = (uint8_t)SolverType::PBF;
            solverObjCfg["exportFlag"] = false;
            solverObjCfg["velocityStart"] = {0.f,0.f,0.f};
            solverObjCfg["colorStart"] = {0.f,0.6f,1.f};
            obj->attachSpecificSolverObjectComponentConfig(solverObjCfg);
            obj->rename("fluid_" + std::to_string(id));
            obj->update();
            solver->attachObject(obj);
            id2obj[id]=obj;
            attachOrder.push_back(id);
            int32_t ok=1; sendAll(s,&ok,sizeof(ok));
        }

        void onAddSolid(SOCKET s) {
            ensureSolver();
            int32_t id=0, B=0; 
            if (!recvAll(s, &id, sizeof(id))) return;
            if (!recvAll(s, &B, sizeof(B))) return;
            if (B<=0) { int32_t ok=0; sendAll(s,&ok,sizeof(ok)); return; }
            std::vector<float> pos( (size_t)B*3 );
            std::vector<float> nrm( (size_t)B*3 );
            if (!recvAll(s, pos.data(), pos.size()*sizeof(float))) return;
            if (!recvAll(s, nrm.data(), nrm.size()*sizeof(float))) return;

            Object* obj = objMgr->createObject(ObjectType::Particle_Common);
            auto &ocfg = obj->getObjectComponentConfig();
            auto solverObjCfg = solver->getSolverObjectComponentConfigTemplate();
            auto pbfCfg = solver->getSolverConfigTemplate();
            float r = pbfCfg["PBF"]["Required"]["particleRadius"].get<float>();
            ocfg["particleRadius"] = r;
            ocfg["epmMaterial"] = EPM_BOUNDARY;
            ocfg["__rawPoints__"] = pos;
            solverObjCfg["solverType"] = (uint8_t)SolverType::PBF;
            solverObjCfg["exportFlag"] = false;
            solverObjCfg["velocityStart"] = {0.f,0.f,0.f};
            solverObjCfg["colorStart"] = {0.8f,0.8f,0.8f};
            obj->attachSpecificSolverObjectComponentConfig(solverObjCfg);
            obj->rename("solid_" + std::to_string(id));
            obj->update();
            solver->attachObject(obj);
            id2obj[id]=obj;
            attachOrder.push_back(id);
            int32_t ok=1; sendAll(s,&ok,sizeof(ok));
        }

        void onNextFrame(SOCKET s) {
            // log
            LOG_INFO("Received next frame.");

            ensureSolver();
            float dt = 0.f; if (!recvAll(s, &dt, sizeof(dt))) return; 
            // if (dt>0) {
            //     auto pbf = dynamic_cast<VT_Physics::pbf::PBFSolver*>(solver);
            //     if (pbf) pbf->setTimeStep(dt);
            // }

            // initialize once lazily via tickNsteps (内部会自动 initialize)
            // solver->initialize();   // <-- 移除这行

            solver->tickNsteps(1);

            // Read back from PBFSolver
            auto pbf = dynamic_cast<VT_Physics::pbf::PBFSolver*>(solver);
            if (!pbf) { int32_t ok=0; sendAll(s,&ok,sizeof(ok)); return; }
            std::vector<float3> allPos, allVel;
            pbf->fetchAllParticles(allPos, allVel);
            std::vector<int> start, end; pbf->getAttachedObjectRanges(start, end);

            int32_t ok = 1; sendAll(s, &ok, sizeof(ok));
            int objCount = std::min((int)attachOrder.size(), (int)start.size());
            int32_t objCount32 = objCount; sendAll(s, &objCount32, sizeof(objCount32));

            for (int i=0;i<objCount;++i) {
                int32_t id = attachOrder[i];
                int sIdx = start[i]; int eIdx = end[i]; if (eIdx < sIdx) eIdx = sIdx;
                int N = eIdx - sIdx; int32_t N32 = N;
                sendAll(s, &id, sizeof(id));
                sendAll(s, &N32, sizeof(N32));
                if (N>0) {
                    // pack pos and vel into contiguous float arrays
                    std::vector<float> buf; buf.reserve((size_t)N*3);
                    for (int k=sIdx; k<eIdx; ++k) { auto &p = allPos[k]; buf.push_back(p.x); buf.push_back(p.y); buf.push_back(p.z); }
                    sendAll(s, buf.data(), buf.size()*sizeof(float));
                    buf.clear(); buf.reserve((size_t)N*3);
                    for (int k=sIdx; k<eIdx; ++k) { auto &v = allVel[k]; buf.push_back(v.x); buf.push_back(v.y); buf.push_back(v.z); }
                    sendAll(s, buf.data(), buf.size()*sizeof(float));
                }
            }

            // log
            LOG_INFO("Frame done. dt=" + std::to_string(dt) + ", objects=" + std::to_string(objCount) + ", particles=" + std::to_string(allPos.size()));
        }

        void onClear(SOCKET s) {
            objMgr->clear();
            id2obj.clear();
            attachOrder.clear();
            if (solver) { solver->destroy(); delete solver; solver = nullptr; }
            int32_t ok=1; sendAll(s,&ok,sizeof(ok));
        }

    private:
        int m_port{55001};
        SOCKET m_listen{INVALID_SOCKET};
    };
}

int main(int argc, char** argv) {
    int port = 55001;
    if (argc>=2) {
        port = std::atoi(argv[1]);
        if (port<=0) port = 55001;
    }
    PBFService svc(port);
    if (!svc.init()) return -1;
    svc.serve();
    return 0;
}
