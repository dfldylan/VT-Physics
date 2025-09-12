PROJECT(VP-CMD)

INCLUDE_DIRECTORIES(
        ${VP_SOURCE_ROOT}
        ${VP_RUNTIME_ROOT}/Include
        ${VP_CONFIG_ROOT}
)

ADD_EXECUTABLE(vp SimulatorCMD.cpp)
TARGET_LINK_LIBRARIES(vp PUBLIC vpmanager)

# New lightweight PBF TCP server executable
ADD_EXECUTABLE(vp_pbf_server PBFServer.cpp)
TARGET_LINK_LIBRARIES(vp_pbf_server PUBLIC vpmanager)
if (WIN32)
    target_link_libraries(vp_pbf_server PRIVATE Ws2_32)
endif()