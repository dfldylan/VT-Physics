PROJECT(VPModule-UGNS)

FIND_PACKAGE(CUDAToolkit)
FIND_PACKAGE(nlohmann_json CONFIG REQUIRED)

# Build VP-UGNS =================================================================================
FILE(GLOB_RECURSE UGNS_FILES
        ${VP_RUNTIME_ROOT}/Include/Modules/NeighborSearch/UGNS/*.hpp
        ${VP_RUNTIME_ROOT}/Source/Modules/NeighborSearch/UGNS/*.cpp
        ${VP_RUNTIME_ROOT}/Source/Modules/NeighborSearch/UGNS/*.cuh
        ${VP_RUNTIME_ROOT}/Source/Modules/NeighborSearch/UGNS/*.cu
)

ADD_LIBRARY(vpugns STATIC ${UGNS_FILES})
TARGET_INCLUDE_DIRECTORIES(vpugns PRIVATE ${VP_LIB_ROOT}/Include)
TARGET_INCLUDE_DIRECTORIES(vpugns PUBLIC ${NLOHMANN_JSON_INCLUDE_DIRS})
TARGET_LINK_LIBRARIES(vpugns PRIVATE vpcore)
ADD_DEPENDENCIES(vpugns vpcore)
if(MSVC)
    ADD_COMPILE_OPTIONS("/utf-8")
endif()
# Build VP-UGNS =================================================================================