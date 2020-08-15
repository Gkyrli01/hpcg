cmake_minimum_required(VERSION 3.10)
project(hpcg)

set(CMAKE_CXX_STANDARD 11)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#Dont change
add_definitions(-DSyCL_SPMV)
add_definitions(-DSyCL_RESTRICTION)
add_definitions(-DSyCL_PROLONGATION)
add_definitions(-DSyCL_WAXPBY)
add_definitions(-DSyCL_DOT)
add_definitions(-DSyCL_SYMGS)
##Dont change end

add_definitions(-DHPCG_NO_OPENMP)
add_definitions(-DSyCL_CPU)#Comment out if you want to run on the gpu
#add_definitions(-DTRANSPOSE)#Comment out if you do not want to use transposed 2D arrays (Transpose for GPUs)

#add_definitions(-DHPCG_DEBUG)
set(CMAKE_MODULE_PATH ~/Desktop/ccppsdk/cmake/Modules/)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
get_directory_property(MYDEFS COMPILE_DEFINITIONS)
if(MYDEFS MATCHES "SyCL_CPU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Ofast")
    add_definitions(-DSYMGSSIZE=36)
    add_definitions(-DSPMVSIZE=36)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ffast-math -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Ofast")
    add_definitions(-DSYMGSSIZE=768)
    add_definitions(-DSPMVSIZE=512)
endif()

include_directories(src)
add_executable(sycli
        src/CG.cpp
        src/CG.hpp
        src/CG_ref.cpp
        src/CG_ref.hpp
        src/CGData.hpp
        src/CheckAspectRatio.cpp
        src/CheckAspectRatio.hpp
        src/CheckProblem.cpp
        src/CheckProblem.hpp
        src/ComputeDotProduct.cpp
        src/ComputeDotProduct.hpp
        src/ComputeDotProduct_ref.cpp
        src/ComputeDotProduct_ref.hpp
        src/ComputeDotProduct_SyCL.cpp
        src/ComputeDotProduct_SyCL.hpp
        src/ComputeMG.cpp
        src/ComputeMG.hpp
        src/ComputeMG_ref.cpp
        src/ComputeMG_ref.hpp
        src/ComputeMG_SyCL.cpp
        src/ComputeMG_SyCL.hpp
        src/ComputeOptimalShapeXYZ.cpp
        src/ComputeOptimalShapeXYZ.hpp
        src/ComputeProlongation_ref.cpp
        src/ComputeProlongation_ref.hpp
        src/ComputeProlongation_SyCL.cpp
        src/ComputeProlongation_SyCL.hpp
        src/ComputeResidual.cpp
        src/ComputeResidual.hpp
        src/ComputeRestriction_ref.cpp
        src/ComputeRestriction_ref.hpp
        src/ComputeRestriction_SyCL.cpp
        src/ComputeRestriction_SyCL.hpp
        src/ComputeSPMV.cpp
        src/ComputeSPMV.hpp
        src/ComputeSPMV_ref.cpp
        src/ComputeSPMV_ref.hpp
        src/ComputeSPMV_SyCL.cpp
        src/ComputeSPMV_SyCL.hpp
        src/ComputeSYMGS.cpp
        src/ComputeSYMGS.hpp
        src/ComputeSYMGS_ref.cpp
        src/ComputeSYMGS_ref.hpp
        src/ComputeSYMGS_SyCL.cpp
        src/ComputeSYMGS_SyCL.hpp
        src/ComputeWAXPBY.cpp
        src/ComputeWAXPBY.hpp
        src/ComputeWAXPBY_ref.cpp
        src/ComputeWAXPBY_ref.hpp
        src/ComputeWAXPBY_SyCL.cpp
        src/ComputeWAXPBY_SyCL.hpp
        src/ExchangeHalo.cpp
        src/ExchangeHalo.hpp
        src/finalize.cpp
        src/GenerateCoarseProblem.cpp
        src/GenerateCoarseProblem.hpp
        src/GenerateGeometry.cpp
        src/GenerateGeometry.hpp
        src/GenerateProblem.cpp
        src/GenerateProblem.hpp
        src/GenerateProblem_ref.cpp
        src/GenerateProblem_ref.hpp
        src/Geometry.hpp
        src/hpcg.hpp
        src/init.cpp
        src/main.cpp
        src/MGData.hpp
        src/MixedBaseCounter.cpp
        src/MixedBaseCounter.hpp
        src/mytimer.cpp
        src/mytimer.hpp
        src/OptimizeProblem.cpp
        src/OptimizeProblem.hpp
        src/OutputFile.cpp
        src/OutputFile.hpp
        src/ReadHpcgDat.cpp
        src/ReadHpcgDat.hpp
        src/ReportResults.cpp
        src/ReportResults.hpp
        src/SetupHalo.cpp
        src/SetupHalo.hpp
        src/SetupHalo_ref.cpp
        src/SetupHalo_ref.hpp
        src/SparseMatrix.hpp
        src/TestCG.cpp
        src/TestCG.hpp
        src/TestNorms.cpp
        src/TestNorms.hpp
        src/TestSymmetry.cpp
        src/TestSymmetry.hpp
        src/Vector.hpp
        src/WriteProblem.cpp
        src/WriteProblem.hpp
        src/YAML_Doc.cpp
        src/YAML_Doc.hpp
        src/YAML_Element.cpp
        src/YAML_Element.hpp
        src/SaveArrays.h
        src/SyCLBufferContainer.h
        src/SyCLResources.h
        src/SyCLResources.cpp)

target_link_libraries(sycli ${MPI_LIBRARIES})

#include(FindComputeCpp)
#set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} -fopenmp)
#set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS} -fopenmp=libiomp)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

#set(HPCG_NO_OPENMP)
#set(HPCG_NO_MPI)
#
#FIND_PACKAGE(OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(OMP_NUM_THREADS 12)
#endif()
#set(-DHPCG_NO_OPENMP)

