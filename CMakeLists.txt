CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(sloan)
INCLUDE(cmake/SBELUtils.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

enable_cuda_support()
enable_openmp_support()
MESSAGE(STATUS "Cuda version: ${CUDA_VERSION}")

SET(SLOAN_HEADERS
	sloan/common.h
	sloan/sloan.h
	sloan/timer.h
  )

SET(MMIO_FILES
  mm_io/mm_io.h
  mm_io/mm_io.c
  )

SOURCE_GROUP("Headers" FILES ${SLOAN_HEADERS})
# SOURCE_GROUP("CUDA Headers" FILES ${SLOAN_CUHEADERS})
SOURCE_GROUP("MM_IO" FILES ${MMIO_FILES})

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    )

cuda_add_executable(testing testing.cu ${SLOAN_HEADERS} ${MMIO_FILES})
