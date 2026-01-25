/* opencl_stub.h - Complete headers for RingTensor GPU support */
#ifndef OPENCL_STUB_H
#define OPENCL_STUB_H

#include <stdlib.h>

/* Windows Calling Convention */
#ifdef _WIN32
    #define CL_API_ENTRY
    #define CL_API_CALL __stdcall
#else
    #define CL_API_ENTRY
    #define CL_API_CALL
#endif

/* Data Types */
typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;

typedef unsigned int                cl_uint;
typedef int                         cl_int;
typedef unsigned long long          cl_ulong;
typedef cl_ulong                    cl_bitfield;

/* Constants */
#define CL_SUCCESS                                  0
#define CL_DEVICE_TYPE_CPU                          (1 << 1)
#define CL_DEVICE_TYPE_GPU                          (1 << 2)

#define CL_MEM_READ_WRITE                           (1 << 0)
#define CL_MEM_WRITE_ONLY                           (1 << 1)
#define CL_MEM_READ_ONLY                            (1 << 2)
#define CL_MEM_COPY_HOST_PTR                        (1 << 5)

#define CL_TRUE                                     1
#define CL_FALSE                                    0

/* --- Added Constants for Diagnostics --- */
#define CL_DEVICE_NAME                              0x102B
#define CL_PROGRAM_BUILD_LOG                        0x1183

/* Function Prototypes */
CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id, cl_bitfield, cl_uint, cl_device_id *, cl_uint *);
CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id, cl_uint, size_t, void *, size_t *); // <--- NEW

CL_API_ENTRY cl_context CL_API_CALL clCreateContext(const void *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context, cl_device_id, cl_bitfield, cl_int *);

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context, cl_bitfield, size_t, void *, cl_int *);

CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *);
CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *); // <--- NEW

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program, const char *, cl_int *);
CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel, cl_uint, size_t, const void *);

CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem);
CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel);
CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program);
CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue);
CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context);

#endif