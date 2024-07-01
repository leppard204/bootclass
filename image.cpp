#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <windows.h>
#include "cbmp.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define CHECK_ERROR(err) if( err != CL_SUCCESS )        \
        { std::cout << __LINE__ << ", OpenCL Error: " << err << std::endl; exit(-1); }

int main(int argc, char* argv[])
{
    BMP* bmp = bopen("image.bmp");
    unsigned int width, height, length;
    width = get_width(bmp);
    height = get_height(bmp);
    length = width * height * 3;

    float rot_degree = 45.f;    // TODO

    unsigned char* h_pixels         = (unsigned char*)malloc(length * sizeof(unsigned char));
    unsigned char* h_pixels_result  = (unsigned char*)malloc(length * sizeof(unsigned char));
    get_pixels(bmp, h_pixels);

    // Set up the OpenCL platform 
    int err;
    int ndim = 1;
    cl_context          context;
    cl_command_queue    cmdQ;
    cl_program          program;
    cl_kernel           kernel;
    cl_uint             numPlatforms;
    cl_uint             numDevices;
    cl_platform_id      platform_id = NULL;
    cl_platform_id* platform_id_list;
    cl_device_id        device_id = NULL;
    cl_device_id* device_id_list;

    char cBuffer[1024];

    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    platform_id_list = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    printf(" %d OpenCL Platforms found\n\n", numPlatforms);

    err = clGetPlatformIDs(numPlatforms, platform_id_list, NULL);
    printf(" ----------------------------------\n");
    for (int i = 0; i < numPlatforms; ++i) {
        err = clGetPlatformInfo(platform_id_list[i], CL_PLATFORM_NAME, sizeof(cBuffer), &cBuffer, NULL);
        printf(" CL_PLATFORM_NAME: \t%s\n", cBuffer);
    }
    printf(" ----------------------------------\n\n");

    platform_id = platform_id_list[1];    // Select OpenCL platform 0 ~ numPlatforms-1

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    device_id_list = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, numDevices, device_id_list, &numDevices);
    printf(" %d OpenCL devices found\n\n", numDevices);
    printf(" ----------------------------------\n");
    for (unsigned int i = 0; i < numDevices; ++i) {
        clGetDeviceInfo(device_id_list[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
        printf(" CL_DEVICE_NAME: \t%s\n", cBuffer);
    }
    printf(" ----------------------------------\n\n");

    device_id = device_id_list[0];     // Select OpenCL platform 0 ~ numDevices-1

    cl_context_properties properties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0
    };

    context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);
    cmdQ = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

    // Print device info
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
    printf(" Selected device: %s\n", cBuffer);

    // Set up the buffers, initialize matrices, and wirte them into global memory
    cl_mem d_pixels, d_pixels_result;
    d_pixels        = clCreateBuffer(context, CL_MEM_READ_ONLY,  
                                     length * sizeof(unsigned char), NULL, NULL);
    d_pixels_result = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                     length * sizeof(unsigned char), NULL, NULL);

    // Write the A and B matrices into device memory
    err = clEnqueueWriteBuffer(cmdQ, d_pixels, CL_TRUE, 0, 
                               length * sizeof(unsigned char), h_pixels, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Create the OpenCL program 
    std::ifstream file("image_processing.cl");
    std::stringstream ss;
    ss << file.rdbuf();
    std::string str = ss.str();
    const char* source = str.c_str();

    program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
    CHECK_ERROR(err);

    // Build the program
    char build_option[1000] = { 0, };
    err = clBuildProgram(program, 0, NULL, build_option, NULL, NULL);

    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
            sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return -1;
    }

    // Create the compute kernel from the program
    kernel = clCreateKernel(program, "image_rotate_gpu", &err);
    CHECK_ERROR(err);

    size_t global[] = { width*height };	                // number of total work-items

    // Set the arguments to our compute kernel
    err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_pixels);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_pixels_result);
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &height);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &rot_degree);
    CHECK_ERROR(err);
  
    // Launch GPU kernel
    err = clEnqueueNDRangeKernel(cmdQ, kernel, ndim, NULL, global, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    // Wait for the command to be completed before reading back results
    clFinish(cmdQ);

    // Read back the results from the compute device
    err = clEnqueueReadBuffer(cmdQ, d_pixels_result, CL_TRUE, 0, 
                              length * sizeof(unsigned char), h_pixels_result, 0, NULL, NULL);
    CHECK_ERROR(err);

    set_pixels(bmp, h_pixels_result);
    bwrite(bmp, "result.bmp");

    return 0;
}

