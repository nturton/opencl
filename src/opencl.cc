#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 100
#include <CL/cl2.hpp>
#include <cstdint>
#include <iostream>
#include <string>

// kernel calculates for each element C=A+B
static const char kernel_code[]=
{
#include "kernel.h"
};

static const cl::Program::Sources sources {
  { kernel_code, sizeof(kernel_code) },
};

void report_device(cl::Device &device)
{
  std::cout << "\n";
  std::cout << "\tOpenCL\tDevice : "  << device.getInfo<CL_DEVICE_NAME>().c_str() << std::endl;
  std::cout << "\t\t Type : "  << device.getInfo<CL_DEVICE_TYPE>() << std::endl;
  std::cout << "\t\t Vendor : "  << device.getInfo<CL_DEVICE_VENDOR>().c_str() << std::endl;
  std::cout << "\t\t Version : "  << device.getInfo<CL_DEVICE_VERSION>().c_str() << std::endl;
  std::cout << "\t\t Extensions : "  << device.getInfo<CL_DEVICE_EXTENSIONS>().c_str() << std::endl;
  std::cout << "\t\t Driver : "  << device.getInfo<CL_DRIVER_VERSION>().c_str() << std::endl;
  std::cout << "\t\t Global Mem : "  << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/(1024*1024) << " MBytes" << std::endl;
  std::cout << "\t\t Local Mem : "  << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024 << " KBytes" << std::endl;
  std::cout << "\t\t Compute Units : "  << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
  std::cout << "\t\t Max work item dimensions : "  << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
  std::cout << "\t\t Max work group size : "  << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  std::cout << "\t\t Preferred vector width char : "  << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR>() << std::endl;
  std::cout << "\t\t Preferred vector width short : "  << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT>() << std::endl;
  std::cout << "\t\t Preferred vector width int : "  << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT>() << std::endl;
  std::cout << "\t\t Preferred vector width long : "  << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG>() << std::endl;
  std::cout << "\t\t Preferred vector width float : "  << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() << std::endl;
  std::cout << "\t\t Preferred vector width double : "  << device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE>() << std::endl;
  std::cout << "\t\t Clock Rate : "  << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << " MHz" << std::endl;
  std::cout << "\t\t Address bits : "  << device.getInfo<CL_DEVICE_ADDRESS_BITS>() << std::endl;
  std::cout << "\t\t Max read image args : "  << device.getInfo<CL_DEVICE_MAX_READ_IMAGE_ARGS>() << std::endl;
  std::cout << "\t\t Max write image args : "  << device.getInfo<CL_DEVICE_MAX_WRITE_IMAGE_ARGS>() << std::endl;
  std::cout << "\t\t Max memory allocation size : "  << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()/(1024*1024) << " MBytes" << std::endl;
  std::cout << "\t\t Local memory type : "  << device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>() << std::endl;
  std::cout << "\t\t Local memory size : "  << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()/1024 << " KBytes" << std::endl;
}

void test_device(cl::Device &device)
{
  cl::Context context(device);
  cl::Program program(context, sources);
  if (program.build({device}) != CL_SUCCESS) {
    std::cout << "Failed to compile program: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    return;
  }

  // create buffers on the device
  cl::Buffer buffer_A(context,CL_MEM_READ_WRITE,sizeof(int)*10);
  cl::Buffer buffer_B(context,CL_MEM_READ_WRITE,sizeof(int)*10);
  cl::Buffer buffer_C(context,CL_MEM_READ_WRITE,sizeof(int)*10);

  // The local buffers.
  int A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, device);

  //write arrays A and B to the device
  queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(int)*10,A);
  queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(int)*10,B);

  cl::Kernel simple_add(program, "simple_add");
  simple_add.setArg(0, buffer_A);
  simple_add.setArg(1, buffer_B);
  simple_add.setArg(2, buffer_C);

  cl::Event event;
  queue.enqueueNDRangeKernel(simple_add,
                             cl::NullRange,
                             cl::NDRange(10),
                             cl::NDRange(10),
                             NULL,
                             &event);
  event.wait();

  int C[10];
  //read result C from the device to array C
  queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(int)*10,C);

  std::cout << "\t\tresult:";
  for(int i=0;i<10;i++){
    std::cout << " " << A[i] << "+" << B[i] << "=" << C[i];
  }
  std::cout << "\n";
}

void process_device(cl::Device &device)
{
  report_device(device);
  test_device(device);
}

void report_platform(cl::Platform &plat)
{
  std::cout << "OpenCL \tPlatform : " << plat.getInfo<CL_PLATFORM_NAME>().c_str() << std::endl;
  std::cout << "\tVendor: " << plat.getInfo<CL_PLATFORM_VENDOR>().c_str() << std::endl;
  std::cout << "\tVersion : " << plat.getInfo<CL_PLATFORM_VERSION>().c_str() << std::endl;
  std::cout << "\tExtensions : " << plat.getInfo<CL_PLATFORM_EXTENSIONS>().c_str() << std::endl;
}

void process_platform(cl::Platform &plat)
{
  report_platform(plat);

// get devices
  std::vector<cl::Device> devices;

  try {
    plat.getDevices(CL_DEVICE_TYPE_ALL,&devices);
  }
  catch (cl::Error err) {
    std::cout
      << std::endl
      << "\tFailed to get devices: "
      << err.what()
      << "("
      << err.err()
      << ")"
      << std::endl;
  }

// iterate over available devices
  for(std::vector<cl::Device>::iterator dev_it=devices.begin(); dev_it!=devices.end(); dev_it++)
  {
    process_device(*dev_it);
  }
  std::cout << "\n";
}

int process_platforms()
{
  typedef std::vector<cl::Platform> Platform_vec;
  Platform_vec platforms;
  cl::Platform::get(&platforms);
  if (platforms.size() == 0) {
    std::cout << "No OpenCL platforms available\n";
    return 1;
  }

  for(Platform_vec::iterator it = platforms.begin();
      it != platforms.end();
      it++) {
    process_platform(*it);
  }

  return 0;
}

int main()
{
  try {
    return process_platforms();
  }
  catch (cl::Error err) {
    std::cerr 
      << "ERROR: "
      << err.what()
      << "("
      << err.err()
      << ")"
      << std::endl;
    return 1;
  }
}
