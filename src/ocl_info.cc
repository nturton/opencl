#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 100
#include <CL/cl2.hpp>
#include <cstdint>
#include <list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

void report_device(cl::Device &device)
{
  std::cout << "\n";
  std::cout << "\tOpenCL\tDevice : "  << device.getInfo<CL_DEVICE_NAME>().c_str() << std::endl;
  cl_int val = device.getInfo<CL_DEVICE_TYPE>();
  std::cout << "\t\t Type : "  << val << " (";
  std::list<std::string> types;
  if (val & CL_DEVICE_TYPE_DEFAULT)
    types.push_back("default");
  if (val & CL_DEVICE_TYPE_CPU)
    types.push_back("cpu");
  if (val & CL_DEVICE_TYPE_GPU)
    types.push_back("gpu");
  if (val & CL_DEVICE_TYPE_ACCELERATOR)
    types.push_back("accel");
  if (val & CL_DEVICE_TYPE_CUSTOM)
    types.push_back("custom");
  for(auto it=types.begin(); it!=types.end(); it++) {
    std::cout << (it != types.begin() ? " " : "") << *it;
  }
  std::cout << ")\n";
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

void process_device(cl::Device &device)
{
  report_device(device);
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
