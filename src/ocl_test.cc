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

// kernel calculates for each element C=A+B
static const char kernel_code[]=
{
#include "kernel.h"
};

static const cl::Program::Sources sources {
  { kernel_code, sizeof(kernel_code) },
};

void test_device(cl::Context &context, cl::Device &device)
{
  cl::Program program(context, sources);
  if (program.build({device}) != CL_SUCCESS) {
    std::cout << "Failed to compile program: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    return;
  }

  int count = 100;

  // create buffers on the device
  cl::Buffer iter_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*3);
  cl::Buffer result_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*count);

  // The local buffers.
  int iters[] = {1024, 1024, 1024};
  int results[count];

  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, device);

  // Write array iters to the device
  queue.enqueueWriteBuffer(iter_buffer, CL_TRUE, 0, sizeof(cl_uint)*3, iters);

  cl::Kernel simple_add(program, "crc_iter");
  simple_add.setArg(0, iter_buffer);
  simple_add.setArg(1, result_buffer);

  cl::Event event;
  queue.enqueueNDRangeKernel(simple_add,
                             cl::NullRange,
                             cl::NDRange(count),
                             cl::NDRange(count),
                             NULL,
                             &event);
  event.wait();

  //read results from the device to array results
  queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0,
                          sizeof(cl_uint)*count, results);

  cl_uint total = 0;
  for(int i=0;i<count;i++)
    total += results[i];

  std::stringstream ss;
  ss << std::hex << std::setw(8) << std::setfill('0') << total;
  std::cout << "Total: 0x" << ss.str() << '\n';
}

int test_devices()
{
  cl::Context context(CL_DEVICE_TYPE_DEFAULT);

  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  for(auto it=devices.begin(); it!=devices.end(); it++) {
    cl::Platform platform(it->getInfo<CL_DEVICE_PLATFORM>());
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
    std::cout << "Device:   " << it->getInfo<CL_DEVICE_NAME>() << '\n';
    test_device(context, *it);
    std::cout << '\n';
  }

  return 0;
}

int main()
{
  try {
    return test_devices();
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
  catch (std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
