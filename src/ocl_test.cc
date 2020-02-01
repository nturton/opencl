#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_HPP_TARGET_OPENCL_VERSION 100
#include <CL/cl2.hpp>

#include <boost/program_options.hpp>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace po = boost::program_options;

typedef std::vector<unsigned> uint_vec;

// kernel calculates for each element C=A+B
static const char kernel_code[]=
{
#include "kernel.h"
};

static const cl::Program::Sources sources {
  { kernel_code, sizeof(kernel_code) },
};

void test_device(cl::Context &context, cl::Device &device,
                 unsigned count, unsigned size,
                 const uint_vec &iters_vec)
{
  cl::Program program(context, sources);
  if (program.build({device}) != CL_SUCCESS) {
    std::cout << "Failed to compile program: "
              << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
              << std::endl;
    return;
  }

  // create buffers on the device
  cl::Buffer iter_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*3);
  cl::Buffer result_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*count);

  // The local buffers.
  int iters[] = {1024, 1024, 1024};
  int results[count];

  switch (iters_vec.size()) {
  case 1:
    iters[0] = iters_vec[0];
    iters[1] = iters_vec[0];
    iters[2] = iters_vec[0];
    break;

  case 3:
    iters[0] = iters_vec[0];
    iters[1] = iters_vec[1];
    iters[2] = iters_vec[2];
    break;

  default:
    assert(false);
  }

  //create queue to which we will push commands for the device.
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  // Write array iters to the device
  queue.enqueueWriteBuffer(iter_buffer, CL_TRUE, 0, sizeof(cl_uint)*3, iters);

  cl::Kernel crc_iter(program, "crc_iter");
  crc_iter.setArg(0, iter_buffer);
  crc_iter.setArg(1, result_buffer);

  cl::Event event;

  auto start = std::chrono::high_resolution_clock::now();

  queue.enqueueNDRangeKernel(crc_iter,
                             cl::NullRange,
                             cl::NDRange(count),
                             cl::NDRange(size),
                             NULL,
                             &event);
  event.wait();

  auto end = std::chrono::high_resolution_clock::now();

  //read results from the device to array results
  queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0,
                          sizeof(cl_uint)*count, results);

  cl_uint total = 0;
  for(int i=0;i<count;i++)
    total += results[i];

  std::stringstream ss;
  ss << std::hex << std::setw(8) << std::setfill('0') << total;
  std::cout << "Total: 0x" << ss.str() << '\n';

  const double ns_per_s = 1000000000.;
  double t_queued = double(event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>()) / ns_per_s;
  double t_submit = double(event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>()) / ns_per_s;
  double t_start = double(event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) / ns_per_s;
  double t_end = double(event.getProfilingInfo<CL_PROFILING_COMMAND_END>()) / ns_per_s;

  typedef std::chrono::duration<double> duration_double;
  auto d = std::chrono::duration_cast<duration_double>(end-start);
  std::cout << "Elapsed:   " << d.count() << " s\n";
  std::cout << "Queued:    " << t_submit - t_queued << " s\n";
  std::cout << "Submitted: " << t_start - t_submit << " s\n";
  std::cout << "Running:   " << t_end - t_start << " s\n";
}

int test_devices(unsigned count, unsigned size, const uint_vec &iters_vec)
{
  cl::Context context(CL_DEVICE_TYPE_DEFAULT);

  std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  for(auto it=devices.begin(); it!=devices.end(); it++) {
    cl::Platform platform(it->getInfo<CL_DEVICE_PLATFORM>());
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
    std::cout << "Device:   " << it->getInfo<CL_DEVICE_NAME>() << '\n';
    test_device(context, *it, count, size, iters_vec);
    std::cout << '\n';
  }

  return 0;
}

static uint_vec
parse_uint_vec(const std::string &s)
{
  uint_vec result;
  const char *p = s.c_str();

  while(true) {
    char *end;
    result.push_back(strtoul(p, &end, 0));
    if (*end == 0)
      break;

    if (*end != ',') {
      std::stringstream ss;
      ss << "Invalid character '" << *end << "' in number list.";
      throw std::runtime_error(ss.str());
    }
    p = end+1;
  }

  return result;
}

int main(int argc, char **argv)
{
  try {
    unsigned count;
    unsigned size;
    std::string iters_str = "1024,1024,1024";

    // Declare the supported options.
    po::options_description desc("Options");
    desc.add_options()
      ("help", "Produce this help message")
      ("count,c", po::value(&count)->default_value(1), "The number of work items")
      ("iters,i", po::value(&iters_str), "The number of iterations")
      ("size,s", po::value(&size)->default_value(1), "The work group size")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }

    uint_vec iters_vec = parse_uint_vec(iters_str);

    if (iters_vec.size() != 1 && iters_vec.size() != 3) {
      std::cerr << "Only 1 or 3 iteration counts are allowed.\n";
      return 1;
    }

    return test_devices(count, size, iters_vec);
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
