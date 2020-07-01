//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct_ref.cpp

 HPCG routine
 */
#include "SyCLResources.h"

#ifndef HPCG_NO_MPI

#include <mpi.h>
#include "mytimer.hpp"

#endif
#ifndef HPCG_NO_OPENMP

#include <omp.h>

#endif

#include <cassert>
#include "ComputeDotProduct_SyCL.hpp"


template<typename T>
void CallReductionKernel(sycl::queue &queue, int wgroup_size, sycl::buffer<T, 1> &buf, sycl::buffer<T, 1> &output,
						 int n_wgroups, int len) {
	queue.submit([&](sycl::handler &cgh) {
		sycl::accessor<T, 1, sycl::access::mode::read_write,
				sycl::access::target::local>
				local_mem(sycl::range<1>(wgroup_size+32), cgh);
		auto global_mem = buf.template get_access<sycl::access::mode::read>(cgh);
		auto result = output.template get_access<sycl::access::mode::write>(cgh);
		cgh.parallel_for<class dot_kernel>(
				sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
				[=](sycl::nd_item<1> item) {
					int id = item.get_local_id(0);
					int grouId = item.get_group_linear_id();
					int myGlobalReductionIndex = grouId * (wgroup_size * 2) + id;
					int gridSize = wgroup_size * 2 * n_wgroups;
					local_mem[id] = 0.0;
					local_mem[id+32] = 0.0;

					while (myGlobalReductionIndex < len) {
						local_mem[id] += global_mem[myGlobalReductionIndex] +
										 global_mem[myGlobalReductionIndex + wgroup_size];
						myGlobalReductionIndex += gridSize;
					}
					item.barrier();
					if (wgroup_size >= 32) {
						local_mem[id] += local_mem[id + 16];
						item.barrier();
					}
					if (wgroup_size >= 16)
						local_mem[id] += local_mem[id + 8];
					if (wgroup_size >= 8)
						local_mem[id] += local_mem[id + 4];
					if (wgroup_size >= 4)
						local_mem[id] += local_mem[id + 2];
					if (wgroup_size >= 2)
						local_mem[id] += local_mem[id + 1];
					if (id == 0)
						result[grouId] = local_mem[0];
				});
	});
}

template<typename T>
void CallDotSingleBufferKernel(sycl::queue &queue, int wgroup_size, sycl::buffer<T, 1> &buf, sycl::buffer<T, 1> &output,
							   int n_wgroups, int len) {

	queue.submit([&](sycl::handler &cgh) {
		sycl::accessor<T, 1, sycl::access::mode::read_write,
				sycl::access::target::local>
				local_mem(sycl::range<1>(wgroup_size+32), cgh);
		auto global_mem = buf.template get_access<sycl::access::mode::read>(cgh);
		auto result = output.template get_access<sycl::access::mode::write>(cgh);
		cgh.parallel_for<class single_dot_kernel>(
				sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
				[=](sycl::nd_item<1> item) {
					int id = item.get_local_id(0);
					int grouId = item.get_group_linear_id();
					int myGlobalReductionIndex = grouId * (wgroup_size * 2) + id;
					int gridSize = wgroup_size * 2 * n_wgroups;
					local_mem[id] = 0.0;
					local_mem[id+32] = 0.0;

					while (myGlobalReductionIndex < len) {
						local_mem[id] += pow(global_mem[myGlobalReductionIndex], 2) +
										 pow(global_mem[myGlobalReductionIndex + wgroup_size], 2);
						myGlobalReductionIndex += gridSize;
					}
					item.barrier();
					if (wgroup_size >= 32) {
						local_mem[id] += local_mem[id + 16];
						item.barrier();
					}
					if (wgroup_size >= 16)
						local_mem[id] += local_mem[id + 8];
					if (wgroup_size >= 8)
						local_mem[id] += local_mem[id + 4];
					if (wgroup_size >= 4)
						local_mem[id] += local_mem[id + 2];
					if (wgroup_size >= 2)
						local_mem[id] += local_mem[id + 1];
					if (id == 0)
						result[grouId] = local_mem[0];
				});
	});
}

template<typename T>
void CallDotDoubleBufferKernel(sycl::queue &queue, int wgroup_size, sycl::buffer<T, 1> &buf, sycl::buffer<T, 1> &buf2,
							   sycl::buffer<T, 1> &output,
							   int n_wgroups, int len) {

	queue.submit([&](sycl::handler &cgh) {
		sycl::accessor<T, 1, sycl::access::mode::read_write,
				sycl::access::target::local>
				local_mem(sycl::range<1>(wgroup_size+32), cgh);
		auto global_mem = buf.template get_access<sycl::access::mode::read>(cgh);
		auto global_mem2 = buf2.template get_access<sycl::access::mode::read>(cgh);
		auto result = output.template get_access<sycl::access::mode::write>(cgh);
		cgh.parallel_for<class double_dot_kernel>(
				sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
				[=](sycl::nd_item<1> item) {
					int id = item.get_local_id(0);
					int grouId = item.get_group_linear_id();
					int myGlobalReductionIndex = grouId * (wgroup_size * 2) + id;
					int gridSize = wgroup_size * 2 * n_wgroups;
					local_mem[id] = 0.0;
					local_mem[id+32] = 0.0;

					while (myGlobalReductionIndex < len) {
						local_mem[id] += global_mem[myGlobalReductionIndex] * global_mem2[myGlobalReductionIndex] +
										 global_mem[myGlobalReductionIndex + wgroup_size] *
										 global_mem2[myGlobalReductionIndex + wgroup_size];
						myGlobalReductionIndex += gridSize;
					}
					item.barrier();
					if (wgroup_size >= 32) {
						local_mem[id] += local_mem[id + 16];
						item.barrier();
					}
					if (wgroup_size >= 16)
						local_mem[id] += local_mem[id + 8];
					if (wgroup_size >= 8)
						local_mem[id] += local_mem[id + 4];
					if (wgroup_size >= 4)
						local_mem[id] += local_mem[id + 2];
					if (wgroup_size >= 2)
						local_mem[id] += local_mem[id + 1];
					if (id == 0)
						result[grouId] = local_mem[0];
				});
	});
}


int ReturnArrayWithPad(double *&arr, int *originalLength, int modulo) {
	int fillers = modulo - (*originalLength) % modulo;
	int originalEnd = *originalLength;
	*originalLength = *originalLength + fillers;

	if(dotProductArrays.find(*originalLength) == dotProductArrays.end()) {
		arr = new double[*originalLength];
		dotProductArrays.insert(std::pair<int,double *>(*originalLength, arr));
	}
	arr=dotProductArrays[*originalLength];

	for (int j = 0; j < fillers; ++j) {
		arr[originalEnd + j] = 0;
	}
	return originalEnd;
}

/*!
  Routine to compute the dot product of two vectors where:

  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] x, y the input vectors
  @param[in] result a pointer to scalar value, on exit will contain result.
  @param[out] time_allreduce the time it took to perform the communication between processes

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct
*/
int ComputeDotProduct_SyCL(const local_int_t n, const Vector &x, const Vector &y,
						   double &result, double &time_allreduce) {
	assert(x.localLength >= n); // Test vector lengths
	assert(y.localLength >= n);
	auto wgroup_size = 32;
	auto part_size = wgroup_size * 2;
	double local_result = 0.0;
	double *xv = x.values;
	double *yv = y.values;


//	sycl::buffer<double, 1> x_buf(x.values, sycl::range<1>(x.paddedLength));
//	sycl::buffer<double, 1> y_buf(y.values, sycl::range<1>(x.paddedLength));
	auto x_buf = *bufferFactory.GetBuffer(x.values, sycl::range<1>(x.paddedLength));
	auto y_buf = *bufferFactory.GetBuffer(y.values, sycl::range<1>(y.paddedLength));
	auto len = x.paddedLength;
	int extraMembers = 0;
	double *output;
	local_int_t newLength;
	int initialLoop = 1;
	while (len != 1) {
		while (len < part_size) {
			part_size = wgroup_size;
			wgroup_size /= 2;
		}
		auto n_wgroups = (len + part_size - 1) / part_size;
		if (n_wgroups != 1) {
			newLength = n_wgroups;
			ReturnArrayWithPad(output, &newLength, part_size);
		} else {
			if(dotProductArrays.find(1) == dotProductArrays.end()) {
				output = new double[1];
				dotProductArrays.insert(std::pair<int,double *>(1,output));
			}
			output=dotProductArrays[1];
			newLength = 1;
		}
		auto result1=*dotFactory.GetBuffer(output, sycl::range<1>(newLength));

		if (initialLoop) {
			if (yv == xv)
				CallDotSingleBufferKernel<double>(queue, wgroup_size, x_buf, result1, n_wgroups, len);
			else
				CallDotDoubleBufferKernel<double>(queue, wgroup_size, x_buf, y_buf, result1, n_wgroups, len);
		} else {
			CallReductionKernel<double>(queue, wgroup_size, x_buf, result1, n_wgroups, len);
		}
		initialLoop = 0;
//		auto access = result1.get_access<sycl::access::mode::read>();
		x_buf = *dotFactory.GetBuffer(output, sycl::range<1>(newLength));
		len = newLength;
	}
	if(dotAccess){
		auto access = x_buf.get_access<sycl::access::mode::read>();
		result = access[0];
	}
//
////	if (yv == xv) {
////#ifndef HPCG_NO_OPENMP
////#pragma omp parallel for reduction (+:local_result)
////#endif
////		for (local_int_t i = 0; i < n; i++) local_result += xv[i] * xv[i];
////	} else {
////#ifndef HPCG_NO_OPENMP
////#pragma omp parallel for reduction (+:local_result)
////#endif
////		for (local_int_t i = 0; i < n; i++) local_result += xv[i] * yv[i];
////	}
////
//
//#ifndef HPCG_NO_MPI
//	// Use MPI's reduce function to collect all partial sums
//	double t0 = mytimer();
//	double global_result = 0.0;
//	MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
//				  MPI_COMM_WORLD);
//	result = global_result;
//	time_allreduce += mytimer() - t0;
//#else
//	time_allreduce += 0.0;
//	result = local_result;
//#endif

	return 0;
}



//
////@HEADER
//// ***************************************************
////
//// HPCG: High Performance Conjugate Gradient Benchmark
////
//// Contact:
//// Michael A. Heroux ( maherou@sandia.gov)
//// Jack Dongarra     (dongarra@eecs.utk.edu)
//// Piotr Luszczek    (luszczek@eecs.utk.edu)
////
//// ***************************************************
////@HEADER
//
///*!
// @file ComputeDotProduct_ref.cpp
//
// HPCG routine
// */
//#include "SyCLResources.h"
//
//#ifndef HPCG_NO_MPI
//
//#include <mpi.h>
//#include "mytimer.hpp"
//
//#endif
//#ifndef HPCG_NO_OPENMP
//
//#include <omp.h>
//
//#endif
//
//#include <cassert>
//#include "ComputeDotProduct_SyCL.hpp"
//
//
//template<typename T>
//void CallReductionKernel(sycl::queue &queue, int wgroup_size, sycl::buffer<T, 1> &buf, sycl::buffer<T, 1> &output,
//						 int n_wgroups, int len) {
//	queue.submit([&](sycl::handler &cgh) {
//		sycl::accessor<T, 1, sycl::access::mode::read_write,
//				sycl::access::target::local>
//				local_mem(sycl::range<1>(wgroup_size + 32), cgh);
//		auto global_mem = buf.template get_access<sycl::access::mode::read>(cgh);
//		auto result = output.template get_access<sycl::access::mode::write>(cgh);
//		cgh.parallel_for<class dot_kernel>(
//				sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
//				[=](sycl::nd_item<1> item) {
//					int id = item.get_local_id(0);
//					int grouId = item.get_group_linear_id();
//					int myGlobalReductionIndex = grouId * (wgroup_size * 2) + id;
//					int gridSize = wgroup_size * 2 * n_wgroups;
//					local_mem[id] = 0.0;
//					local_mem[id + 32] = 0.0;
//					item.barrier();
//
////						while (myGlobalReductionIndex < len) {
//					local_mem[id] += global_mem[myGlobalReductionIndex] +
//									 global_mem[myGlobalReductionIndex + wgroup_size];
////							myGlobalReductionIndex += gridSize;
////						}
//					item.barrier();
//					if (id < 16) {
//						local_mem[id] += local_mem[id + 16];
//						item.barrier();
//					}
//					if (id < 8)
//						local_mem[id] += local_mem[id + 8];
//					if (id < 4)
//						local_mem[id] += local_mem[id + 4];
//					if (id < 2)
//						local_mem[id] += local_mem[id + 2];
//					if (id < 1)
//						local_mem[id] += local_mem[id + 1];
//					if (id == 0)
//						result[grouId] = local_mem[0];
//				});
//	});
//}
//
//template<typename T>
//void CallDotSingleBufferKernel(sycl::queue &queue, int wgroup_size, sycl::buffer<T, 1> &buf, sycl::buffer<T, 1> &output,
//							   int n_wgroups, int len) {
//
//	queue.submit([&](sycl::handler &cgh) {
//		sycl::accessor<T, 1, sycl::access::mode::read_write,
//				sycl::access::target::local>
//				local_mem(sycl::range<1>(wgroup_size + 32), cgh);
//		auto global_mem = buf.template get_access<sycl::access::mode::read>(cgh);
//		auto result = output.template get_access<sycl::access::mode::write>(cgh);
//		cgh.parallel_for<class single_dot_kernel>(
//				sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
//				[=](sycl::nd_item<1> item) {
//					int id = item.get_local_id(0);
//					int grouId = item.get_group_linear_id();
//					int myGlobalReductionIndex = grouId * (wgroup_size * 2) + id;
//					int gridSize = wgroup_size * 2 * n_wgroups;
//					local_mem[id] = 0.0;
//					local_mem[id + 32] = 0.0;
//					item.barrier();
//
////						while (myGlobalReductionIndex < len) {
//					local_mem[id] += pow(global_mem[myGlobalReductionIndex], 2) +
//									 pow(global_mem[myGlobalReductionIndex + wgroup_size], 2);
////						myGlobalReductionIndex += gridSize;
////						}
//					item.barrier();
//					if (id < 16) {
//						local_mem[id] += local_mem[id + 16];
//						item.barrier();
//					}
//					if (id < 8)
//						local_mem[id] += local_mem[id + 8];
//					if (id < 4)
//						local_mem[id] += local_mem[id + 4];
//					if (id < 2)
//						local_mem[id] += local_mem[id + 2];
//					if (id < 1)
//						local_mem[id] += local_mem[id + 1];
//					if (id == 0)
//						result[grouId] = local_mem[0];
//				});
//	});
//}
//
//template<typename T>
//void CallDotDoubleBufferKernel(sycl::queue &queue, int wgroup_size, sycl::buffer<T, 1> &buf, sycl::buffer<T, 1> &buf2,
//							   sycl::buffer<T, 1> &output,
//							   int n_wgroups, int len) {
//
//	queue.submit([&](sycl::handler &cgh) {
//		sycl::accessor<T, 1, sycl::access::mode::read_write,
//				sycl::access::target::local>
//				local_mem(sycl::range<1>(wgroup_size + 32), cgh);
//		auto global_mem = buf.template get_access<sycl::access::mode::read>(cgh);
//		auto global_mem2 = buf2.template get_access<sycl::access::mode::read>(cgh);
//		auto result = output.template get_access<sycl::access::mode::write>(cgh);
//		cgh.parallel_for<class double_dot_kernel>(
//				sycl::nd_range<1>(n_wgroups * wgroup_size, wgroup_size),
//				[=](sycl::nd_item<1> item) {
//					int id = item.get_local_id(0);
//					int grouId = item.get_group_linear_id();
//
//					int myGlobalReductionIndex = grouId * (wgroup_size * 2) + id;
//					int gridSize = wgroup_size * 2 * n_wgroups;
//					local_mem[id] = 0.0;
//					local_mem[id + 32] = 0.0;
//					item.barrier();
//
////						while (myGlobalReductionIndex < len) {
//					local_mem[id] +=
//							(global_mem[myGlobalReductionIndex] * global_mem2[myGlobalReductionIndex]) +
//							(global_mem[myGlobalReductionIndex + wgroup_size] *
//							 global_mem2[myGlobalReductionIndex + wgroup_size]);
////							myGlobalReductionIndex += gridSize;
////						}
//					item.barrier();
////						if (id >= 32) {
////							local_mem[id] += local_mem[id + 16];
////							item.barrier();
////						}
////						if (wgroup_size >= 16)
////							local_mem[id] += local_mem[id + 8];
////						if (wgroup_size >= 8)
////							local_mem[id] += local_mem[id + 4];
////						if (wgroup_size >= 4)
////							local_mem[id] += local_mem[id + 2];
////						if (wgroup_size >= 2)
////							local_mem[id] += local_mem[id + 1];
////						if (id == 0)
////							result[grouId] = local_mem[0];
//
//					if (id < 16) {
//						local_mem[id] += local_mem[id + 16];
//						item.barrier();
//					}
//					if (id < 8)
//						local_mem[id] += local_mem[id + 8];
//					if (id < 4)
//						local_mem[id] += local_mem[id + 4];
//					if (id < 2)
//						local_mem[id] += local_mem[id + 2];
//					if (id < 1)
//						local_mem[id] += local_mem[id + 1];
//					if (id == 0)
//						result[grouId] = local_mem[0];
//
//
//				});
//	});
//}
//
//
//int ReturnArrayWithPad(double *&arr, int *originalLength, int modulo) {
//	int fillers = modulo - (*originalLength) % modulo;
//	int originalEnd = *originalLength;
//	*originalLength = *originalLength + fillers;
//	arr = new double[*originalLength];
//	for (int j = 0; j < *originalLength; ++j) {
//		arr[j] = 0;
//	}
//	return originalEnd;
//}
//
///*!
//  Routine to compute the dot product of two vectors where:
//
//  This is the reference dot-product implementation.  It _CANNOT_ be modified for the
//  purposes of this benchmark.
//
//  @param[in] n the number of vector elements (on this processor)
//  @param[in] x, y the input vectors
//  @param[in] result a pointer to scalar value, on exit will contain result.
//  @param[out] time_allreduce the time it took to perform the communication between processes
//
//  @return returns 0 upon success and non-zero otherwise
//
//  @see ComputeDotProduct
//*/
//int ComputeDotProduct_SyCL(const local_int_t n, const Vector &x, const Vector &y,
//						   double &result, double &time_allreduce) {
//	assert(x.localLength >= n); // Test vector lengths
//	assert(y.localLength >= n);
//	auto wgroup_size = 32;
//	auto part_size = wgroup_size * 2;
//	double local_result = 0.0;
//	double *xv = x.values;
//	double *yv = y.values;
////	std::cout << n << std::endl;
////	std::cout << x.paddedLength << std::endl;
////	std::cout << y.paddedLength << std::endl;
//
//	auto x_buf = *bufferFactory.GetBuffer(x.values, sycl::range<1>(x.paddedLength));
//	auto y_buf = *bufferFactory.GetBuffer(y.values, sycl::range<1>(y.paddedLength));
//
////	sycl::buffer<double ,1> x_buf(x.values,sycl::range<1>(x.paddedLength));
////	sycl::buffer<double ,1> y_buf(y.values,sycl::range<1>(y.paddedLength));
//
//	auto len = x.paddedLength;
//	double *output;
//	local_int_t newLength;
//	int initialLoop = 1;
//	while (len != 1) {
//
//		while (len < part_size) {
//			part_size = wgroup_size;
//			wgroup_size /= 2;
//		}
//		auto n_wgroups = (len + part_size - 1) / part_size;
//
//		if (n_wgroups != 1) {
//			newLength = n_wgroups;
//			ReturnArrayWithPad(output, &newLength, part_size);
////			std::cout << newLength<<" | "<<n_wgroups<<" | "<<len << std::endl;
//		} else {
//			output = new double[1];
//			output[0] = 0;
//			newLength = 1;
//		}
//		sycl::buffer<double, 1> result1(output, sycl::range<1>(newLength));
//		if (initialLoop) {
//			if (yv == xv) {
//				CallDotSingleBufferKernel<double>(queue, wgroup_size, x_buf, result1, n_wgroups, len);
//			}
//			else {
//				CallDotDoubleBufferKernel<double>(queue, wgroup_size, x_buf, y_buf, result1, n_wgroups, len);
//			}
//		} else {
//			CallReductionKernel<double>(queue, wgroup_size, x_buf, result1, n_wgroups, len);
//		}
//		initialLoop = 0;
//		x_buf = result1;
////		auto access = x_buf.get_access<sycl::access::mode::read>();
////		for (int i = n_wgroups; i <newLength ; ++i) {
//////			if(access[i]!=0){
////				std::cout<<access[i]<<std::endl;
//////			}
////		}
//		len = newLength;
//	}
//	auto access = x_buf.get_access<sycl::access::mode::read>();
//	local_result=access[0];
////
////	for (int i = 0; i < len; ++i) {
////	}
////	local_result = access[0];
//
//#ifndef HPCG_NO_MPI
//	// Use MPI's reduce function to collect all partial sums
//	double t0 = mytimer();
//	double global_result = 0.0;
//	MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
//				  MPI_COMM_WORLD);
//	result = global_result;
//	time_allreduce += mytimer() - t0;
//#else
//	time_allreduce += 0.0;
//	result = local_result;
//#endif
//
//	return 0;
//}
