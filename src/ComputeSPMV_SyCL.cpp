
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
 @file ComputeSPMV_ref.cpp

 HPCG routine
 */
#include "SyCLResources.h"

#include "ComputeSPMV_SyCL.hpp"
//#include <SYCL/sycl.hpp>

#ifndef HPCG_NO_MPI

#include "ExchangeHalo.hpp"

#endif

#ifndef HPCG_NO_OPENMP

#include <omp.h>

#endif

#include <cassert>
#include <fstream>
#include <iostream>

template<int wgroup_size>
void ThreadPerRowSpMV(sycl::queue &queue, sycl::buffer<double, 2> &matrixbuf, sycl::buffer<int, 2> &mtxIndLbuf,
					  sycl::buffer<char, 1> &nonzerosinrowBuf, sycl::buffer<double, 1> &xBuf,
					  sycl::buffer<double, 1> &groups,
					  const int rows) {
	queue.submit([&](sycl::handler &cgh) {
					 auto matrixMem = matrixbuf.get_access<sycl::access::mode::read>(cgh);
					 auto mtxIndLMem = mtxIndLbuf.get_access<sycl::access::mode::read>(cgh);
					 auto nonZeros = nonzerosinrowBuf.get_access<sycl::access::mode::read>(cgh);
					 auto xvAccessor = xBuf.get_access<sycl::access::mode::read>(cgh);
					 auto results = groups.get_access<sycl::access::mode::discard_write>(cgh);
					 if (!transpose)
						 cgh.parallel_for<class spmv_kernel>(
								 sycl::nd_range<1>(rows, wgroup_size),
								 [=](sycl::nd_item<1> item) {
									 size_t globalLinearId = item.get_global_linear_id();
									 double sum = 0;
									 for (int i = 0; i < nonZeros[globalLinearId]; ++i) {
										 sum += matrixMem[globalLinearId][i] * xvAccessor[mtxIndLMem[globalLinearId][i]];
									 }
									 results[globalLinearId] = sum;
								 });
					 else {
//						 int size = 1;
//						 int val = wgroup_size * (size - 1);
//						 cgh.parallel_for<class transposed_spmv_kernel>(
//								 sycl::nd_range<1>((rows / size), 64),
//								 [=](sycl::nd_item<1> item) {
//									 for (int j = 0; j < size; ++j) {
//										 size_t globalLinearId =
//												 item.get_global_linear_id() + (item.get_global_linear_id() / (wgroup_size)) * val +
//												 j * wgroup_size;
//										 double sum = 0;
//										 for (int i = 0; i < nonZeros[globalLinearId]; ++i) {
//											 sum += matrixMem[i][globalLinearId] * xvAccessor[mtxIndLMem[i][globalLinearId]];
//										 }
//										 results[globalLinearId] = sum;
//									 }
//									 size_t ids[4];
//									 for (int j = 0; j < size; ++j) {
//										 ids[j] =
//												 item.get_global_linear_id() + (item.get_global_linear_id() / (wgroup_size)) * val +
//												 j * wgroup_size;
//									 }
//									 double sum[4] = {0, 0, 0, 0};
//									 for (int i = 0; i < 27; ++i) {
//										 sum[0] += matrixMem[i][ids[0]] * xvAccessor[mtxIndLMem[i][ids[0]]];
//										 sum[1] += matrixMem[i][ids[1]] * xvAccessor[mtxIndLMem[i][ids[1]]];
//										 sum[2] += matrixMem[i][ids[2]] * xvAccessor[mtxIndLMem[i][ids[2]]];
//										 sum[3] += matrixMem[i][ids[3]] * xvAccessor[mtxIndLMem[i][ids[3]]];
//									 }
//									 results[ids[0]] = sum[0];
//									 results[ids[1]] = sum[1];
//									 results[ids[2]] = sum[2];
//									 results[ids[3]] = sum[3];

//
//									 for (int j = 0; j < size; ++j) {
//										 results[ids[j]] = sum[j];
//									 }


//								 });
			cgh.parallel_for<class transposed_spmv_kernel>(
					sycl::nd_range<1>(rows, wgroup_size),
					[=](sycl::nd_item<1> item) {
						size_t globalLinearId = item.get_global_linear_id();
						double sum = 0;
						for (int i = 0; i < nonZeros[globalLinearId]; ++i) {
							sum += matrixMem[i][globalLinearId] * xvAccessor[mtxIndLMem[i][globalLinearId]];
						}
						results[globalLinearId] = sum;
					});
//		}
//	});
					 }
				 }

	);
}

/*!
  Routine to compute matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This is the reference SPMV implementation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV
*/
int ComputeSPMV_SyCL(const SparseMatrix &A, Vector &x, Vector &y) {

	assert(x.localLength >= A.localNumberOfColumns); // Test vector lengths
	assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif
	const local_int_t nrow = A.localNumberOfRows;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
	auto groups = y.buf;
	auto mtxIndLbuf = A.mtxIndLB;
	auto matrixbuf = A.matrixValuesB;
	if (transpose) {
		matrixbuf = A.matrixValuesBT;
		mtxIndLbuf = A.mtxIndLBT;
	}
	auto xBuf = x.buf;
	auto nonzerosinrowBuf = A.nonzerosInRow;
	ThreadPerRowSpMV<32>(queue, matrixbuf, mtxIndLbuf, nonzerosinrowBuf, xBuf, groups, nrow);
	if (doAccess)
		auto access = groups.get_access<sycl::access::mode::read>();

	return 0;
}
