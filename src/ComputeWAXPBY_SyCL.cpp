
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
 @file ComputeWAXPBY_ref.cpp

 HPCG routine
 */

#include "ComputeWAXPBY_SyCL.hpp"
#include "SaveArrays.h"
#include "SyCLResources.h"

#ifndef HPCG_NO_OPENMP

#include <omp.h>

#endif

#include <cassert>
#include <iostream>

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This is the reference WAXPBY impmentation.  It CANNOT be modified for the
  purposes of this benchmark.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY
*/
int ComputeWAXPBY_SyCL(const local_int_t n, const double alpha, const Vector &x,
					   const double beta, const Vector &y, Vector &w) {

	assert(x.localLength >= n); // Test vector lengths
	assert(y.localLength >= n);

	auto xv_buf = *x.buf;
	auto yv_buf = *y.buf;
	auto wv_buf = *w.buf;

	{
		queue.submit([&](sycl::handler &cgh) {
			auto xv_acc = xv_buf.get_access<sycl::access::mode::read>(cgh);
			auto yv_acc = yv_buf.get_access<sycl::access::mode::read>(cgh);
			auto wv_acc = wv_buf.get_access<sycl::access::mode::write>(cgh);

			auto kernelNoAlpha = [=](sycl::nd_item<1> item) {
				int i = item.get_global_linear_id();
					wv_acc[i] = xv_acc[i] + beta * yv_acc[i];
			};
			auto kernelNoBeta = [=](sycl::nd_item<1> item) {
				int i = item.get_global_linear_id();
					wv_acc[i] = alpha * xv_acc[i] + yv_acc[i];
			};
			auto kernelBoth = [=](sycl::nd_item<1> item) {
				int i = item.get_global_linear_id();
					wv_acc[i] = alpha * xv_acc[i] + beta * yv_acc[i];
			};
			if (alpha == 1.0)
				cgh.parallel_for<class noalpha>(
						sycl::nd_range<1>(n, 32), kernelNoAlpha);
			else if (beta == 1.0)
				cgh.parallel_for<class nobeta>(
						sycl::nd_range<1>(n, 32), kernelNoBeta);
			else
				cgh.parallel_for<class both>(
						sycl::nd_range<1>(n, 32), kernelBoth);
		});
	}
	if (doAccess)
		auto access = wv_buf.get_access<sycl::access::mode::read>();
//	saveArray(wv,n,"waxpby_wv");
//	exit(1);
	return 0;
}
