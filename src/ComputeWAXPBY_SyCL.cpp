
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

	double *xv = x.values;
	double *yv = y.values;
	double *const wv = w.values;
//	saveArray(xv,n,"waxpby_xv");
//	saveArray(yv,n,"waxpby_yv");

//	std::cout<< n<<std::endl;

//	if (alpha == 1.0) {
//#ifndef HPCG_NO_OPENMP
//#pragma omp parallel for
//#endif
//		for (local_int_t i = 0; i < n; i++) wv[i] = xv[i] + beta * yv[i];
//	} else if (beta == 1.0) {
//#ifndef HPCG_NO_OPENMP
//#pragma omp parallel for
//#endif
//		for (local_int_t i = 0; i < n; i++) wv[i] = alpha * xv[i] + yv[i];
//	} else {
//#ifndef HPCG_NO_OPENMP
//#pragma omp parallel for
//#endif
//		for (local_int_t i = 0; i < n; i++) wv[i] = alpha * xv[i] + beta * yv[i];
//	}
	auto xv_buf = *bufferFactory.GetBuffer(xv, sycl::range<1>(x.paddedLength));
	auto yv_buf = *bufferFactory.GetBuffer(yv, sycl::range<1>(y.paddedLength));
	auto wv_buf = *bufferFactory.GetBuffer(wv, sycl::range<1>(w.paddedLength));
//	sycl::buffer<double, 1> xv_buf(xv, sycl::range<1>(n));
//	sycl::buffer<double, 1> yv_buf(yv, sycl::range<1>(n));
//	sycl::buffer<double, 1> wv_buf(wv, sycl::range<1>(n));
	{
		queue.submit([&](sycl::handler &cgh) {
			auto xv_acc = xv_buf.get_access<sycl::access::mode::read>(cgh);
			auto yv_acc = yv_buf.get_access<sycl::access::mode::read>(cgh);
			auto wv_acc = wv_buf.get_access<sycl::access::mode::write>(cgh);

			auto kernelNoAlpha = [=](sycl::nd_item<1> item) {
				int i = item.get_global_linear_id();
				if (i < n)
					wv_acc[i] = xv_acc[i] + beta * yv_acc[i];
			};
			auto kernelNoBeta = [=](sycl::nd_item<1> item) {
				int i = item.get_global_linear_id();
				if (i < n)
					wv_acc[i] = alpha * xv_acc[i] + yv_acc[i];
			};
			auto kernelBoth = [=](sycl::nd_item<1> item) {
				int i = item.get_global_linear_id();
				if (i < n)
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
	auto access = wv_buf.get_access<sycl::access::mode::read>();
//	saveArray(wv,n,"waxpby_wv");
//	exit(1);
	return 0;
}
