
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
 @file ComputeProlongation_ref.cpp

 HPCG routine
 */
//#include <CL/sycl.hpp>
#include "SyCLResources.h"

#ifndef HPCG_NO_OPENMP

#include <omp.h>

#endif

#include <iostream>
#include "ComputeProlongation_SyCL.hpp"
#include "SaveArrays.h"

/*!
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeProlongation_SyCL(const SparseMatrix &Af, Vector &xf) {
	auto f2c_buf = *Af.mgData->f2cOperator;


	local_int_t nc = Af.mgData->rc->localLength;
	auto xfv_buf = xf.buf;
	auto xcv_buf = Af.mgData->xc->buf;
	{
		queue.submit([&](sycl::handler &cgh) {
			auto xfv_acc = xfv_buf.get_access<sycl::access::mode::write>(cgh);
			auto xcv_acc = xcv_buf.get_access<sycl::access::mode::read>(cgh);
			auto f2c_acc = f2c_buf.get_access<sycl::access::mode::read>(cgh);
			cgh.parallel_for<class prolongation>(
					sycl::nd_range<1>(nc, 32),
					[=](sycl::nd_item<1> item) {
						int id = item.get_global_linear_id();
//						if (id < nc)
							xfv_acc[f2c_acc[id]] += xcv_acc[id];
					});
		});
	}
	if (doAccess)
		auto access = xfv_buf.get_access<sycl::access::mode::read>();

	return 0;
}
