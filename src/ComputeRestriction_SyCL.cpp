
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
 @file ComputeRestriction_ref.cpp

 HPCG routine
 */

//#include <CL/sycl.hpp>

#include "SyCLResources.h"

#ifndef HPCG_NO_OPENMP

#include <omp.h>

#endif

#include <iostream>
#include "ComputeRestriction_SyCL.hpp"
#include "SaveArrays.h"

/*!
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeRestriction_SyCL(const SparseMatrix &A, const Vector &rf) {
	local_int_t nc = A.mgData->rc->localLength;
	auto f2c_buf = A.mgData->f2cOperator;
	auto rfv_buf = rf.buf;
	auto axfv_buf = A.mgData->Axf->buf;
	auto results_buf = A.mgData->rc->buf;
	{
		queue.submit([&](sycl::handler &cgh) {
			auto axfv_acc = axfv_buf.get_access<sycl::access::mode::read>(cgh);
			auto rfv_acc = rfv_buf.get_access<sycl::access::mode::read>(cgh);
			auto f2c_acc = f2c_buf.get_access<sycl::access::mode::read>(cgh);
			auto results_acc = results_buf.get_access<sycl::access::mode::write>(cgh);
			cgh.parallel_for<class prolongation  >(
					sycl::nd_range<1>(nc, 8),
					[=](sycl::nd_item<1> item) {
						int i = item.get_global_linear_id();
						local_int_t f2c = f2c_acc[i];
						results_acc[i] = rfv_acc[f2c] - axfv_acc[f2c];
					});
		});
	}
	if (doAccess)
		auto access = results_buf.get_access<sycl::access::mode::read>();

	return 0;
}
