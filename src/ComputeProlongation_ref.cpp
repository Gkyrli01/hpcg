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

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeProlongation_ref.hpp"

/*!
  Routine to compute the coarse residual vector.
  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.
  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.
  @return Returns zero on success and a non-zero value otherwise.
*/
int ComputeProlongation_ref(const SparseMatrix & Af, Vector & xf) {

	double * xfv = xf.values;
	double * xcv = Af.mgData->xc->values;

	auto access=Af.mgData->f2cOperator->get_access<sycl::access::mode::read>();
	local_int_t * f2c=access.get_pointer();
	local_int_t nc = Af.mgData->rc->localLength;

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
// TODO: Somehow note that this loop can be safely vectorized since f2c has no repeated indices
	for (local_int_t i=0; i<nc; ++i) xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize


	xf.buf.get_access<sycl::access::mode::write>();
	return 0;
}