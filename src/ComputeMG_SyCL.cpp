
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
 @file ComputeSYMGS_ref.cpp

 HPCG routine
 */

#include "ComputeMG_SyCL.hpp"

#include "ComputeSYMGS_ref.hpp"
#include "ComputeSPMV_ref.hpp"
#include "ComputeRestriction_ref.hpp"
#include "ComputeProlongation_ref.hpp"
#include "ComputeRestriction_SyCL.hpp"
#include "ComputeProlongation_SyCL.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"
#include <cassert>
#include <iostream>
#include "SyCLResources.h"




//
//void SyCLZeroVector(Vector_STRUCT &x){
//	local_int_t size=x.paddedLength;
//	auto x_buf=*x.buf;
//	{
//
//		queue.submit([&](sycl::handler &cgh) {
//			auto x_acc = x_buf.get_access<sycl::access::mode::write>(cgh);
//			cgh.parallel_for<class prolongation>(
//					sycl::nd_range<1>(size, 32),
//					[=](sycl::nd_item<1> item) {
//						size_t i=item.get_global_linear_id();
//						if(i<size)
//							x_acc[i] =0;
//					});
//		});
//	}
//	x_buf.get_access<sycl::access::mode::read>();
//}

/*!

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG
*/
int ComputeMG_SyCL(SparseMatrix &A, const Vector &r, Vector &x) {
	assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values

	SyCLZeroVector(x);
	goEasyOnFwd=true;
//	ZeroVector(x); // initialize x to zero

	int ierr = 0;
	if (A.mgData != 0) { // Go to next coarse level if defined
		int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
		for (int i = 0; i < numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
		if (ierr != 0) return ierr;
		ierr = ComputeSPMV(A, x, *A.mgData->Axf);
		if (ierr != 0) return ierr;
		// Perform restriction operation using simple injection

#if defined(SyCL_RESTRICTION)
		ierr = ComputeRestriction_SyCL(A, r);
#else
		ierr = ComputeRestriction_ref(A, r);
#endif
		if (ierr != 0) return ierr;
		ierr = ComputeMG_SyCL(*A.Ac, *A.mgData->rc, *A.mgData->xc);
		if (ierr != 0) return ierr;
#if defined(SyCL_PROLONGATION)
		ierr = ComputeProlongation_SyCL(A, x);
#else
		ierr = ComputeProlongation_ref(A, x);
#endif
//		ierr = ComputeProlongation_ref(A, x);
		if (ierr != 0) return ierr;
		int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
		for (int i = 0; i < numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
		if (ierr != 0) return ierr;
	} else {
		ierr = ComputeSYMGS(A, r, x);
		if (ierr != 0) return ierr;
	}
	return 0;
}

