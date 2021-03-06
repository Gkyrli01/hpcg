
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

#ifndef HPCG_NO_MPI

#include "ExchangeHalo.hpp"

#endif

#include "ComputeSYMGS_SyCL.hpp"
#include "SaveArrays.h"
#include "SyCLResources.h"
#include "ComputeSYMGS_ref.hpp"
#include <cassert>
#include <iostream>
#ifndef SYMGSSIZE
#define SYMGSSIZE 32
#endif

/*!
  Computes one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  x should be initially zero on the first GS sweep, but we do not attempt to exploit this fact.
  - We then perform one back sweep.
  - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.


  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSYMGS
*/
int ComputeSYMGS_SyCL(SparseMatrix &A, const Vector &r, Vector &x) {

	assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values
	if (!A.optimizationData) {
		std::cout << "Rows are: " << A.localNumberOfRows << std::endl;
		return ComputeSYMGS_ref(A, r, x);
	}

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
	auto *permutation = static_cast<local_int_t *>(A.optimizationData);
	int allcolors = A.allColors;
	int *numberOfColors = static_cast<int *>(A.numberOfColors);
	auto xv_buf = x.buf;
	auto rv_buf = r.buf;
	auto matrixDiagonal_buf = A.matrixDiagonalSYMGS;
	auto matrix_buf = A.matrixValuesB;
	auto mtxIndL_buf = A.mtxIndLB;

	if(using_reordering) {
#ifdef TRANSPOSE
		matrix_buf = A.matrixValuesBT;
		mtxIndL_buf = A.mtxIndLBT;
#endif
	}
	auto nonzerosinrow_buf = A.nonzerosInRow;
	//Keep for now
	auto permutation_buf = *(bufferFactory.GetBuffer(permutation, sycl::range<1>(nrow)));
	{
		for (int currentColor = 0; currentColor < allcolors * 2; ++currentColor) {
			queue.submit([&](sycl::handler &cgh) {
				auto xv_acc = xv_buf.get_access<sycl::access::mode::read_write>(cgh);
				auto rv_acc = (rv_buf).get_access<sycl::access::mode::read>(cgh);
				auto nonzerosinrow_acc = nonzerosinrow_buf.get_access<sycl::access::mode::read>(cgh);

				auto matrix_acc = matrix_buf.get_access<sycl::access::mode::read>(cgh);
				auto mtxIndL_acc = mtxIndL_buf.get_access<sycl::access::mode::read>(cgh);
				int tmpCol = currentColor;
				//Provides forward and backward sweeps
				if (currentColor >= allcolors) {
					tmpCol = (allcolors - 1) - currentColor % allcolors;
				}
				const int col = tmpCol;
				int tmp = 0;
				for (int k = 0; k < col; ++k) {
					tmp += numberOfColors[k];
				}
				const int offset = tmp;
				const int items = numberOfColors[col];
				if (!using_reordering) {
					auto matrixDiagonal_acc = matrixDiagonal_buf.get_access<sycl::access::mode::read>(cgh);
					auto permutation_acc = permutation_buf.get_access<sycl::access::mode::read>(cgh);
					cgh.parallel_for<class symgs>(
							sycl::nd_range<1>(items, SYMGSSIZE),
							[=](sycl::nd_item<1> item) {
								if (item.get_global_linear_id() < items) {
									unsigned long z = item.get_global_linear_id() + offset;
									auto i = permutation_acc[z];
									const double currentDiagonal = matrixDiagonal_acc[i]; // Current diagonal value
									double sum = rv_acc[i]; // RHS value
									for (int j = 0; j < nonzerosinrow_acc[i]; j++) {
										sum -= matrix_acc[i][j] * xv_acc[mtxIndL_acc[i][j]];
									}
									sum += xv_acc[i] * currentDiagonal;
									xv_acc[i] = (sum) / currentDiagonal;
								}
							});
				} else {
					if (goEasyOnFwd&&currentColor<allcolors) {
						cgh.parallel_for<class symgsTranposedFwD>(
								sycl::nd_range<1>(items, SYMGSSIZE),
								[=](sycl::nd_item<1> item) {
									if (item.get_global_linear_id() < items) {
										unsigned long i = item.get_global_linear_id() + offset;
										double sum = rv_acc[i]; // RHS value
										double val;int col;
										for (int j = 0; j < nonzerosinrow_acc[i]; j++) {
#ifdef TRANSPOSE
											col = mtxIndL_acc[j][i];
#else
											col = mtxIndL_acc[i][j];
#endif
											if (col < i) {
#ifdef TRANSPOSE
												val = matrix_acc[j][i];
#else
												val = matrix_acc[i][j];
#endif
												sum -= val * xv_acc[col];
											}
											else {
												if (col == i) {
#ifdef TRANSPOSE
													val = matrix_acc[j][i];
#else
													val = matrix_acc[i][j];
#endif
													xv_acc[i] = (sum) / val;
													break;
												}
											}
										}
									}
								});
					} else {
						cgh.parallel_for<class symgsTranposed>(
								sycl::nd_range<1>(items, SYMGSSIZE),
								[=](sycl::nd_item<1> item) {
									if (item.get_global_linear_id() < items) {
										unsigned long i = item.get_global_linear_id() + offset;
										double sum = rv_acc[i]; // RHS value
										double currentDiagonal = 0;
										for (int j = 0; j < nonzerosinrow_acc[i]; j++) {
#ifdef TRANSPOSE
											auto col = mtxIndL_acc[j][i];
											auto val = matrix_acc[j][i];
#else
											auto col = mtxIndL_acc[i][j];
											auto val = matrix_acc[i][j];
#endif
											auto check=col != i;
											if (check)
												sum =sum- val * xv_acc[col];
											else
												currentDiagonal = val;
										}
										xv_acc[i] = (sum) / currentDiagonal;
									}
								});
					}

				}
			});
		}
	}
	goEasyOnFwd= false;
	if (doAccess)
		auto access = xv_buf.get_access<sycl::access::mode::read>();
	return 0;
}

