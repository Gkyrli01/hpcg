
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


void DoAccess(double *xv, double *rv, int nrow, local_int_t **pInt, double **pDouble, double **pDouble1,
			  char *string, int *pInt1) {
//	auto a1 = (doubleBuffers1D->GetBuffer(xv, sycl::range<1>(
//			nrow)))->get_access<sycl::access::mode::read>();
//	auto a2 = (doubleBuffers1D->GetBuffer(rv, sycl::range<1>(nrow)))->get_access<sycl::access::mode::read>();
//
//	auto a3 = (integerBuffers2D->GetBuffer(pInt, sycl::range<2>(nrow,27)))->get_access<sycl::access::mode::read>();
//	auto a4= (doubleBuffers2D->GetBuffer(pDouble, sycl::range<2>(nrow,27)))->get_access<sycl::access::mode::read>();
//	auto a5= (doubleBuffers2D->GetBuffer(pDouble1, sycl::range<2>(nrow,1)))->get_access<sycl::access::mode::read>();
//	auto a6 = (charBuffers1D->GetBuffer(string, sycl::range<1>(
//			nrow)))->get_access<sycl::access::mode::read>();
//	auto a7 = (integerBuffers1D->GetBuffer(pInt1, sycl::range<1>(nrow)))->get_access<sycl::access::mode::read>();

}

//template<typename T,typename T1, int dims,sycl::access::mode Mode>
//sycl::accessor<T1,dims,Mode> GetVals(T arr, sycl::range<dims> range,sycl::handler& cgh){
//
//	if (strcmp(typeid(T1).name(), "char") == 1) {
//		return charBuffers1D->GetBuffer(arr, range)->template get_access<Mode>(cgh);
//	} else if (typeid(T1).name(), "double") {
//		if (dims == 2) {
//			return doubleBuffers2D->GetBuffer(arr, range)->template get_access<Mode>(cgh);
//		} else {
//			return doubleBuffers1D->GetBuffer(arr, range)->template get_access<Mode>(cgh);
//		}
//	} else {
//		if (dims == 2) {
//			return integerBuffers2D->GetBuffer(arr, range)->template get_access<Mode>(cgh);
//		} else {
//			return integerBuffers1D->GetBuffer(arr, range)->template get_access<Mode>(cgh);
//		}
//	}
//}
//template<typename type, int dims>
//sycl::buffer<char, dims> GetBuffer(type *arr, sycl::range<dims> range) {
//	return sycl::buffer<type, dims>(arr, range);
//}


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
int ComputeSYMGS_SyCL(const SparseMatrix &A, const Vector &r, Vector &x) {

	assert(x.localLength == A.localNumberOfColumns); // Make sure x contain space for halo values
	if (!A.optimizationData) {
		std::cout << "Rows are: " << A.localNumberOfRows << std::endl;
		return ComputeSYMGS_ref(A, r, x);
	}

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
//	std::cout << nrow << std::endl;
	double *matrixDiagonal = A.matrixDiagonalSYMGS;  // An array of pointers to the diagonal entries A.matrixValues
	double *rv = r.values;
	double *xv = x.values;
	auto *permutation = static_cast<local_int_t *>(A.optimizationData);
	int allcolors = A.allColors;

//	DoAccess(xv, rv, nrow, A.mtxIndL, A.matrixValues, matrixDiagonal, A.nonzerosInRow, permutation);

//	std::cout << "All colors: " << allcolors << std::endl;


	int *numberOfColors = static_cast<int *>(A.numberOfColors);

//	sycl::buffer<char, 1> nonzerosinrow_buf(A.nonzerosInRow, sycl::range<1>(A.localNumberOfRows));
//	sycl::buffer<int, 1> permutation_buf(permutation, sycl::range<1>(nrow));
//	sycl::buffer<double, 1> xv_buf(xv, sycl::range<1>(nrow));
//
//	sycl::buffer<double, 1> rv_buf(rv, sycl::range<1>(nrow));

//	sycl::buffer<int, 2> mtxIndL_buf(*A.mtxIndL, sycl::range<2>(A.localNumberOfRows, 27));
//	sycl::buffer<double, 2> matrix_buf(*A.matrixValues, sycl::range<2>(A.localNumberOfRows, 27));
//	sycl::buffer<double, 1> matrixDiagonal_buf(A.matrixDiagonalSYMGS, sycl::range<1>(A.localNumberOfRows));

//	auto matrixDiagonal_buf = *(doubleBuffers1D->GetBuffer(A.matrixDiagonalSYMGS, sycl::range<1>(A.localNumberOfRows)));
//	auto matrix_buf = *(doubleBuffers2D->GetBuffer(A.matrixValues, sycl::range<2>(A.localNumberOfRows,27)));
//	auto mtxIndL_buf = *(integerBuffers2D->GetBuffer(A.mtxIndL, sycl::range<2>(A.localNumberOfRows,27)));
//	auto nonzerosinrow_buf = *(charBuffers1D->GetBuffer(A.nonzerosInRow,sycl::range<1>(A.localNumberOfRows)));
//	auto permutation_buf =* (integerBuffers1D->GetBuffer(permutation, sycl::range<1>(nrow)));
	auto xv_buf=*bufferFactory.GetBuffer(xv, sycl::range<1>(x.paddedLength));
	auto rv_buf=*bufferFactory.GetBuffer(rv, sycl::range<1>(r.paddedLength));

	auto matrixDiagonal_buf = *(bufferFactory.GetBuffer(A.matrixDiagonalSYMGS, sycl::range<1>(A.localNumberOfRows)));
	auto matrix_buf = *(bufferFactory.GetBuffer(A.matrixValues, sycl::range<2>(A.localNumberOfRows,27)));
	auto mtxIndL_buf = *(bufferFactory.GetBuffer(A.mtxIndL, sycl::range<2>(A.localNumberOfRows,27)));
	auto nonzerosinrow_buf = *(bufferFactory.GetBuffer(A.nonzerosInRow,sycl::range<1>(A.localNumberOfRows)));
	auto permutation_buf =* (bufferFactory.GetBuffer(permutation, sycl::range<1>(nrow)));


//	(*r.buf).get_access<sycl::access::mode::read>();
//	auto xv_buf = *(doubleBuffers1D->GetBuffer(xv, sycl::range<1>(nrow)));
//	auto rv_buf = *(doubleBuffers1D->GetBuffer(rv, sycl::range<1>(nrow)));
//	std::cout<< &nonzerosinrow_buf<<std::endl;

//GetAccessor()
//	auto matrixDiagonal_buf = *(doubleBuffers2D->GetBuffer(matrixDiagonal,sycl::range<2>(A.localNumberOfRows,1)));
//	auto matrix_buf = *(doubleBuffers2D->GetBuffer(A.matrixValues, sycl::range<2>(A.localNumberOfRows,27)));
//	auto mtxIndL_buf = *(integerBuffers2D->GetBuffer(A.mtxIndL, sycl::range<2>(A.localNumberOfRows,27)));



//	auto nonzerosinrow_acc = (charBuffers1D->GetBuffer(A.nonzerosInRow.data(), sycl::range<1>(A.localNumberOfRows)))->get_access<sycl::access::mode::read>();
//	auto matrixDiagonal_acc = (doubleBuffers2D->GetBuffer(matrixDiagonal, sycl::range<2>(A.localNumberOfRows,1)))->get_access<sycl::access::mode::read>();
//	auto matrix_acc = (doubleBuffers2D->GetBuffer(A.matrixValues, sycl::range<2>(A.localNumberOfRows,27)))->get_access<sycl::access::mode::read>();
//	auto mtxIndL_acc = (integerBuffers2D->GetBuffer(A.mtxIndL, sycl::range<2>(A.localNumberOfRows,27)))->get_access<sycl::access::mode::read>();
//	auto permutation_acc = (integerBuffers1D->GetBuffer(permutation, sycl::range<1>(nrow)))->get_access<sycl::access::mode::read>();

//	GetBuffer<char*,char,1>(A.nonzerosInRow,sycl::range<1>(A.localNumberOfRows));
	{
//		std::cout<<A.localNumberOfRows<<std::endl;
//		auto nonzerosinrow_buf =*GetBuffer(A.nonzerosInRow,sycl::range<1>(A.localNumberOfRows));

		for (int currentColor = 0; currentColor < allcolors*2; ++currentColor) {
			queue.submit([&](sycl::handler &cgh) {
				auto xv_acc = xv_buf.get_access<sycl::access::mode::read_write>(cgh);
				auto rv_acc = (rv_buf).get_access<sycl::access::mode::read>(cgh);
				auto nonzerosinrow_acc =nonzerosinrow_buf.get_access<sycl::access::mode::read>(cgh);
//				nonzerosinrow_buf.get_access<sycl::access::mode::read>(cgh);
				auto matrixDiagonal_acc = matrixDiagonal_buf.get_access<sycl::access::mode::read>(cgh);
				auto matrix_acc = matrix_buf.get_access<sycl::access::mode::read>(cgh);
				auto mtxIndL_acc = mtxIndL_buf.get_access<sycl::access::mode::read>(cgh);
				auto permutation_acc = permutation_buf.get_access<sycl::access::mode::read>(cgh);
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
				cgh.parallel_for<class symgs>(
						sycl::nd_range<1>(items, 8),
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
			});
			queue.wait();
		}
	}
//	BufferFactory sth;

//	bufferFactory.GetBuffer(A.mtxIndL,sycl::range<1>(A.localNumberOfRows));
	auto access = xv_buf.get_access<sycl::access::mode::read>();
//
//	std::cout << access[0]<<" | "<< matrixDiagonal[0]<<" | "<<A.matrixDiagonal[0][0]<<" | "<<A.matrixValues[0][0] << std::endl;
//	if(nrow==1124864)
//		std::cout << A.matrixValues[0][0] << std::endl;


	return 0;
}

