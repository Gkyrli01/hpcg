
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include <vector>
#include <cassert>
#include "Geometry.hpp"
#include "Vector.hpp"
#include "MGData.hpp"

#if __cplusplus < 201103L
// for C++03
#include <map>

typedef std::map<global_int_t, local_int_t> GlobalToLocalMap;
#else
// for C++11 or greater
#include <unordered_map>
using GlobalToLocalMap = std::unordered_map< global_int_t, local_int_t >;
#endif

struct SparseMatrix_STRUCT {

	SparseMatrix_STRUCT() : matrixValuesB(sycl::buffer<double ,2>(sycl::range<2>(1,1))), matrixDiagonalSYMGS(NULL),
							mtxIndLB(sycl::buffer<local_int_t ,2>(sycl::range<2>(1,1))), nonzerosInRow(NULL),
							matrixValuesBT(sycl::buffer<double ,2>(sycl::range<2>(1,1))),
							mtxIndLBT(sycl::buffer<local_int_t ,2>(sycl::range<2>(1,1))){
	}
	char *title; //!< name of the sparse matrix
	Geometry *geom; //!< geometry associated with this matrix
	global_int_t totalNumberOfRows; //!< total number of matrix rows across all processes
	global_int_t totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes
	local_int_t localNumberOfRows; //!< number of rows local to this process
	local_int_t localNumberOfColumns;  //!< number of columns local to this process
	local_int_t localNumberOfNonzeros;  //!< number of nonzeros local to this process

	sycl::buffer<char, 1> nonzerosInRow;
//	sycl::buffer<global_int_t,2>*mtxIndG;
	sycl::buffer<double, 1> matrixDiagonalSYMGS;
	sycl::buffer<double, 2> matrixValuesB;
	sycl::buffer<local_int_t, 2> mtxIndLB;

	sycl::buffer<double, 2> matrixValuesBT;
	sycl::buffer<local_int_t, 2> mtxIndLBT;


//	char *nonzerosInRow;  //!< The number of nonzeros in a row will always be 27 or fewer
	global_int_t **mtxIndG; //!< matrix indices as global values
	local_int_t **mtxIndL; //!< matrix indices as local values
	double **matrixValues; //!< values of matrix entries
//	double *matrixDiagonalSYMGS; //!< values of matrix diagonal entries
	double **matrixDiagonal; //!< values of matrix diagonal entries

	GlobalToLocalMap globalToLocalMap; //!< global-to-local mapping
	std::vector<global_int_t> localToGlobalMap; //!< local-to-global mapping
	mutable bool isDotProductOptimized;
	mutable bool isSpmvOptimized;
	mutable bool isMgOptimized;
	mutable bool isWaxpbyOptimized;
	/*!
	 This is for storing optimized data structres created in OptimizeProblem and
	 used inside optimized ComputeSPMV().
	 */
	mutable struct SparseMatrix_STRUCT *Ac; // Coarse grid matrix
	mutable MGData *mgData; // Pointer to the coarse level data for this fine matrix
	void *optimizationData;  // pointer that can be used to store implementation-specific data
	void *numberOfColors;
	int allColors;
#ifndef HPCG_NO_MPI
	local_int_t numberOfExternalValues; //!< number of entries that are external to this process
	int numberOfSendNeighbors; //!< number of neighboring processes that will be send local data
	local_int_t totalToBeSent; //!< total number of entries to be sent
	local_int_t *elementsToSend; //!< elements to send to neighboring processes
	int *neighbors; //!< neighboring processes
	local_int_t *receiveLength; //!< lenghts of messages received from neighboring processes
	local_int_t *sendLength; //!< lenghts of messages sent to neighboring processes
	double *sendBuffer; //!< send buffer for non-blocking sends
#endif
};
typedef struct SparseMatrix_STRUCT SparseMatrix;

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
inline void InitializeSparseMatrix(SparseMatrix &A, Geometry *geom) {
	A.title = 0;
	A.geom = geom;
	A.totalNumberOfRows = 0;
	A.totalNumberOfNonzeros = 0;
	A.localNumberOfRows = 0;
	A.localNumberOfColumns = 0;
	A.localNumberOfNonzeros = 0;
	A.mtxIndG = 0;
	A.mtxIndL = 0;
	A.matrixValues = 0;
	A.matrixDiagonal = 0;

	// Optimization is ON by default. The code that switches it OFF is in the
	// functions that are meant to be optimized.
	A.isDotProductOptimized = true;
	A.isSpmvOptimized = true;
	A.isMgOptimized = true;
	A.isWaxpbyOptimized = true;

#ifndef HPCG_NO_MPI
	A.numberOfExternalValues = 0;
	A.numberOfSendNeighbors = 0;
	A.totalToBeSent = 0;
	A.elementsToSend = 0;
	A.neighbors = 0;
	A.receiveLength = 0;
	A.sendLength = 0;
	A.sendBuffer = 0;
#endif
	A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.
	A.Ac = 0;
	return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
inline void CopyMatrixDiagonal(SparseMatrix &A, Vector &diagonal) {
	double **curDiagA = A.matrixDiagonal;
	auto access = diagonal.buf.get_access<sycl::access::mode::write>();
	assert(A.localNumberOfRows == diagonal.localLength);
	for (local_int_t i = 0; i < A.localNumberOfRows; ++i) access[i] = *(curDiagA[i]);
	return;
}

/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
inline void ReplaceMatrixDiagonal(SparseMatrix &A, Vector &diagonal) {

	auto tmp=(A.matrixValuesB).get_access<sycl::access::mode::read_write>();

	double **curDiagA = A.matrixDiagonal;
	auto accessDiag = diagonal.buf.get_access<sycl::access::mode::read>();


	assert(A.localNumberOfRows == diagonal.localLength);
	auto access = (A.matrixDiagonalSYMGS).get_access<sycl::access::mode::read_write>();
	for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
		*(curDiagA[i]) = accessDiag[i];
		access[i] = accessDiag[i];
	}

	std::cout<<tmp[0][0]<<" | "<<curDiagA[0][0]<<" | "<<access[0]<<std::endl;
	return;
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteMatrix(SparseMatrix &A) {

//	delete[] A.matrixValues;
//	delete[] A.mtxIndG;
//	delete[] A.mtxIndL;
#ifndef HPCG_CONTIGUOUS_ARRAYS
//	for (local_int_t i = 0; i < A.localNumberOfRows; ++i) {
//
//	}
#else
	delete [] A.matrixValues[0];
	delete [] A.mtxIndG[0];
	delete [] A.mtxIndL[0];
#endif
	if (A.title) delete[] A.title;
//	if (A.nonzerosInRow) A.nonzerosInRow=NULL;
	if (A.mtxIndG) delete[] A.mtxIndG;
	if (A.mtxIndL) delete[] A.mtxIndL;
	if (A.matrixValues) delete[] A.matrixValues;
	if (A.matrixDiagonal) delete[] A.matrixDiagonal;

#ifndef HPCG_NO_MPI
	if (A.elementsToSend) delete[] A.elementsToSend;
	if (A.neighbors) delete[] A.neighbors;
	if (A.receiveLength) delete[] A.receiveLength;
	if (A.sendLength) delete[] A.sendLength;
	if (A.sendBuffer) delete[] A.sendBuffer;
#endif

	if (A.geom != 0) {
		DeleteGeometry(*A.geom);
		delete A.geom;
		A.geom = 0;
	}
	if (A.Ac != 0) {
		DeleteMatrix(*A.Ac);
		delete A.Ac;
		A.Ac = 0;
	} // Delete coarse matrix
	if (A.mgData != 0) {
		DeleteMGData(*A.mgData);
		delete A.mgData;
		A.mgData = 0;
	} // Delete MG data
	return;
}

#endif // SPARSEMATRIX_HPP
