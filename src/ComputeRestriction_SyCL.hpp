
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

#ifndef COMPUTERESTRICTION_SyCL_HPP
#define COMPUTERESTRICTION_SyCL_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
int ComputeRestriction_SyCL(const SparseMatrix & A, const Vector & rf);
#endif // COMPUTERESTRICTION_REF_HPP
