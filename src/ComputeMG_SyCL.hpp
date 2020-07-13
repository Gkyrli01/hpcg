
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

#ifndef COMPUTEMG_SyCL_HPP
#define COMPUTEMG_SyCL_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

int ComputeMG_SyCL( SparseMatrix  & A, const Vector & r, Vector & x);

#endif // COMPUTEMG_REF_HPP
