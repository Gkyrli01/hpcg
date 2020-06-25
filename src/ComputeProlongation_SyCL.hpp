
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

#ifndef COMPUTEPROLONGATION_SyCL_HPP
#define COMPUTEPROLONGATION_SyCL_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
int ComputeProlongation_SyCL(const SparseMatrix & Af, Vector & xf);
#endif // COMPUTEPROLONGATION_REF_HPP
