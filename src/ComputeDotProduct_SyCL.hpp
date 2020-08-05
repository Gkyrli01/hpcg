
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

#ifndef COMPUTEDOTPRODUCT_SyCL_HPP
#define COMPUTEDOTPRODUCT_SyCL_HPP
#include "Vector.hpp"
int ComputeDotProduct_SyCL(const local_int_t n,  Vector & x,  Vector & y,
    double & result, double & time_allreduce);

#endif // COMPUTEDOTPRODUCT_REF_HPP
