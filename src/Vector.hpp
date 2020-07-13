
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
 @file Vector.hpp

 HPCG data structures for dense vectors
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cassert>
#include "SyCLResources.h"
#include <cstdlib>
#include "Geometry.hpp"

struct Vector_STRUCT {
	Vector_STRUCT() : buf(NULL) {

	}

	local_int_t paddedLength;
	local_int_t localLength;  //!< length of local portion of the vector
	double *values;          //!< array of values
	sycl::buffer<double, 1> buf;


	/*!
	 This is for storing optimized data structures created in OptimizeProblem and
	 used inside optimized ComputeSPMV().
	 */
	void *optimizationData;

};
typedef struct Vector_STRUCT Vector;

/*!
  Initializes input vector.

  @param[in] v
  @param[in] localLength Length of local portion of input vector
 */
inline void InitializeVector(Vector &v, local_int_t localLength, local_int_t paddedLength) {
	v.paddedLength = paddedLength;
	v.localLength = localLength;
	std::cout << v.paddedLength << " | " << v.localLength << std::endl;
	v.values = new double[paddedLength];
	v.buf = sycl::buffer<double, 1>(v.values, sycl::range<1>(paddedLength));
	v.optimizationData = 0;
	return;
}

/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
inline void ZeroVector(Vector &v) {
	local_int_t localLength = v.paddedLength;
	auto access = v.buf.get_access<sycl::access::mode::write>();
	for (int i = 0; i < localLength; ++i) access[i] = 0.0;
	return;
}

/*!
  Multiply (scale) a specific vector entry by a given value.

  @param[inout] v Vector to be modified
  @param[in] index Local index of entry to scale
  @param[in] value Value to scale by
 */
inline void ScaleVectorValue(Vector &v, local_int_t index, double value) {
	assert(index >= 0 && index < v.localLength);
	auto access = v.buf.get_access<sycl::access::mode::read_write>();
	access[index] *= value;
	return;
}

/*!
  Fill the input vector with pseudo-random values.

  @param[in] v
 */
inline void FillRandomVector(Vector &v) {
	local_int_t localLength = v.localLength;
	auto access = v.buf.get_access<sycl::access::mode::write>();
	for (int i = 0; i < localLength; ++i) access[i] = rand() / (double) (RAND_MAX) + 1.0;

	for (int i = localLength; i < v.paddedLength; ++i) {
		access[i] = 0;
	}
	return;
}

/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
inline void CopyVector(Vector &v, Vector &w) {
	local_int_t localLength = v.localLength;
	assert(w.localLength >= localLength);
	auto access = v.buf.get_access<sycl::access::mode::read>();
	auto accessw = w.buf.get_access<sycl::access::mode::write>();
	for (int i = 0; i < localLength; ++i) accessw[i] = access[i];
	return;
}

inline void SyCLCopyVector(Vector_STRUCT &x, Vector_STRUCT &to) {
	local_int_t size = x.localLength;
	auto x_buf = x.buf;
	auto to_buf = to.buf;
	{
		queue.submit([&](sycl::handler &cgh) {
			auto to_acc = to_buf.get_access<sycl::access::mode::write>(cgh);
			auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
			cgh.parallel_for<class copy>(
					sycl::nd_range<1>(size, 32),
					[=](sycl::nd_item<1> item) {
						size_t i = item.get_global_linear_id();
						if (i < size)
							to_acc[i] = x_acc[i];
					});
		});
	}
//	auto access = to_buf.get_access<sycl::access::mode::read>();
}

inline void SyCLZeroVector(Vector_STRUCT &x) {
	local_int_t size = x.paddedLength;
	auto x_buf = x.buf;
	{

		queue.submit([&](sycl::handler &cgh) {
			auto x_acc = x_buf.get_access<sycl::access::mode::write>(cgh);
			cgh.parallel_for<class zero>(
					sycl::nd_range<1>(size, 32),
					[=](sycl::nd_item<1> item) {
						size_t i = item.get_global_linear_id();
						if (i < size)
							x_acc[i] = 0;
					});
		});
	}
//	x_buf.get_access<sycl::access::mode::read>();
}

/*!
  Deallocates the members of the data structure of the known system matrix provided they are not 0.

  @param[in] A the known system matrix
 */
inline void DeleteVector(Vector &v) {
	auto access = v.buf.get_access<sycl::access::mode::read>();
	delete[] v.values;
	v.localLength = 0;
	v.paddedLength = 0;
	return;
}

#endif // VECTOR_HPP
