
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

#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"

int OptimizeProblem(SparseMatrix &A, CGData &data, Vector &b, Vector &x, Vector &xexact);

// This helper function should be implemented in a non-trivial way if OptimizeProblem is non-trivial
// It should return as type double, the total number of bytes allocated and retained after calling OptimizeProblem.
// This value will be used to report Gbytes used in ReportResults (the value returned will be divided by 1000000000.0).

double OptimizeProblemMemoryUse(const SparseMatrix &A);

template<typename T>
inline void PermuteVector(sycl::buffer<T, 1> &buf, local_int_t *permutation, int len) {
	auto nbuf = sycl::buffer<T, 1>(sycl::range<1>(len));
	{
		std::cout << len << "\n";
		auto permutation_buf = *(bufferFactory.GetBuffer(permutation, sycl::range<1>(len)));
		std::cout << permutation_buf.get_count() << " is buffer size \n";

		queue.submit([&](sycl::handler &cgh) {
			auto bufAcc = buf.template get_access<sycl::access::mode::read>(cgh);
			auto reordered = nbuf.template get_access<sycl::access::mode::write>(cgh);
			auto permutation_acc = permutation_buf.get_access<sycl::access::mode::read>(cgh);
			cgh.parallel_for<sycl::buffer<T, 1>>(
					sycl::nd_range<1>(len, 32),
					[=](sycl::nd_item<1> item) {
						size_t z = item.get_global_linear_id();
						if (z < len) {
							auto i = permutation_acc[z];
							reordered[i] = bufAcc[z];
						}
					});
		});
	}
//	nbuf.template get_access<sycl::access::mode::read>();
	buf = nbuf;
}


inline void
PermuteFine2Coarse(sycl::buffer<local_int_t, 1> &buf, local_int_t *permutation, local_int_t *permutation2, int len,
				   int len2) {
	auto nbuf = sycl::buffer<local_int_t, 1>(sycl::range<1>(len));
	{
		std::cout << len << " of fine \n";
		std::cout << len2 << " of coarse \n";
		auto permutation_buf = *(bufferFactory.GetBuffer(permutation, sycl::range<1>(len)));
		auto permutation2_buf = *(bufferFactory.GetBuffer(permutation2, sycl::range<1>(len2)));
		queue.submit([&](sycl::handler &cgh) {
			auto bufAcc = buf.get_access<sycl::access::mode::read>(cgh);
			auto reordered = nbuf.get_access<sycl::access::mode::write>(cgh);
			auto permutation_acc = permutation_buf.get_access<sycl::access::mode::read>(cgh);
			auto permutationnext = permutation2_buf.get_access<sycl::access::mode::read>(cgh);
			cgh.parallel_for<class FineToCoarse>(
					sycl::nd_range<1>(len2, 32),
					[=](sycl::nd_item<1> item) {
						size_t z = item.get_global_linear_id();
						reordered[permutationnext[z]]=permutation_acc[bufAcc[z]];
//						reordered[z] = permutation_acc[bufAcc[permutationnext[z]]];
					});
		});
	}
	buf = nbuf;
}

//template<typename T>
//inline void ReversePermuteVector(sycl::buffer<T, 1> &buf, local_int_t *permutation, int len) {
//	auto nbuf = sycl::buffer<T, 1>(sycl::range<1>(len));
//
//	{
//		std::cout << len << "\n";
//		auto permutation_buf = *(bufferFactory.GetBuffer(permutation, sycl::range<1>(len)));
//		queue.submit([&](sycl::handler &cgh) {
//			auto bufAcc = buf.template get_access<sycl::access::mode::read>(cgh);
//			auto reordered = nbuf.template get_access<sycl::access::mode::read_write>(cgh);
//			auto permutation_acc = permutation_buf.get_access<sycl::access::mode::read>(cgh);
//			cgh.parallel_for<sycl::buffer<T, 3>>(
//					sycl::nd_range<1>(len, 32),
//					[=](sycl::nd_item<1> item) {
//						size_t z = item.get_global_linear_id();
//						if (z < len) {
//							auto i = permutation_acc[z];
//							reordered[z] = bufAcc[i];
//						}
//					});
//		});
//	}
//	buf = nbuf;
//}

template<typename T>
inline void PermuteMatrix(sycl::buffer<T, 2> &buf, local_int_t *permutation, int len,sycl::buffer<char , 1> &nonzeros) {
	auto nbuf = sycl::buffer<T, 2>(sycl::range<2>(27, len));
	{
		auto permutation_buf = *(bufferFactory.GetBuffer(permutation, sycl::range<1>(len)));
		queue.submit([&](sycl::handler &cgh) {
			auto bufAcc = buf.template get_access<sycl::access::mode::read>(cgh);
			auto reordered = nbuf.template get_access<sycl::access::mode::write>(cgh);
			auto permutation_acc = permutation_buf.get_access<sycl::access::mode::read>(cgh);
			auto nonzeros_acc = nonzeros.get_access<sycl::access::mode::read>(cgh);

			cgh.parallel_for<sycl::buffer<T, 2>>(
					sycl::nd_range<1>(len, 32),
					[=](sycl::nd_item<1> item) {
						size_t z = item.get_global_linear_id();
						if (z < len) {
							auto i = permutation_acc[z];
							for (int j = 0; j < 27; ++j) {
								if(j<nonzeros_acc[z])
									reordered[j][i] = bufAcc[j][z];
								else
									reordered[j][i]=0;
							}
						}
					});
		});
	}
	buf = nbuf;
}

template<typename T>
inline void PermuteMatrixAndContents(sycl::buffer<T, 2> &buf,local_int_t *permutation, int len,sycl::buffer<char , 1> &nonzeros) {
	auto nbuf = sycl::buffer<T, 2>(sycl::range<2>(27, len));
	{
		auto permutation_buf = *(bufferFactory.GetBuffer(permutation, sycl::range<1>(len)));

		queue.submit([&](sycl::handler &cgh) {
			auto bufAcc = buf.template get_access<sycl::access::mode::read>(cgh);
			auto reordered = nbuf.template get_access<sycl::access::mode::write>(cgh);
			auto permutation_acc = permutation_buf.get_access<sycl::access::mode::read>(cgh);
			auto nonzeros_acc = nonzeros.get_access<sycl::access::mode::read>(cgh);

			cgh.parallel_for<sycl::buffer<T, 2>>(
					sycl::nd_range<1>(len, 32),
					[=](sycl::nd_item<1> item) {
						size_t z = item.get_global_linear_id();
						if (z < len) {
							auto i = permutation_acc[z];
							for (int j = 0; j < 27; ++j) {
								if(j<nonzeros_acc[z])
									reordered[j][i] = permutation_acc[bufAcc[j][z]];
								else
									reordered[j][i] = 0;
							}
//							char nonzeroes=nonzeros_acc[z];
//							for (int k = 0; k < nonzeroes; ++k) {
//								for (int j = k+1; j < nonzeroes; ++j) {
//									if(reordered[k][i]>reordered[j][i]){
//										auto tmp=reordered[k][i];
//										reordered[k][i]=reordered[j][i];
//										reordered[j][i]=tmp;
//									}
//								}
//							}
						}
					});
		});
	}


	buf = nbuf;
}


int ReorderAll(SparseMatrix &A, CGData &data, Vector &b, Vector &x, Vector &xexact, int firstLevel);

#endif  // OPTIMIZEPROBLEM_HPP
