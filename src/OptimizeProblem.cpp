
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include <iostream>
#include <random>
#include <zconf.h>
#include "OptimizeProblem.hpp"


void
SetColorPermutation(int nrow, const int *colors, const int *permutation, const int *permutationOpposite, int newIndex,
					const int *numberOfColors, int m);

int HpcgColoring(local_int_t **graph, int rows, char *nonzeros, int *colors, int val) {
	int maxVertexColor = -1;
	int maxcolor = -1;
	int idx=0;

	for (int i = 0; i < rows; ++i) {
		std::vector<local_int_t> neighbourColors;
		maxcolor = -1;
		for (int j = 0; j < nonzeros[i]; ++j) {
			local_int_t neighbour = graph[i][j];
			if (neighbour != i) {
//				if(neighbour==(i-1) ||neighbour==(i+1)||neighbour==(i+val)||neighbour==(i-val)||neighbour==(i+val*val)||neighbour==(i-val*val) ) {
//					if (colors[neighbour] > maxcolor && neighbour < i)
//						maxcolor = colors[neighbour];

				neighbourColors.push_back(colors[neighbour]);
//				}
			}
		}
//		if (i + 2 < rows)
//			neighbourColors.push_back(colors[i + 2]);
//		if (i - 2 >= 0)
//			neighbourColors.push_back(colors[i - 2]);

		//		if(i==20000){
//			std::cout<<neighbourColors.size()<<"\n";
//			for (int j = 0; j <  nonzeros[i]; ++j) {
//				std::cout<<"Neighbour: "<<graph[0][j]<<"\n";
//			}
//		}
//		std::cout<<neighbourColors.size()<<"\n";
//		std::reverse(neighbourColors.begin(),neighbourColors.end());
//		std::vector<short> cols({1,2,3,4,5,6,7,8});
//		std::shuffle(cols.begin(),cols.end(), std::mt19937(std::random_device()()));
//		std::cout<<i<<" \n";
		bool VertexColored = false;
		int VertexColor= 1; // We init all vertices to color=1
		idx=((idx+3)%8);

		if (maxcolor != -1)
			VertexColor = maxcolor;

		while (VertexColored != true) {
			bool IsNeighborColor = false;
			// Check if the color we're attempting to assign
			// is available
			std::vector<local_int_t>::iterator it;
			for (it = neighbourColors.begin(); it != neighbourColors.end(); it++) {
				if (*it == VertexColor) {
//					std::cout<<VertexColor<<"is the dix \n";

					IsNeighborColor = true;
					break;
				}
			}

			// If the color we're attempting is not already
			// assigned to one of the neighbors...
			if (IsNeighborColor == false) {
				// This is a valid color to assign
				colors[i] = VertexColor;
				VertexColored = true;
				break;
			} else {
				// Try with the next color
				VertexColor++;
//				std::cout<<idx<<"is the dix \n";

				idx=((idx+3)%16);
			}
		}

		if (maxVertexColor < VertexColor) {
			maxVertexColor = VertexColor;
		}
	}
	return maxVertexColor;
}


void
SetColorPermutation(int nrow, const int *colors, int *&permutation, int *&permutationOpposite, int &newIndex,
					int *&numberOfColors, int m) {
	for (int l = 0; l < nrow; ++l) {
		if (colors[l] == m) {
			permutation[newIndex] = l;
			permutationOpposite[l] = newIndex;
			newIndex++;
			numberOfColors[m - 1]++;
		}
	}
}

/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/

int level = 0;

int OptimizeProblem(SparseMatrix &A, CGData &data, Vector &b, Vector &x, Vector &xexact) {

	A.matrixValuesBT = sycl::buffer<double, 2>(sycl::range<2>(27, A.localNumberOfRows));
	A.mtxIndLBT = sycl::buffer<local_int_t, 2>(sycl::range<2>(27, A.localNumberOfRows));
	{
		auto matr2 = A.matrixValuesBT.get_access<sycl::access::mode::write>();
		auto mtx2 = A.mtxIndLBT.get_access<sycl::access::mode::write>();

		for (int l1 = 0; l1 < A.localNumberOfRows; ++l1) {
			for (int i = 0; i < 27; ++i) {
				matr2[i][l1] = A.matrixValues[l1][i];
				mtx2[i][l1] = A.mtxIndL[l1][i];
			}
		}
	}

	std::cout << "Optimizing started" << std::endl;
	int nrow = A.localNumberOfRows;

	auto *colors = new int[nrow];
	for (int k = 0; k < nrow; ++k) {
		colors[k] = -1;
	}
	auto *permutation = new int[nrow];
	auto *permutationOpposite = new int[nrow];

	for (int k = 0; k < nrow; ++k) {
		permutation[k] = -1;
		permutationOpposite[k] = -1;
	}
	std::cout << "Optimizing 1" << std::endl;

	int allcolors = -1;
	int prevColors = 0;
	auto access = A.nonzerosInRow.get_access<sycl::access::mode::read>();
	char *nonzeros = access.get_pointer();
	while (allcolors != prevColors) {
		prevColors = allcolors;
//		std::cout<<A.geom->nx<<std::endl;

		allcolors = HpcgColoring(A.mtxIndL, nrow, nonzeros, colors, A.geom->nx);
		std::cout << allcolors << std::endl;
	}
	std::cout << "Optimizing 2" << std::endl;

	//Create permutationMatrix

	int newIndex = 0;
	int *numberOfColors = new int[allcolors];
	for (int n = 0; n < allcolors; ++n) {
		numberOfColors[n] = 0;
	}
	std::cout << "Optimizing 3" << std::endl;

	for (int m = 1; m <=allcolors; ++m) {
		SetColorPermutation(nrow, colors, permutation, permutationOpposite, newIndex, numberOfColors, m);
	}

	std::cout << "Optimizing 4" << std::endl;

	A.optimizationData = permutation;
	A.optimizationDataOpposite = permutationOpposite;

	std::cout << "Optimizing 5" << std::endl;

	A.numberOfColors = numberOfColors;
	std::cout << "Optimizing 6" << std::endl;

	A.allColors = allcolors;
	delete[] colors;
	std::cout << "Optimizing 7" << std::endl;

	std::cout << "Optimizing finished" << std::endl;

	if (A.Ac) {
		OptimizeProblem(*A.Ac, data, b, x, xexact);
	}

#if defined(HPCG_USE_MULTICOLORING)
	const local_int_t nrow = A.localNumberOfRows;
	std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
	int totalColors = 1;
	colors[0] = 0; // first point gets color 0

	// Finds colors in a greedy (a likely non-optimal) fashion.

	for (local_int_t i=1; i < nrow; ++i) {
	  if (colors[i] == nrow) { // if color not assigned
		std::vector<int> assigned(totalColors, 0);
		int currentlyAssigned = 0;
		const local_int_t * const currentColIndices = A.mtxIndL[i];
		const int currentNumberOfNonzeros = A.nonzerosInRow[i];

		for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
		  local_int_t curCol = currentColIndices[j];
		  if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
			if (assigned[colors[curCol]] == 0)
			  currentlyAssigned += 1;
			assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
		  } // else // could take advantage of indices being sorted
		}

		if (currentlyAssigned < totalColors) { // if there is at least one color left to use
		  for (int j=0; j < totalColors; ++j)  // try all current colors
			if (assigned[j] == 0) { // if no neighbor with this color
			  colors[i] = j;
			  break;
			}
		} else {
		  if (colors[i] == nrow) {
			colors[i] = totalColors;
			totalColors += 1;
		  }
		}
	  }
	}

	std::vector<local_int_t> counters(totalColors);
	for (local_int_t i=0; i<nrow; ++i)
	  counters[colors[i]]++;

	// form in-place prefix scan
	local_int_t old=counters[0], old0;
	for (local_int_t i=1; i < totalColors; ++i) {
	  old0 = counters[i];
	  counters[i] = counters[i-1] + old;
	  old = old0;
	}
	counters[0] = 0;

	// translate `colors' into a permutation
	for (local_int_t i=0; i<nrow; ++i) // for each color `c'
	  colors[i] = counters[colors[i]]++;
#endif

	return 0;
}


//		{
//			auto non = A.nonzerosInRow.get_access<sycl::access::mode::read>();
//
//			for (int i = 0; i < 30; ++i) {
//				std::cout << (int)non[i] << "\n";
//			}
//		}
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int ReorderAll(SparseMatrix &A, CGData &data, Vector &b, Vector &x, Vector &xexact, int firstLevel) {
//	auto *permutation = static_cast<local_int_t *>(A.optimizationData);
	auto *permutation = static_cast<local_int_t *>(A.optimizationDataOpposite);

	local_int_t *permutationnext;
	if (A.Ac) {
		permutationnext = static_cast<local_int_t *>(A.Ac->optimizationData);
	}
//	usleep(2000000);

	PermuteMatrix(A.matrixValuesBT, permutation, A.localNumberOfRows, A.nonzerosInRow, true);
	{
		auto a = A.matrixValuesBT.get_access<sycl::access::mode::read>();
	}
	PermuteMatrix(A.matrixValuesB, permutation, A.localNumberOfRows, A.nonzerosInRow, false);
	{
		auto a = A.matrixValuesB.get_access<sycl::access::mode::read>();
	}
//	usleep(1000000);

	PermuteMatrixAndContents(A.mtxIndLBT, permutation, A.localNumberOfRows, A.nonzerosInRow, A.matrixValuesBT,true);
	{
		auto a = A.mtxIndLBT.get_access<sycl::access::mode::read>();
		auto c = A.nonzerosInRow.get_access<sycl::access::mode::read>();
		for (int i = 0; i < 27; ++i) {
			std::cout << a[i][1000] << " ";

		}
		std::cout << "\n";
		for (int i = 0; i < 27; ++i) {
			std::cout << a[i][1001] << " ";

		}
	}
	PermuteMatrixAndContents(A.mtxIndLB, permutation, A.localNumberOfRows, A.nonzerosInRow, A.matrixValuesB, false);


	PermuteVector(A.nonzerosInRow, permutation, A.localNumberOfRows);

	PermuteVector(A.matrixDiagonalSYMGS, permutation, A.localNumberOfRows);

	if (firstLevel) {
		PermuteVector(x.buf, permutation, x.localLength);
		PermuteVector(xexact.buf, permutation, xexact.localLength);
		PermuteVector(b.buf, permutation, b.localLength);
		PermuteVector(data.Ap.buf, permutation, data.Ap.localLength);
		PermuteVector(data.p.buf, permutation, data.p.localLength);
		PermuteVector(data.r.buf, permutation, data.r.localLength);
		PermuteVector(data.z.buf, permutation, data.z.localLength);
	}
//
	if (A.mgData) {
		{
			auto vals = A.mgData->f2cOperator.get_access<sycl::access::mode::read>();
			for (int i = 0; i < 10; ++i) {
				std::cout << permutation[vals[permutationnext[i]]] << "  f2c |Permutation: " << permutationnext[i]
						  << "|New value" << permutation[vals[i]] << "|Another value" <<
						  permutation[vals[i]] << "\n";
			}
		}
		PermuteFine2Coarse(A.mgData->f2cOperator, permutation, permutationnext, A.localNumberOfRows,
						   A.mgData->rc->localLength);
		{
			auto vals = A.mgData->f2cOperator.get_access<sycl::access::mode::read>();
			for (int i = 0; i < 10; ++i) {
				std::cout << vals[permutationnext[i]] << "  f2c |Permutation: " << vals[i] << "|New value"
						  << permutation[vals[i]] << "|Another value" <<
						  vals[i] << "\n";
			}
		}

		PermuteVector(A.mgData->Axf->buf, permutation, A.mgData->Axf->localLength);
		PermuteVector(A.mgData->rc->buf, permutationnext, A.mgData->rc->localLength);
		PermuteVector(A.mgData->xc->buf, permutationnext, A.mgData->xc->localLength);
	}
	if (A.Ac)
		ReorderAll(*A.Ac, data, b, x, xexact, 0);
	std::cout << "OutOf level\n";

	return 0;
}


// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix &A) {

	return 0.0;

}
