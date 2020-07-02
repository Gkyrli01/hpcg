
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
#include "OptimizeProblem.hpp"






int HpcgColoring(local_int_t **graph, int rows, char *nonzeros, int *colors) {
	int maxVertexColor = -1;
	int maxcolor = -1;
	for (int i = 0; i < rows; ++i) {
		std::vector<local_int_t > neighbourColors;
		maxcolor = -1;
		for (int j = 0; j < nonzeros[i]; ++j) {
			local_int_t neighbour = graph[i][j];
			if (neighbour != i) {
//				if (colors[neighbour] > maxcolor && neighbour < i)
//					maxcolor = colors[neighbour];
				neighbourColors.push_back(colors[neighbour]);
			}
		}
		bool VertexColored = false;
		int VertexColor = 1; // We init all vertices to color=1

		if (maxcolor != -1)
			VertexColor = maxcolor;

		while (VertexColored != true) {
			bool IsNeighborColor = false;
			// Check if the color we're attempting to assign
			// is available
			std::vector<local_int_t >::iterator it;
			for (it = neighbourColors.begin(); it != neighbourColors.end(); it++) {
				if (*it == VertexColor) {
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
			}
		}

		if (maxVertexColor < VertexColor) {
			maxVertexColor = VertexColor;
		}
	}
	return maxVertexColor;
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
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {


	std::cout<<"Optimizing started"<<std::endl;
  	int nrow=A.localNumberOfRows;

	auto *colors = new int[nrow];
	for (int k = 0; k < nrow; ++k) {
		colors[k] = -1;
	}
	auto *permutation = new int[nrow];
	for (int k = 0; k < nrow; ++k) {
		permutation[k] = -1;
	}
	std::cout<<"Optimizing 1"<<std::endl;

	int allcolors = -1;
	int prevColors = 0;
	auto access=A.nonzerosInRow->get_access<sycl::access::mode::read>();
	char * nonzeros=access.get_pointer();
	while (allcolors != prevColors) {
		prevColors = allcolors;
		allcolors = HpcgColoring(A.mtxIndL, nrow, nonzeros, colors);
//		std::cout<<allcolors<<std::endl;
	}
	std::cout<<"Optimizing 2"<<std::endl;

	//Create permutationMatrix

	int newIndex = 0;
	int *numberOfColors = new int[allcolors];
	for (int n = 0; n < allcolors; ++n) {
		numberOfColors[n] = 0;
	}
	std::cout<<"Optimizing 3"<<std::endl;

	for (int m = 1; m <= allcolors; ++m) {
		for (int l = 0; l < nrow; ++l) {
			if (colors[l] == m) {
				permutation[newIndex] = l;
				newIndex++;
				numberOfColors[m - 1]++;
			}
		}
	}
	std::cout<<"Optimizing 4"<<std::endl;

	A.optimizationData=permutation;
	std::cout<<"Optimizing 5"<<std::endl;

	A.numberOfColors=numberOfColors;
	std::cout<<"Optimizing 6"<<std::endl;

	A.allColors=allcolors;
	delete[] colors;
	std::cout<<"Optimizing 7"<<std::endl;

	std::cout<<"Optimizing finished"<<std::endl;

	if(A.Ac){
		OptimizeProblem(*A.Ac, data,  b, x, xexact);
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

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

  return 0.0;

}
