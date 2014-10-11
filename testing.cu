#include <algorithm>
#include <fstream>
#include <cmath>
#include <string>

extern "C" {
#include "mm_io/mm_io.h"
}

#include <sloan/common.h>
#include <sloan/timer.h>
#include <sloan/sloan.h>

#include <thrust/host_vector.h>

typedef typename thrust::host_vector<int>     IntVector;
typedef typename thrust::host_vector<double>  DoubleVector;

enum TestColor {
	COLOR_NO = 0,
	COLOR_RED,
	COLOR_GREEN
};

// -----------------------------------------------------------------------------

class OutputItem
{
	public:
		OutputItem(std::ostream &o) : m_o(o), m_additional_item_count(19) {}

		int  m_additional_item_count;

		template <typename T>
		void operator() (T item, TestColor c = COLOR_NO)
		{
			m_o << "<td style=\"border-style: inset;\">\n";
			switch (c)
			{
				case COLOR_RED:
					m_o << "<p> <FONT COLOR=\"Red\">" << item << " </FONT> </p>\n";
					break;
				case COLOR_GREEN:
					m_o << "<p> <FONT COLOR=\"Green\">" << item << " </FONT> </p>\n";
					break;
				default:
					m_o << "<p> " << item << " </p>\n";
					break;
			}
			m_o << "</td>\n";
		}

	private:
		std::ostream &m_o;
};

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "Usage:\n  driver input_file" << std::endl;
    return 1;
  } 

  // Read matrix from file (COO format)
  MM_typecode matcode;
  int M, N, nnz;
  int* row_i;
  int* col_j;
  double* vals;

  int err = mm_read_mtx_crd(argv[1], &M, &N, &nnz, &row_i, &col_j, &vals, &matcode);
  if (err != 0) {
    std::cerr << "error: " << err << std::endl;
    return 1;
  }

  // Switch to 0-based indices
  for (int i = 0; i < nnz; i++) {
    row_i[i]--;
    col_j[i]--;
  }
  // Convert to CSR format and load into thrust vectors.
  IntVector    row_offsets(N + 1);
  IntVector    column_indices(nnz);
  DoubleVector values(nnz);

  sloan::coo2csr(M, N, nnz, row_i, col_j, vals, row_offsets, column_indices, values);

  // Run the RCM algorithm
  sloan::Sloan algo(row_offsets, column_indices, values);
  OutputItem outputItem(std::cout);

  std::cout << "<tr valing=top>" << std::endl;
  {
	  std::string fileMat(argv[1]);
	  int i;
	  for (i = fileMat.size()-1; i>=0 && fileMat[i] != '/' && fileMat[i] != '\\'; i--);
	  i++;
	  fileMat = fileMat.substr(i);

	  size_t j = fileMat.rfind(".mtx");
	  if (j != std::string::npos)
		  outputItem( fileMat.substr(0, j));
	  else
		  outputItem( fileMat);
  }
  outputItem(N);
  outputItem(nnz);

  sloan::CPUTimer cpu_timer;

  cpu_timer.Start();
  algo.execute();
  cpu_timer.Stop();

  outputItem(algo.getOriginalRMSWf());
  outputItem(algo.getRMSWf());
  outputItem(cpu_timer.getElapsed());
  std::cout << "</tr>" << std::endl;

  return 0;
}
