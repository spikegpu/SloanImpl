#ifndef SLOAN_H
#define SLOAN_H

#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/logical.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>

#include <queue>

#include <stdio.h>

extern "C" {
#include "mm_io/mm_io.h"
}

namespace sloan {

	struct empty_row_functor
	{
		typedef bool result_type;
		typedef typename thrust::tuple<int, int>       IntTuple;
			__host__ __device__
			bool operator()(const IntTuple& t) const
			{
				const int a = thrust::get<0>(t);
				const int b = thrust::get<1>(t);

				return a != b;
			}
	};

class Sloan_base
{
protected:
	size_t         m_n;
	size_t         m_nnz;

	double         m_ori_rmswf;
	double         m_rmswf;

	typedef typename thrust::tuple<int, int, int, int>       IntTuple;
	typedef typename thrust::tuple<int, int, int>            IntTriple;
	typedef typename thrust::tuple<int, int>                 IntPair;


	template <typename IVector>
	static void offsets_to_indices(const IVector& offsets, IVector& indices)
	{
		// convert compressed row offsets into uncompressed row indices
		thrust::fill(indices.begin(), indices.end(), 0);
		thrust::scatter_if( thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(offsets.size()-1),
				offsets.begin(),
                    	thrust::make_transform_iterator(
                                thrust::make_zip_iterator( thrust::make_tuple( offsets.begin(), offsets.begin()+1 ) ),
                                empty_row_functor()),
				indices.begin());
		thrust::inclusive_scan(indices.begin(), indices.end(), indices.begin(), thrust::maximum<int>());
	}

	template <typename IVector>
	static void indices_to_offsets(const IVector& indices, IVector& offsets)
	{
		// convert uncompressed row indices into compressed row offsets
		thrust::lower_bound(indices.begin(),
				indices.end(),
				thrust::counting_iterator<int>(0),
				thrust::counting_iterator<int>(offsets.size()),
				offsets.begin());
	}

	template <typename IVectorIterator>
	static void indices_to_offsets(IVectorIterator begin, IVectorIterator end, IVectorIterator output, int start_value, int end_value) 
	{
		thrust::lower_bound(begin,
				end,
				thrust::counting_iterator<int>(start_value),
				thrust::counting_iterator<int>(end_value + 1),
				output);
	}

public:

	virtual ~Sloan_base() {}

    virtual void execute() = 0;

	double  getOriginalRMSWf() const {return m_ori_rmswf;}
	double  getRMSWf()         const {return m_rmswf;}
};

class Sloan: public Sloan_base
{
private:
  typedef typename thrust::host_vector<int>       IntVectorH;
  typedef typename thrust::host_vector<double>    DoubleVectorH;
  typedef typename thrust::host_vector<bool>      BoolVectorH;

  typedef typename IntVectorH::iterator           IntIterator;
  typedef typename thrust::tuple<IntIterator, IntIterator>     IntIteratorTuple;
  typedef typename thrust::zip_iterator<IntIteratorTuple>      EdgeIterator;

  typedef typename thrust::tuple<int, int>        NodeType;

  typedef enum {
	  INACTIVE,
	  PREACTIVE,
	  ACTIVE,
	  POSTACTIVE
  } Status;

  typedef typename thrust::host_vector<Status>    StatusVectorH;

  IntVectorH     m_row_offsets;
  IntVectorH     m_column_indices;
  DoubleVectorH  m_values;

  IntVectorH     m_perm;

  void buildTopology(EdgeIterator&      begin,
                     EdgeIterator&      end,
				     int                node_begin,
				     int                node_end,
                     IntVectorH&        row_offsets,
                     IntVectorH&        column_indices);

  void unorderedBFS(IntVectorH&   tmp_reordering,
					IntVectorH&   row_offsets,
					IntVectorH&   column_indices,
					IntVectorH&   visited,
					IntVectorH&   levels,
					IntVectorH&   ori_degrees,
					BoolVectorH&  tried);

  void unorderedBFSIteration(int            width,
							 int            start_idx,
							 int            end_idx,
							 IntVectorH&    tmp_reordering,
							 IntVectorH&    levels,
							 IntVectorH&    visited,
							 IntVectorH&    row_offsets,
							 IntVectorH&    column_indices,
							 IntVectorH&    ori_degrees,
							 BoolVectorH&   tried,
							 IntVectorH&    costs,
							 IntVectorH&    ori_costs,
							 StatusVectorH& status,
							 int &          next_level);

public:
  Sloan(const IntVectorH&    row_offsets,
        const IntVectorH&    column_indices,
        const DoubleVectorH& values)
  : m_row_offsets(row_offsets),
    m_column_indices(column_indices),
    m_values(values)
  {
    size_t n = row_offsets.size() - 1;
    m_perm.resize(n);
    m_n               = n;
    m_nnz             = m_values.size();
	m_ori_rmswf       = 0.0;
	m_rmswf           = 0.0;
  }

  ~Sloan() {}

  void execute();
};

void
Sloan::execute()
{
	IntVectorH tmp_reordering(m_n);
	IntVectorH tmp_perm(m_n);

	thrust::sequence(tmp_reordering.begin(), tmp_reordering.end());

	IntVectorH row_indices(m_nnz);
	IntVectorH tmp_row_indices(m_nnz << 1);
	IntVectorH tmp_column_indices(m_nnz << 1);
	IntVectorH tmp_row_offsets(m_n + 1);
	offsets_to_indices(m_row_offsets, row_indices);

	EdgeIterator begin = thrust::make_zip_iterator(thrust::make_tuple(row_indices.begin(), m_column_indices.begin()));
	EdgeIterator end   = thrust::make_zip_iterator(thrust::make_tuple(row_indices.end(),   m_column_indices.end()));
	buildTopology(begin, end, 0, m_n, tmp_row_offsets, tmp_column_indices);

	IntVectorH  ori_degrees(m_n);
	BoolVectorH tried(m_n, false);
	IntVectorH  visited(m_n, -1);
	IntVectorH  levels(m_n);

	thrust::transform(tmp_row_offsets.begin() + 1, tmp_row_offsets.end(), tmp_row_offsets.begin(), ori_degrees.begin(), thrust::minus<int>());

	BoolVectorH counted(m_n, false);
	int tmp_wf = 0;

	{
		for (int i = 0; i < m_n; i++) {
			int start_idx = tmp_row_offsets[i], end_idx = tmp_row_offsets[i+1];

			if (!counted[i]) {
				tmp_wf ++;
				counted[i] = true;
			}

			for (int l = start_idx; l < end_idx; l++) {
				int column = tmp_column_indices[l];
				if (column <= i || counted[column]) continue;
				counted[column] = true;
				tmp_wf ++;
			}

			m_ori_rmswf += 1.0 * tmp_wf * tmp_wf;

			tmp_wf --;
		}
		m_ori_rmswf = sqrt(m_ori_rmswf / m_n);
	}

	unorderedBFS(tmp_reordering,
				 tmp_row_offsets,
				 tmp_column_indices,
				 visited,
				 levels,
				 ori_degrees,
				 tried);

	thrust::scatter(thrust::make_counting_iterator(0),
					thrust::make_counting_iterator(int(m_n)),
					tmp_reordering.begin(),
					tmp_perm.begin());

	{
		thrust::fill(counted.begin(), counted.end(), false);

		tmp_wf = 0;
		for (int i = 0; i < m_n; i++) {
			int row = tmp_reordering[i];
			int start_idx = tmp_row_offsets[row], end_idx = tmp_row_offsets[row+1];

			if (!counted[row]) {
				tmp_wf ++;
				counted[row] = true;
			}

			for (int l = start_idx; l < end_idx; l++) {
				int column  = tmp_column_indices[l];
				int new_pos = tmp_perm[column];
				if (new_pos <= i || counted[column]) continue;
				counted[column] = true;
				tmp_wf ++;
			}

			m_rmswf += 1.0 * tmp_wf * tmp_wf;

			tmp_wf --;
		}
		m_rmswf = sqrt(m_rmswf / m_n);
	}
}

void
Sloan::buildTopology(EdgeIterator&      begin,
                     EdgeIterator&      end,
				     int                node_begin,
				     int                node_end,
                     IntVectorH&        row_offsets,
                     IntVectorH&        column_indices)
{
	if (row_offsets.size() != m_n + 1)
		row_offsets.resize(m_n + 1, 0);
	else
		thrust::fill(row_offsets.begin(), row_offsets.end(), 0);

	IntVectorH row_indices((end - begin) << 1);
	column_indices.resize((end - begin) << 1);
	int actual_cnt = 0;

	for(EdgeIterator edgeIt = begin; edgeIt != end; edgeIt++) {
		int from = thrust::get<0>(*edgeIt), to = thrust::get<1>(*edgeIt);
		if (from != to) {
			row_indices[actual_cnt]        = from;
			column_indices[actual_cnt]     = to;
			row_indices[actual_cnt + 1]    = to;
			column_indices[actual_cnt + 1] = from;
			actual_cnt += 2;
		}
	}
	row_indices.resize(actual_cnt);
	column_indices.resize(actual_cnt);
	// thrust::sort_by_key(row_indices.begin(), row_indices.end(), column_indices.begin());
	{
		int&      nnz = actual_cnt;
		IntVectorH tmp_column_indices(nnz);
		for (int i = 0; i < nnz; i++)
			row_offsets[row_indices[i]] ++;

		thrust::inclusive_scan(row_offsets.begin() + node_begin, row_offsets.begin() + (node_end + 1), row_offsets.begin() + node_begin);

		for (int i = nnz - 1; i >= 0; i--) {
			int idx = (--row_offsets[row_indices[i]]);
			tmp_column_indices[idx] = column_indices[i];
		}
		column_indices = tmp_column_indices;
	}
}

void 
Sloan::unorderedBFS(IntVectorH&   tmp_reordering,
					IntVectorH&   row_offsets,
					IntVectorH&   column_indices,
					IntVectorH&   visited,
					IntVectorH&   levels,
					IntVectorH&   ori_degrees,
					BoolVectorH&  tried)
{
	int min_idx = thrust::min_element(ori_degrees.begin(), ori_degrees.end()) - ori_degrees.begin();

	visited[min_idx]   = 0;
	tried[min_idx]     = true;
	tmp_reordering[0]  = min_idx;

	int queue_begin = 0, queue_end = 1;

	IntVectorH comp_offsets(1, 0);

	int last = 0;
	int cur_comp = 0;

	int width = 0, max_width = 0;

	IntVectorH costs(m_n), ori_costs(m_n);

	StatusVectorH status(m_n, INACTIVE);

	for (int l = 0; l < m_n; l ++) {
		int n_queue_begin = queue_end;
		if (n_queue_begin - queue_begin > 0) {
			if (width < n_queue_begin - queue_begin)
				width = n_queue_begin - queue_begin;

			for (int l2 = queue_begin; l2 < n_queue_begin; l2 ++) {
				levels[l2] = l;
				int  row = tmp_reordering[l2];
				int  start_idx = row_offsets[row], end_idx = row_offsets[row + 1];

				for (int j = start_idx; j < end_idx; j++) {
					int column = column_indices[j];
					if (visited[column] != 0) {
						visited[column] = 0;
						tmp_reordering[queue_end ++] = column;
					}
				}
			}

			queue_begin = n_queue_begin;
		} else {
			comp_offsets.push_back(queue_begin);
			cur_comp ++;

			if (max_width < width) max_width = width;

			if (queue_begin - comp_offsets[cur_comp - 1] > 32) {
				unorderedBFSIteration(width,
						comp_offsets[cur_comp-1],
						comp_offsets[cur_comp],
						tmp_reordering,
						levels,
						visited,
						row_offsets,
						column_indices,
						ori_degrees,
						tried,
						costs,
						ori_costs,
						status,
						l);
			}
			width = 0;

			if (queue_begin >= m_n) break;

			for (int j = last; j < m_n; j++)
				if (visited[j] < 0) {
					visited[j] = 0;
					tmp_reordering[n_queue_begin] = j;
					last = j;
					tried[j] = true;
					queue_end ++;
					l --;
					break;
				}
		}
	}
}

void
Sloan::unorderedBFSIteration(int            width,
							 int            start_idx,
							 int            end_idx,
							 IntVectorH&    tmp_reordering,
							 IntVectorH&    levels,
							 IntVectorH&    visited,
							 IntVectorH&    row_offsets,
							 IntVectorH&    column_indices,
							 IntVectorH&    ori_degrees,
							 BoolVectorH&   tried,
							 IntVectorH&    costs,
							 IntVectorH&    ori_costs,
							 StatusVectorH& status,
							 int &          next_level)
{
	int S = tmp_reordering[start_idx], E = -1;
	int pS = S, pE;

	int next_level_bak = next_level;

	const int ITER_COUNT = 10;

	int p_max_level = levels[end_idx - 1];
	int max_level = p_max_level;
	int start_level = levels[start_idx];

	IntVectorH tmp_reordering_bak(end_idx - start_idx);

	for (int i = 1; i < ITER_COUNT; i++)
	{
		int max_level_start_idx = thrust::lower_bound(levels.begin() + start_idx, levels.begin() + end_idx, max_level) - levels.begin();

		int max_count = end_idx - max_level_start_idx;

		IntVectorH max_level_valence(max_count);
		if( max_count > 1 ) {

			thrust::gather(tmp_reordering.begin() + max_level_start_idx, tmp_reordering.begin() + end_idx, ori_degrees.begin(), max_level_valence.begin());

			thrust::sort_by_key(max_level_valence.begin(), max_level_valence.end(), tmp_reordering.begin() + max_level_start_idx);

			E = tmp_reordering[max_level_start_idx];
		}
		else
			E = tmp_reordering[end_idx - 1];

		if (tried[E]) {
			int j;
			for (j = max_level_start_idx; j < end_idx; j++)
				if (!tried[tmp_reordering[j]]) {
					E = tmp_reordering[j];
					break;
				}
			if (j >= end_idx) {
				E = pE;
				S = pS;
				break;
			}
		}
		pE = E;

		int queue_begin = start_idx;
		int queue_end   = start_idx + 1;

		tmp_reordering[start_idx] = E;
		tried[E] = true;
		visited[E] = i;
		levels[start_idx]  = start_level;

		int l;
		int tmp_width = 0;
		for (l = start_level; l < m_n; l ++) {
			int n_queue_begin = queue_end;
			if (tmp_width < n_queue_begin - queue_begin)
				tmp_width = n_queue_begin - queue_begin;

			if (n_queue_begin - queue_begin > 0)
			{
				for (int l2 = queue_begin; l2 < n_queue_begin; l2++) {
					levels[l2] = l;
					int row = tmp_reordering[l2];
					int start_idx = row_offsets[row], end_idx = row_offsets[row + 1];
					for (int j = start_idx; j < end_idx; j++) {
						int column = column_indices[j];
						if (visited[column] != i) {
							visited[column] = i;
							tmp_reordering[queue_end++] = column;
						}
					}
				}
				queue_begin = n_queue_begin;
			} else
				break;
		}

		if (tmp_width > width) {
			next_level = next_level_bak;
			break;
		}

		max_level = levels[end_idx - 1];
		if (max_level <= p_max_level) {
			next_level = max_level + 1;

			break;
		}

		width = tmp_width;


		p_max_level = max_level;
		next_level_bak = next_level = l;

		pS = S;
		S  = E;
	}

	const int W1 = 2, W2 = 1;

	for (int i = start_idx; i < end_idx; i++)
		costs[i] = (m_n - 1 - ori_degrees[tmp_reordering[i]]) * W1 + levels[i] * W2;

	thrust::scatter(costs.begin() + start_idx, costs.begin() + end_idx, tmp_reordering.begin() + start_idx, ori_costs.begin());

	//// int head = 0, tail = 1;
	tmp_reordering_bak[0] = S;
	status[S] = PREACTIVE;

	int cur_idx = start_idx;

	std::priority_queue<thrust::tuple<int, int > > pq;
	pq.push(thrust::make_tuple(ori_costs[S],S));

	//// while(head < tail) {
	while(! pq.empty()) {
		//// int cur_node = tmp_reordering_bak[head];
		thrust::tuple<int, int> tmp_tuple = pq.top();
		pq.pop();
		int cur_node = thrust::get<1>(tmp_tuple);
		//// int max_cost = ori_costs[cur_node];
		//// int idx = head;
		bool found = (status[cur_node] != POSTACTIVE);
		while (!found) {
			if (pq.empty()) break;
			tmp_tuple = pq.top();
			pq.pop();
			cur_node = thrust::get<1>(tmp_tuple);
			found = (status[cur_node] != POSTACTIVE);
		}

		if (!found) break;

		//// {
			////for (int i = head + 1; i < tail; i++)
	////			if (max_cost < ori_costs[tmp_reordering_bak[i]]) {
	////				idx = i;
	////				cur_node = tmp_reordering_bak[i];
	////				max_cost = ori_costs[cur_node + start_idx];
	////			}
	////	}

		//// if (idx != head) 
			//// tmp_reordering_bak[idx] = tmp_reordering_bak[head];

		if (status[cur_node] == PREACTIVE) {
			int start_idx2 = row_offsets[cur_node], end_idx2 = row_offsets[cur_node + 1];

			for (int l = start_idx2; l < end_idx2; l++) {
				int column = column_indices[l];
				if (status[column] == POSTACTIVE) continue;
				ori_costs[column] += W1;
				pq.push(thrust::make_tuple(ori_costs[column], column));
				if (status[column] == INACTIVE) {
					//// tmp_reordering_bak[tail] = column;
					status[column] = PREACTIVE;
					//// tail ++;
				}
			}
		}

		status[cur_node] = POSTACTIVE;
		tmp_reordering[cur_idx ++] = cur_node;

		int start_idx2 = row_offsets[cur_node], end_idx2 = row_offsets[cur_node + 1];

		for (int l = start_idx2; l < end_idx2; l++) {
			int column = column_indices[l];
			if (status[column] != PREACTIVE) continue;
			ori_costs[column] += W1;
			status[column] = ACTIVE;
			pq.push(thrust::make_tuple(ori_costs[column], column));

			int start_idx3 = row_offsets[column], end_idx3 = row_offsets[column + 1];

			for (int l2 = start_idx3; l2 < end_idx3; l2++) {
				int column2 = column_indices[l2];
				if (status[column2] == POSTACTIVE) continue;

				ori_costs[column2] += W1;
				pq.push(thrust::make_tuple(ori_costs[column2], column2));
				if (status[column2] == INACTIVE) {
					status[column2] = PREACTIVE;
					//// tmp_reordering_bak[tail] = column2;
					//// tail ++;
				}
			}
		}
		//// head++;
	}
}

} // end namespace sloan

#endif
