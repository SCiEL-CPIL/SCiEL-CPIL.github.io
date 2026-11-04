"""Covering design C(19,8,5): greedy set-cover + redundancy removal."""
import time
import itertools
import numpy as np
from scipy import sparse

def _binom_table(max_n, max_r):
    C = np.zeros((max_n + 1, max_r + 1), dtype=np.int64)
    C[:, 0] = 1
    for n in range(1, max_n + 1):
        C[n, 0] = 1
        for r in range(1, min(n, max_r) + 1):
            C[n, r] = C[n - 1, r - 1] + C[n - 1, r]
    return C

def _subset_index(v, t, binom):
    """Map a t-subset (sorted tuple) to its combinatorial number system index."""
    def idx(s):
        total = 0
        for x in s:
            total += int(binom[v - 1 - x, t - 1 - x])
        return total
    return idx

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0
    
    # 1. Precompute binomial coefficients
    binom = _binom_table(v, max(k, t))
    M = int(binom[v, t])
    idx_func = _subset_index(v, t, binom)
    
    # 2. Generate candidates and their coverage indices
    candidates = []
    cov_indices = []
    
    for c in itertools.combinations(range(v), k):
        candidates.append(c)
        subs = list(itertools.combinations(c, t))
        indices = [idx_func(sub) for sub in subs]
        cov_indices.append(indices)
        
    N = len(candidates)
    
    # 3. Build sparse coverage matrix A (N x M)
    row_indices = []
    col_indices = []
    data = []
    
    for i, inds in enumerate(cov_indices):
        for j in inds:
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(N, M), dtype=np.int8)
    
    # 4. Greedy Set Cover
    uncovered = np.ones(M, dtype=np.int8)
    chosen = []
    
    while True:
        if time.time() > deadline:
            break
            
        if not uncovered.any():
            break
            
        # Calculate gain for each candidate
        gains = A.dot(uncovered)
        
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        chosen.append(best_idx)
        
        # Update uncovered
        cols = A[best_idx].nonzero()[1]
        uncovered[cols] = 0
        
    # 5. Redundancy Removal
    cov_counts = np.zeros(M, dtype=np.int32)
    for idx in chosen:
        cols = A[idx].nonzero()[1]
        cov_counts[cols] += 1
        
    chosen_list = list(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(chosen_list) - 1, -1, -1):
            idx = chosen_list[i]
            cols = A[idx].nonzero()[1]
            if np.all(cov_counts[cols] >= 2):
                cov_counts[cols] -= 1
                chosen_list.pop(i)
                improved = True
    
    return [list(candidates[i]) for i in chosen_list]
