import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
import heapq
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using LP-guided greedy with dynamic weights.
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    # Generate all t-subsets and index them
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Build incidence structure
    rows = []
    cols = []
    data = []
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    
    # LP Formulation:
    # Minimize sum(x_j)
    # Subject to: M.T @ x >= 1 for each t-subset
    # 0 <= x_j <= 1
    c = np.ones(n_k)
    A_ub = -M.T.tocsr()
    b_ub = -np.ones(n_t)
    bounds = [(0, 1)] * n_k
    
    duals = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.ineqlin.marginals is not None:
            # Dual variables: y_i = -marginal[i] for constraint -M.T @ x <= -1
            # The marginal for a <= constraint in a minimization problem is <= 0
            # The dual variable for the original >= constraint is the negative of this
            duals = -res.ineqlin.marginals
            duals = np.maximum(duals, 0)
        else:
            duals = np.ones(n_t) / n_t
    except Exception:
        duals = np.ones(n_t) / n_t
    
    # Dynamic LP-Greedy with heap for efficient updates
    block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            block_weights[j] = np.sum(duals[subsets_of_block[j]])
            
    # Initialize heap with (-weight, block_index)
    heap = [(-block_weights[j], j) for j in range(n_k)]
    heapq.heapify(heap)
    
    covered = np.zeros(n_t, dtype=bool)
    picked = np.zeros(n_k, dtype=bool)
    blocks = []
    
    while heap:
        neg_w, j = heapq.heappop(heap)
        w = -neg_w
        
        # Skip if already picked
        if picked[j]:
            continue
            
        # Skip if weight is stale (old entry in heap)
        if w > block_weights[j] + 1e-9:
            continue
            
        # Skip if no positive weight (no uncovered subsets with positive dual)
        if w <= 1e-9:
            continue
            
        # Select block j
        picked[j] = True
        blocks.append(list(k_subsets[j]))
        
        # Update coverage and weights
        for s_idx in subsets_of_block[j]:
            if not covered[s_idx]:
                covered[s_idx] = True
                y_s = duals[s_idx]
                if y_s > 0:
                    for b_prime in blocks_of_subset[s_idx]:
                        if not picked[b_prime]:
                            block_weights[b_prime] -= y_s
                            heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                            
    # Check if all t-subsets are covered
    if not np.all(covered):
        # Fallback for remaining uncovered subsets
        remaining_indices = np.where(~covered)[0]
        rem_blocks = _greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets)
        blocks.extend(rem_blocks)
        
    return blocks

def _greedy_remaining(v, k, t, remaining_indices, t_index, k_subsets):
    """
    Greedy fallback to cover remaining t-subsets.
    """
    inv_t_index = {i: s for s, i in t_index.items()}
    remaining_sets = set(inv_t_index[i] for i in remaining_indices)
    
    if not remaining_sets:
        return []
        
    blocks = []
    
    while remaining_sets:
        best_block = None
        best_count = -1
        for B in k_subsets:
            count = 0
            for S in itertools.combinations(B, t):
                if S in remaining_sets:
                    count += 1
            if count > best_count:
                best_count = count
                best_block = B
        if best_count == 0:
            break
        blocks.append(list(best_block))
        for S in itertools.combinations(best_block, t):
            if S in remaining_sets:
                remaining_sets.remove(S)
    return blocks
