import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
import itertools
import time

def construct(v, k, t):
    # Parameters
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    # Map t-subsets to indices
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Build incidence matrix A (n_t_subsets x n_blocks)
    # A[i, j] = 1 if t_subset i is in block j
    rows = []
    cols = []
    
    for j, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            i = t_subset_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            
    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_t_subsets, n_blocks))
    
    # Fast Greedy Heuristic for initial incumbent and fallback
    uncovered = np.ones(n_t_subsets, dtype=bool)
    selected_blocks = []
    
    # Precompute block coverage for greedy
    block_covers = [[] for _ in range(n_blocks)]
    for i_idx, i in enumerate(rows):
        block_covers[cols[i_idx]].append(i)
        
    # Greedy Loop
    A_T = A.T.tocsr()
    
    while np.any(uncovered):
        gains = A_T @ uncovered.astype(np.float32)
        best_block = int(np.argmax(gains))
        
        if gains[best_block] == 0:
            break
            
        selected_blocks.append(best_block)
        
        for ts_idx in block_covers[best_block]:
            uncovered[ts_idx] = False
            
    # Post-processing: Remove redundant blocks
    final_selected = []
    coverage_count = np.zeros(n_t_subsets, dtype=np.int32)
    for b in selected_blocks:
        for ts_idx in block_covers[b]:
            coverage_count[ts_idx] += 1
            
    for b in selected_blocks:
        for ts_idx in block_covers[b]:
            coverage_count[ts_idx] -= 1
            
        if np.any(coverage_count < 1):
            for ts_idx in block_covers[b]:
                coverage_count[ts_idx] += 1
            final_selected.append(b)
            
    greedy_blocks = [list(all_blocks[i]) for i in final_selected]
    greedy_size = len(greedy_blocks)
    
    # ILP Setup
    c = np.ones(n_blocks)
    constraints = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_blocks)
    
    start_time = time.time()
    best_solution = greedy_blocks
    best_size = greedy_size
    
    try:
        # Use remaining time for ILP
        # We are allowed 120s total. Matrix construction + Greedy took some time.
        # Let's reserve 5s for overhead and use 110s for ILP.
        ilp_time_limit = 110 - (time.time() - start_time)
        if ilp_time_limit < 10:
            ilp_time_limit = 10 # Minimum reasonable time
            
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": ilp_time_limit, "mip_rel_gap": 0.0})
        
        if res.x is not None:
            x_int = np.round(res.x).astype(int)
            # Verify validity
            if np.all(A @ x_int >= 1):
                ilp_blocks = [list(all_blocks[i]) for i in np.where(x_int == 1)[0]]
                if len(ilp_blocks) < best_size:
                    best_solution = ilp_blocks
                    best_size = len(ilp_blocks)
                    
    except Exception:
        pass
        
    return best_solution
