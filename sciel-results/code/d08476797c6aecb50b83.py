import itertools
import time
import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    deadline = time.monotonic() + 115.0
    points = list(range(v))
    
    # 1. Generate all t-subsets and k-blocks
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)
    
    # 2. Build sparse incidence matrix A (num_tsubs x num_kblocks)
    # A[i, j] = 1 if tsub i is contained in kblock j
    row_indices = []
    col_indices = []
    data = []
    
    for j, block in enumerate(all_kblocks):
        # Find all t-subsets of this block
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csc_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    
    # 3. Greedy Construction
    covered = np.zeros(num_tsubs, dtype=bool)
    x0 = np.zeros(num_kblocks, dtype=float)
    selected_blocks = []
    
    # Precompute t-subset indices for each block for fast updating
    block_tsub_idxs = []
    for j, block in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsub_idxs.append(idxs)
        
    while time.monotonic() < deadline - 5.0:
        u = ~covered
        gains = A.T @ u
        best_gain = gains.max()
        if best_gain == 0:
            break
        
        # Find all blocks with max gain, pick one deterministically
        best_idx = int(np.argmax(gains))
        
        selected_blocks.append(best_idx)
        x0[best_idx] = 1.0
        
        # Update covered
        for idx in block_tsub_idxs[best_idx]:
            covered[idx] = True
            
        if np.all(covered):
            break
            
    # 4. Polish (Remove redundant blocks) - aggressive multi-pass
    counts = np.zeros(num_tsubs, dtype=int)
    for b_idx in selected_blocks:
        for idx in block_tsub_idxs[b_idx]:
            counts[idx] += 1
    
    # Use a set for O(1) removal
    selected_set = set(selected_blocks)
    
    changed = True
    while changed and time.monotonic() < deadline - 15.0:
        changed = False
        # Iterate in reverse to allow more removals in one pass
        for b_idx in sorted(selected_set, reverse=True):
            can_remove = True
            for idx in block_tsub_idxs[b_idx]:
                if counts[idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                selected_set.discard(b_idx)
                x0[b_idx] = 0.0
                for idx in block_tsub_idxs[b_idx]:
                    counts[idx] -= 1
                changed = True
    
    selected_blocks = sorted(selected_set)
                
    # 5. ILP Optimization - use remaining time wisely
    remaining = deadline - time.monotonic()
    if remaining > 20.0:
        try:
            c = np.ones(num_kblocks)
            constraints = LinearConstraint(A, 1, np.inf)
            integrality = np.ones(num_kblocks)
            bounds = Bounds(0, 1)
            
            ilp_time = max(10.0, remaining - 5.0)
            res = milp(
                c=c,
                constraints=constraints,
                integrality=integrality,
                bounds=bounds,
                x0=x0,
                time_limit=ilp_time
            )
            
            if res.x is not None:
                x_sol = np.round(res.x)
                if np.all((x_sol == 0) | (x_sol == 1)):
                    coverage = A @ x_sol
                    if np.all(coverage >= 1):
                        new_count = int(np.sum(x_sol))
                        old_count = len(selected_blocks)
                        if new_count < old_count:
                            selected_blocks = [i for i in range(num_kblocks) if x_sol[i] == 1]
                            # Re-polish after ILP
                            counts2 = np.zeros(num_tsubs, dtype=int)
                            for b_idx in selected_blocks:
                                for idx in block_tsub_idxs[b_idx]:
                                    counts2[idx] += 1
                            sel_set2 = set(selected_blocks)
                            changed2 = True
                            while changed2 and time.monotonic() < deadline - 5.0:
                                changed2 = False
                                for b_idx in sorted(sel_set2, reverse=True):
                                    can_rem = True
                                    for idx in block_tsub_idxs[b_idx]:
                                        if counts2[idx] == 1:
                                            can_rem = False
                                            break
                                    if can_rem:
                                        sel_set2.discard(b_idx)
                                        for idx in block_tsub_idxs[b_idx]:
                                            counts2[idx] -= 1
                                        changed2 = True
                            selected_blocks = sorted(sel_set2)
                            
        except Exception:
            pass
            
    # Convert indices back to lists
    result_blocks = [list(all_kblocks[i]) for i in selected_blocks]
    
    return result_blocks
