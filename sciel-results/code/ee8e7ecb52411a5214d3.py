import itertools
import random
import time
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_masks = []
    for block in all_k_subsets:
        mask = 0
        for s in itertools.combinations(block, t):
            mask |= (1 << t_subset_to_idx[s])
        block_t_masks.append(mask)
        
    # Map each t-subset to the blocks that cover it (for potential use, or just masks)
    # Not strictly needed if we use masks for everything.

    # 2. Phase 1: Greedy Packing (Independent Set)
    pool = set(range(num_k_subsets))
    core_blocks = []
    covered_mask = 0
    
    while pool:
        best_block = -1
        max_new_cov = -1
        
        for b_idx in pool:
            new_cov = (block_t_masks[b_idx] & ~covered_mask).bit_count()
            if new_cov > max_new_cov:
                max_new_cov = new_cov
                best_block = b_idx
                
        if best_block == -1:
            break
            
        core_blocks.append(best_block)
        covered_mask |= block_t_masks[best_block]
        
        best_mask = block_t_masks[best_block]
        new_pool = set()
        for b_idx in pool:
            if b_idx != best_block and (block_t_masks[b_idx] & best_mask) == 0:
                new_pool.add(b_idx)
        pool = new_pool

    # 3. Phase 2: Solve Residual Set Cover
    uncovered_mask = ((1 << num_t_subsets) - 1) & ~covered_mask
    
    if uncovered_mask:
        uncovered_indices = []
        m = uncovered_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            
        # Candidates: blocks not in core that cover uncovered
        core_set = set(core_blocks)
        candidates = []
        cand_masks = []
        for b_idx in range(num_k_subsets):
            if b_idx in core_set:
                continue
            m = block_t_masks[b_idx] & uncovered_mask
            if m:
                candidates.append(b_idx)
                cand_masks.append(m)
                
        if candidates:
            solved_by_ilp = False
            start_time = time.time()
            try:
                n_vars = len(candidates)
                n_cons = len(uncovered_indices)
                
                t_idx_to_row = {t: i for i, t in enumerate(uncovered_indices)}
                
                rows = []
                cols = []
                data = []
                
                for j, mask in enumerate(cand_masks):
                    m = mask
                    while m:
                        lsb = m & -m
                        t_idx = lsb.bit_length() - 1
                        row_idx = t_idx_to_row[t_idx]
                        rows.append(row_idx)
                        cols.append(j)
                        data.append(1)
                        m ^= lsb
                
                A = sparse.csr_matrix((data, (rows, cols)), shape=(n_cons, n_vars))
                c = np.ones(n_vars)
                cons = LinearConstraint(A, lb=np.ones(n_cons), ub=np.inf)
                bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
                integrality = np.ones(n_vars)
                
                # Limit time to 60s
                res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, options={'time_limit': 60})
                
                if res.status == 0:
                    selected = np.where(res.x > 0.5)[0]
                    for idx in selected:
                        core_blocks.append(candidates[idx])
                    solved_by_ilp = True
                elif time.time() - start_time > 60:
                    raise TimeoutError
            except Exception:
                pass

            if not solved_by_ilp:
                # Fallback: Greedy
                residual_mask = uncovered_mask
                # To avoid O(N^2) in greedy, we can just iterate.
                # 50k * 4k is large, but masks make it fast.
                while residual_mask:
                    best_block = -1
                    max_gain = -1
                    # Find best candidate
                    for j, mask in enumerate(cand_masks):
                        gain = (mask & residual_mask).bit_count()
                        if gain > max_gain:
                            max_gain = gain
                            best_block = j
                    if best_block == -1:
                        break
                    core_blocks.append(candidates[best_block])
                    residual_mask &= ~cand_masks[best_block]

    # 4. Minimalization
    active_blocks = set(core_blocks)
    cover_count = [0] * num_t_subsets
    for b_idx in active_blocks:
        m = block_t_masks[b_idx]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb

    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                m = block_t_masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] <= 1:
                        can_remove = False
                        break
                    m ^= lsb
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    m = block_t_masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_counts[idx] -= 1
                        m ^= lsb
                    changed = True
        return active_blocks_set, cover_counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)
    
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
