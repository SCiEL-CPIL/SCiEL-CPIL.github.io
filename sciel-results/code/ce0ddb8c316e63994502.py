import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 120s limit
    
    points = list(range(v))
    
    # 1. Precomputation
    # Generate all k-subsets (potential blocks)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # Generate all t-subsets
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Map block -> list of t-subset indices
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)
        
    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)
            
    def run_greedy(seed):
        random.seed(seed)
        # Initial gains: number of t-subsets covered by each block
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []
        
        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            # Find all blocks with max gain
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            
            selected_indices.append(best_block_idx)
            
            # Update coverage and gains
            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    block_gain[tsub_blocks[si]] -= 1
                        
        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []
        
        n_sel = len(selected_indices)
        # Precompute local index array for O(1) lookup
        local_idx_arr = np.full(N_blocks, -1, dtype=np.int32)
        for i, g in enumerate(selected_indices):
            local_idx_arr[g] = i
        
        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)
        
        # Compute total_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1
                
        # Compute crit_count
        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1
                    
        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = [False] * n_sel
        
        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue
                
            removed[i] = True
            g = selected_indices[i]
            
            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]
                
                if old_c == 2 and new_c == 1:
                    # si is now critical. Update crit_count for all other blocks containing si.
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        i_other = local_idx_arr[g_other]
                        if i_other >= 0 and not removed[i_other]:
                            crit_count[i_other] += 1
                
                elif old_c == 1:
                    # Should not happen if crit_count[i] == 0
                    pass
        
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search_swap(selected_indices, time_budget):
        """
        Try to improve the solution by swapping blocks or removing redundant ones.
        """
        if not selected_indices:
            return selected_indices
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            n_sel = len(selected_indices)
            global_to_local = {g: i for i, g in enumerate(selected_indices)}
            
            # Recompute coverage
            total_count = np.zeros(N_t, dtype=np.int32)
            for g in selected_indices:
                for si in block_tsubs[g]:
                    total_count[si] += 1
            
            # For each block, check if it can be removed or swapped
            for i in range(n_sel):
                if time.time() > time_budget:
                    break
                    
                g = selected_indices[i]
                unique_subs = []
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        unique_subs.append(si)
                
                if not unique_subs:
                    # Can remove this block
                    selected_indices.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all unique_subs
                # and adds minimal new t-subsets
                if len(unique_subs) > 30:
                    continue  # Too expensive to search for replacement
                
                # Find candidate blocks that cover all unique_subs
                candidate_blocks = None
                for si in unique_subs:
                    if candidate_blocks is None:
                        candidate_blocks = set(tsub_blocks[si])
                    else:
                        candidate_blocks &= set(tsub_blocks[si])
                
                if candidate_blocks is None or not candidate_blocks:
                    continue
                
                # Remove current block from candidates
                candidate_blocks.discard(g)
                
                # For each candidate, compute the "cost": number of new t-subsets it covers
                best_candidate = None
                best_cost = len(block_tsubs[g])  # Cost of keeping current block
                for g_cand in candidate_blocks:
                    if g_cand in global_to_local:
                        continue  # Already in solution
                    # Count new t-subsets covered by g_cand
                    new_subs = 0
                    for si in block_tsubs[g_cand]:
                        if total_count[si] == 0:
                            new_subs += 1
                    cost = new_subs
                    if cost < best_cost:
                        best_cost = cost
                        best_candidate = g_cand
                
                if best_candidate is not None and best_cost < len(block_tsubs[g]):
                    # Swap: remove g, add best_candidate
                    selected_indices[i] = best_candidate
                    improved = True
                    break
        
        return selected_indices

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    sol_idx = local_search_swap(sol_idx, DEADLINE - 40.0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol = [all_blocks[i] for i in sol_idx]
    best_len = len(best_sol)
    
    # Randomized restarts with varying time budgets
    attempts = 0
    while time.time() < DEADLINE - 15.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)
        
        # Give more time to local search for early attempts
        remaining = DEADLINE - time.time()
        ls_budget = min(remaining * 0.3, 20.0)
        cand_idx = local_search_swap(cand_idx, time.time() + ls_budget)
        cand_idx = reduce_solution_fast(cand_idx)
        
        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol = [all_blocks[i] for i in cand_idx]
        
        attempts += 1
        # If we've found a good solution, be more conservative with time
        if best_len < 340 and attempts > 5:
            break
            
    return best_sol
