import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112  # 8s margin

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Build data structures for fast access
    # subsets_of_block[j]: indices of t-subsets contained in k-subset j
    # blocks_of_subset[i]: indices of k-subsets containing t-subset i
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    # Convert to numpy arrays for vectorized operations where possible
    # Note: subsets_of_block is a list of lists of different lengths, so we keep it as list of lists
    # but we can use numpy for the 'gains' and 'uncovered' arrays.
    
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
    M_T = M.T  # CSC matrix, efficient for dot product with dense vector

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get duals (weights)
    # This provides a strong initial heuristic for the greedy search.
    duals = None
    try:
        if time.time() > deadline - 30:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        # Constraint: M.T @ x >= 1 => -M.T @ x <= -1.
        # A_ub should be n_t x n_k, which is -M (since M is n_t x n_k)
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=-M, b_ub=b_ub, bounds=bounds_lp, method='highs', 
                      options={"presolve": True, "time_limit": 8})

        if res.success:
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # Pre-compute base weights for blocks based on duals
    # Weight of a block is the sum of duals of the subsets it covers.
    # We will use this as the initial 'gain' in the greedy step.
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Heap-Based Greedy Search
    # We use the LP duals to guide the greedy selection.
    # As we pick blocks, we reduce the weight of subsets they cover,
    # effectively lowering the gain of blocks that share those subsets.
    # Using a heap for lazy max-extraction gives O(log n_k) per step.
    
    num_restarts = 5000
    rng = np.random.default_rng(42)
    
    # Pre-allocate delta_duals buffer
    delta_duals = np.zeros(n_t)
    max_base_weight = np.max(base_block_weights) if n_k > 0 else 1.0

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize gains with base weights + adaptive noise
        if restart == 0:
            gains = base_block_weights.copy()
        else:
            progress = restart / num_restarts
            noise_scale = max_base_weight * (0.02 + 0.15 * progress)
            gains = base_block_weights + rng.uniform(0, noise_scale, n_k)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        # Build heap: (-gain, block_index)
        heap = [(-gains[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        
        # Track which blocks have been selected
        selected = np.zeros(n_k, dtype=bool)
        
        while True:
            if time.time() > deadline - 2:
                break

            # Pop until we find a valid (not selected, positive gain) block
            best_idx = -1
            best_gain = 0
            while heap:
                neg_g, j = heapq.heappop(heap)
                g = -neg_g
                if not selected[j] and g > 1e-9:
                    best_idx = j
                    best_gain = g
                    break
                elif not selected[j] and g <= 1e-9:
                    # All remaining have non-positive gain
                    break
                # else: already selected, skip
            
            if best_idx == -1:
                # Fallback: find any block covering an uncovered subset
                if np.any(uncovered_mask):
                    found = False
                    for j in range(n_k):
                        if selected[j]:
                            continue
                        if any(uncovered_mask[s] for s in subsets_of_block[j]):
                            best_idx = j
                            found = True
                            break
                    if not found:
                        break
                else:
                    break

            # Pick best_idx
            solution_blocks.append(best_idx)
            selected[best_idx] = True
            
            # Update uncovered_mask
            subs_b = subsets_of_block[best_idx]
            newly_covered_indices = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered_indices:
                continue

            uncovered_mask[newly_covered_indices] = False
            
            # Vectorized Update of Gains
            delta_duals[:] = 0
            delta_duals[newly_covered_indices] = duals[newly_covered_indices]
            
            # Compute gain decrements for all blocks
            gain_decrements = M_T.dot(delta_duals)
            
            # Push updated gains back into heap for affected blocks
            # Only push blocks that are not yet selected
            for j in range(n_k):
                if not selected[j]:
                    new_gain = gains[j] - gain_decrements[j]
                    gains[j] = new_gain
                    if new_gain > 1e-9:
                        heapq.heappush(heap, (-new_gain, j))

            # Optimization: If all covered, break
            if not np.any(uncovered_mask):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()

    # 4. Fallback if no solution found (should not happen)
    if best_solution is None:
        # Simple greedy with heap
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        cov_counts = np.zeros(n_k, dtype=np.int32)
        for j in range(n_k):
            cov_counts[j] = len(subsets_of_block[j])
        heap = [(-cov_counts[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        selected = np.zeros(n_k, dtype=bool)
        
        while np.any(uncovered_mask):
            best_idx = -1
            while heap:
                neg_c, j = heapq.heappop(heap)
                c = -neg_c
                if not selected[j] and c > 0:
                    best_idx = j
                    break
            if best_idx == -1:
                break
            solution_blocks.append(best_idx)
            selected[best_idx] = True
            subs_b = subsets_of_block[best_idx]
            newly = [s for s in subs_b if uncovered_mask[s]]
            uncovered_mask[newly] = False
            # Update counts for other blocks
            for s in newly:
                for j in blocks_of_subset[s]:
                    if not selected[j]:
                        cov_counts[j] -= 1
                        if cov_counts[j] > 0:
                            heapq.heappush(heap, (-cov_counts[j], j))
        best_solution = solution_blocks
        best_size = len(best_solution)

    # 5. Post-processing: Deletion Pass
    if best_solution is not None and time.time() < deadline - 1:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break

                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)

            best_solution = new_solution
            best_size = len(best_solution)

    # 6. Swap-based local search
    if best_solution is not None and time.time() < deadline - 2:
        sol_set = set(best_solution)
        sol_list = list(best_solution)
        
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in sol_list:
            cover_count[subsets_of_block[b_idx]] += 1
        
        unique_subs = {}
        for b_idx in sol_list:
            subs_b = subsets_of_block[b_idx]
            uniq = [s for s in subs_b if cover_count[s] == 1]
            if uniq:
                unique_subs[b_idx] = uniq
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            new_sol_list = list(sol_list)
            
            # Try removal first
            for i, b_remove in enumerate(new_sol_list):
                if time.time() > deadline - 2:
                    break
                if b_remove not in unique_subs:
                    del new_sol_list[i]
                    improved = True
                    break
            
            if improved:
                sol_list = new_sol_list
                best_size = len(sol_list)
                best_solution = sol_list
                cover_count = np.zeros(n_t, dtype=np.int32)
                for b_idx in sol_list:
                    cover_count[subsets_of_block[b_idx]] += 1
                unique_subs = {}
                for b_idx in sol_list:
                    subs_b = subsets_of_block[b_idx]
                    uniq = [s for s in subs_b if cover_count[s] == 1]
                    if uniq:
                        unique_subs[b_idx] = uniq
                continue
            
            # Try swap
            for i, b_remove in enumerate(sol_list):
                if time.time() > deadline - 2:
                    break
                if b_remove not in unique_subs:
                    continue
                uniq_set = set(unique_subs[b_remove])
                
                best_j = -1
                best_overlap = -1
                for j in range(n_k):
                    if j in sol_set:
                        continue
                    subs_j = subsets_of_block[j]
                    # Quick check: does j cover all unique subsets?
                    covers_all = True
                    for s in uniq_set:
                        if s not in subs_j:
                            covers_all = False
                            break
                    if covers_all:
                        overlap = sum(1 for s in subs_j if cover_count[s] > 0)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_j = j
                
                if best_j != -1:
                    sol_list[i] = best_j
                    sol_set = set(sol_list)
                    improved = True
                    best_size = len(sol_list)
                    best_solution = sol_list
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx in sol_list:
                        cover_count[subsets_of_block[b_idx]] += 1
                    unique_subs = {}
                    for b_idx in sol_list:
                        subs_b = subsets_of_block[b_idx]
                        uniq = [s for s in subs_b if cover_count[s] == 1]
                        if uniq:
                            unique_subs[b_idx] = uniq
                    break

    # 7. Final cleanup
    if best_solution is not None:
        seen = set()
        unique_sol = []
        for b in best_solution:
            if b not in seen:
                seen.add(b)
                unique_sol.append(b)
        best_solution = unique_sol
        best_size = len(best_solution)
    
    return [list(k_subsets[idx]) for idx in best_solution]
