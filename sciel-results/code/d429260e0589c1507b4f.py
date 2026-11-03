import itertools
import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

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
        if time.time() > deadline - 20:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        A_ub = -M_T.tocsr() # M_T is CSC, convert to CSR for linperf? Linprog accepts sparse.
        # Actually, A_ub should be n_t x n_k. M is n_t x n_k. M.T is n_k x n_t.
        # Constraint: M.T @ x >= 1 => -M.T @ x <= -1.
        # So A_ub = -M.T.
        # M is CSR. M.T is CSC.
        # Linprog handles sparse A_ub.
        
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=-M_T, b_ub=b_ub, bounds=bounds_lp, method='highs', 
                      options={"presolve": True, "time_limit": 2})

        if res.success:
            # Dual variables are associated with the constraints M.T @ x >= 1
            # In linprog, for A_ub @ x <= b_ub, the marginals are for the constraints.
            # Our constraint is -M.T @ x <= -1.
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

    # 3. Multi-start Vectorized Greedy Search
    # We use the LP duals to guide the greedy selection.
    # As we pick blocks, we reduce the weight of subsets they cover, 
    # effectively lowering the gain of blocks that share those subsets.
    
    num_restarts = 10000 # Increased significantly due to vectorization
    rng = np.random.default_rng(42)
    
    # Pre-allocate delta_duals buffer
    delta_duals = np.zeros(n_t)
    max_base_weight = np.max(base_block_weights) if n_k > 0 else 1.0

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize gains with base weights + adaptive noise
        # Noise increases with restart count to explore more diverse solutions
        if restart == 0:
            gains = base_block_weights.copy()
        else:
            # Adaptive noise: starts small, grows to encourage exploration
            progress = restart / num_restarts
            noise_scale = max_base_weight * (0.02 + 0.15 * progress)
            gains = base_block_weights + rng.uniform(0, noise_scale, n_k)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        while True:
            if time.time() > deadline - 2:
                break

            # Find the block with the highest gain
            best_idx = np.argmax(gains)
            
            if gains[best_idx] <= 1e-9:
                # If max gain is ~0, we are stuck or done.
                if np.any(uncovered_mask):
                    # Fallback: pick any block that covers at least one uncovered subset
                    found = False
                    for j in range(n_k):
                        # Check if block j covers any uncovered subset
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
            
            # Update uncovered_mask
            subs_b = subsets_of_block[best_idx]
            newly_covered_indices = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered_indices:
                # Should not happen if gain > 0
                continue

            uncovered_mask[newly_covered_indices] = False
            
            # Vectorized Update of Gains
            # Zero out the buffer (or just overwrite the relevant parts? 
            # Since we only add to gains (subtract), and we start from 0, 
            # we can just create a new vector or reuse.
            # Reusing is faster. We need to ensure non-newly-covered parts are 0.
            # Since we only set newly_covered_indices, we must zero the rest?
            # No, delta_duals is used for M.T @ delta_duals.
            # If we don't zero the old values, they will be included.
            # So we must zero the whole vector or only the old non-zero parts.
            # Zeroing the whole vector (n_t=3876) is cheap.
            delta_duals[:] = 0
            delta_duals[newly_covered_indices] = duals[newly_covered_indices]
            
            # Subtract the contribution from all gains
            gains -= M_T.dot(delta_duals)

            # Optimization: If all covered, break
            if not np.any(uncovered_mask):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            # Verify coverage just in case
            # We can check if uncovered_mask is all False
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()
                
                # Early exit if we hit a very good score?
                # 151 is the target. 150 would be a breakthrough.
                # If we find 151, we might continue to try for 150.
                # If we find 145, we can stop.
                if best_size <= 151:
                    # Keep searching for a while
                    pass

    # 4. Fallback if no solution found (should not happen)
    if best_solution is None:
        # Simple greedy without weights
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask):
            best_idx = -1
            max_cov = -1
            for j in range(n_k):
                cnt = np.sum(uncovered_mask[subsets_of_block[j]])
                if cnt > max_cov:
                    max_cov = cnt
                    best_idx = j
            if best_idx == -1:
                break
            solution_blocks.append(best_idx)
            uncovered_mask[subsets_of_block[best_idx]] = False
        best_solution = solution_blocks
        best_size = len(best_solution)

    # 5. Post-processing: Deletion Pass
    # Try to remove blocks that are redundant.
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
                # If any subset in this block is covered ONLY by this block, we can't remove it
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

    # 6. Swap-based local search: try replacing one block with another to reduce size
    if best_solution is not None and time.time() < deadline - 2:
        # Build a mapping from block index to its position in solution
        sol_set = set(best_solution)
        sol_list = list(best_solution)
        
        # For each block in the solution, compute which subsets it uniquely covers
        # (i.e., subsets where cover_count == 1 and this block is the only cover)
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in sol_list:
            cover_count[subsets_of_block[b_idx]] += 1
        
        # For each block in solution, find subsets it uniquely covers
        unique_subs = {}
        for b_idx in sol_list:
            subs_b = subsets_of_block[b_idx]
            uniq = [s for s in subs_b if cover_count[s] == 1]
            if uniq:
                unique_subs[b_idx] = uniq
        
        # Try to swap: remove block i, add block j that covers all uniquely-covered subsets of i
        # and doesn't introduce new unique constraints
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            new_sol_list = list(sol_list)
            
            for i, b_remove in enumerate(new_sol_list):
                if time.time() > deadline - 2:
                    break
                if b_remove not in unique_subs:
                    # This block has no uniquely covered subsets, so removing it is safe
                    # Try to remove it
                    del new_sol_list[i]
                    improved = True
                    break
            
            if improved:
                sol_list = new_sol_list
                best_size = len(sol_list)
                best_solution = sol_list
                # Recompute cover counts
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
            
            # No simple removal possible; try swap
            for i, b_remove in enumerate(sol_list):
                if time.time() > deadline - 2:
                    break
                if b_remove not in unique_subs:
                    continue
                uniq_set = set(unique_subs[b_remove])
                
                # Find a block j not in solution that covers all subsets in uniq_set
                # and has high overlap with the solution (to minimize disruption)
                best_j = -1
                best_overlap = -1
                for j in range(n_k):
                    if j in sol_set:
                        continue
                    subs_j = set(subsets_of_block[j])
                    if uniq_set.issubset(subs_j):
                        # Count how many subsets of j are already covered by solution
                        overlap = sum(1 for s in subsets_of_block[j] if cover_count[s] > 0)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_j = j
                
                if best_j != -1:
                    # Perform the swap
                    sol_list[i] = best_j
                    sol_set = set(sol_list)
                    improved = True
                    best_size = len(sol_list)
                    best_solution = sol_list
                    # Recompute cover counts
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

    # 7. Final cleanup: ensure no duplicates and sort for determinism
    if best_solution is not None:
        # Remove any duplicate blocks
        seen = set()
        unique_sol = []
        for b in best_solution:
            if b not in seen:
                seen.add(b)
                unique_sol.append(b)
        best_solution = unique_sol
        best_size = len(best_solution)
    
    return [list(k_subsets[idx]) for idx in best_solution]
