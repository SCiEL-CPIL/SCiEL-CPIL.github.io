import itertools
import numpy as np
import time
import random
import heapq
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

    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get duals (weights)
    duals = None
    try:
        if time.time() > deadline - 20:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs', 
                      options={"presolve": True, "time_limit": 30})

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
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Lazy Greedy Search
    # Uses a priority queue with cached gains and submodular lazy evaluation.
    
    num_restarts = 300
    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        if restart == 0:
            init_weights = base_block_weights.copy()
        else:
            noise_scale = np.max(base_block_weights) * 0.05
            init_weights = base_block_weights + rng.uniform(0, noise_scale, n_k)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        # Lazy greedy with priority queue
        versions = np.zeros(n_k, dtype=np.int32)
        cached_gains = init_weights.copy()
        
        # Build initial heap
        heap = []
        for j in range(n_k):
            if cached_gains[j] > 1e-9:
                heapq.heappush(heap, (-cached_gains[j], j, 0))
        
        while heap and time.time() < deadline - 2:
            neg_gain, best_idx, ver = heapq.heappop(heap)
            
            if ver != versions[best_idx]:
                continue
            
            # Recompute actual gain for this block
            subs_b = subsets_of_block[best_idx]
            actual_gain = 0.0
            for s in subs_b:
                if uncovered_mask[s]:
                    actual_gain += duals[s]
            
            if actual_gain <= 1e-9:
                continue
            
            # Accept this block
            solution_blocks.append(best_idx)
            
            # Update uncovered_mask
            newly_covered = []
            for s in subs_b:
                if uncovered_mask[s]:
                    newly_covered.append(s)
                    uncovered_mask[s] = False
            
            if not newly_covered:
                continue
            
            # Update cached gains for all blocks that share newly covered subsets
            for s in newly_covered:
                w = duals[s]
                for j2 in blocks_of_subset[s]:
                    if j2 == best_idx:
                        continue
                    cached_gains[j2] -= w
                    versions[j2] += 1
                    if cached_gains[j2] > 1e-9:
                        heapq.heappush(heap, (-cached_gains[j2], j2, versions[j2]))
            
            if not np.any(uncovered_mask):
                break

        if len(solution_blocks) < best_size:
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()

    # 4. Fallback if no solution found
    if best_solution is None:
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask) and time.time() < deadline - 1:
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

    # 5. Post-processing: Vectorized Deletion Pass
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
                if not np.any(cover_count[subs_b] == 1):
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)

            best_solution = new_solution
            best_size = len(best_solution)

    # 6. Local Search: Try swapping blocks to improve coverage
    if best_solution is not None and time.time() < deadline - 2:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            new_solution = list(best_solution)
            
            for i in range(len(new_solution)):
                if time.time() > deadline - 2:
                    break
                
                b_old = new_solution[i]
                subs_old = subsets_of_block[b_old]
                
                unique_subs = [s for s in subs_old if cover_count[s] == 1]
                
                if len(unique_subs) == 0:
                    cover_count[subs_old] -= 1
                    new_solution.pop(i)
                    improved = True
                    break
                else:
                    if len(unique_subs) <= 3:
                        candidates = set(blocks_of_subset[unique_subs[0]])
                        for s in unique_subs[1:]:
                            candidates &= set(blocks_of_subset[s])
                        candidates.discard(b_old)
                        
                        best_candidate = None
                        best_score = -1
                        for cand in candidates:
                            subs_cand = subsets_of_block[cand]
                            score = np.sum(cover_count[subs_cand] <= 1)
                            if score > best_score:
                                best_score = score
                                best_candidate = cand
                        
                        if best_candidate is not None and best_score > len(unique_subs):
                            cover_count[subs_old] -= 1
                            subs_new = subsets_of_block[best_candidate]
                            cover_count[subs_new] += 1
                            new_solution[i] = best_candidate
                            improved = True
                            break
            
            if improved:
                best_solution = new_solution
                best_size = len(best_solution)

    # 7. Vectorized 2-opt local search: try removing two blocks and adding one
    if best_solution is not None and time.time() < deadline - 3:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1
        
        improved_2opt = True
        while improved_2opt and time.time() < deadline - 3:
            improved_2opt = False
            sol_copy = list(best_solution)
            
            for i in range(len(sol_copy)):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, len(sol_copy)):
                    if time.time() > deadline - 3:
                        break
                    
                    b1 = sol_copy[i]
                    b2 = sol_copy[j]
                    subs1 = subsets_of_block[b1]
                    subs2 = subsets_of_block[b2]
                    
                    # Count subsets that would become uncovered
                    # A subset is uniquely covered if cover_count[s] == 1
                    # and it's in subs1 or subs2
                    unique_to_remove = []
                    for s in subs1:
                        if cover_count[s] == 1:
                            unique_to_remove.append(s)
                    for s in subs2:
                        if cover_count[s] == 1:
                            unique_to_remove.append(s)
                    
                    if not unique_to_remove:
                        # Can remove both
                        cover_count[subs1] -= 1
                        cover_count[subs2] -= 1
                        sol_copy.pop(j)
                        sol_copy.pop(i)
                        improved_2opt = True
                        break
                    elif len(unique_to_remove) <= 4:
                        # Try to find a single block covering all unique_to_remove
                        candidates = set(blocks_of_subset[unique_to_remove[0]])
                        for s in unique_to_remove[1:]:
                            candidates &= set(blocks_of_subset[s])
                        candidates.discard(b1)
                        candidates.discard(b2)
                        
                        best_candidate = None
                        best_score = -1
                        for cand in candidates:
                            subs_cand = subsets_of_block[cand]
                            # Score: number of subsets that would be newly well-covered
                            score = np.sum(cover_count[subs_cand] <= 1)
                            if score > best_score:
                                best_score = score
                                best_candidate = cand
                        
                        if best_candidate is not None and best_score > len(unique_to_remove):
                            cover_count[subs1] -= 1
                            cover_count[subs2] -= 1
                            subs_new = subsets_of_block[best_candidate]
                            cover_count[subs_new] += 1
                            sol_copy.pop(j)
                            sol_copy[i] = best_candidate
                            improved_2opt = True
                            break
                if improved_2opt:
                    break
            
            if improved_2opt:
                best_solution = sol_copy
                best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
