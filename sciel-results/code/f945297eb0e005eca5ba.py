import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Precompute subsets_of_block: for each block j, list of t-subset indices
    subsets_of_block = [[] for _ in range(n_k)]
    # Precompute blocks_of_subset: for each t-subset i, list of block indices
    blocks_of_subset = [[] for _ in range(n_t)]
    
    # Build sparse matrix for LP
    rows = []
    cols = []
    data = []
    
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            rows.append(i)
            cols.append(j)
            data.append(1.0)
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            
    best_solution = None
    best_size = float('inf')
    
    # 2. Try LP Relaxation to get duals
    duals = None
    try:
        if time.time() > deadline - 5:
            raise TimeoutError()
            
        M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
        
        # Minimize sum(x_j)
        # Subject to: M.T @ x >= ones(n_t)  =>  -M.T @ x <= -ones(n_t)
        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            # Duals for the constraints A_ub x <= b_ub
            # In minimization, marginals for <= constraints are typically <= 0 if active?
            # Or they represent the shadow price.
            # For set cover, we want weights for the sets (t-subsets).
            # The dual of min sum x s.t. M^T x >= 1 is max sum y s.t. M y <= 1, y >= 0.
            # Here we solved min sum x s.t. -M^T x <= -1.
            # The dual variables correspond to the constraints.
            duals = res.ineqlin.marginals
            # Ensure non-negative weights for the greedy step
            # If marginals are negative, taking abs is a common heuristic fix if signs are flipped
            duals = np.abs(duals)
            
            if np.sum(duals) == 0:
                duals = None
        else:
            duals = None
            
    except Exception:
        duals = None
        
    # 3. Dynamic Greedy with LP Duals
    if duals is None:
        duals = np.ones(n_t)
        
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize weights
        # Weight of block j = sum of duals of subsets in block j
        # We will use a heap to dynamically select the best block
        # The "gain" of a block is the sum of weights of its *uncovered* subsets.
        
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals[subsets_of_block[j]])
                
        # Add noise for diversification
        if restart > 0:
            block_weights += np.random.uniform(0, 0.01, n_k)
            
        # Heap of (-weight, index)
        heap = [(-block_weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        while heap:
            if time.time() > deadline - 5:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
                
            # Lazy deletion check
            if w > block_weights[j] + 1e-9:
                continue
                
            if w <= 1e-9:
                break
                
            # Select block j
            picked[j] = True
            solution_blocks.append(j)
            
            # Update weights
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 0:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                                
            if np.all(covered):
                break
                
        if len(solution_blocks) < best_size and np.all(covered):
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
    # Fallback
    if best_solution is None:
        # Standard greedy
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = 0
                for s in subsets_of_block[j]:
                    if s in uncovered:
                        cnt += 1
                if cnt > max_cov:
                    max_cov = cnt
                    best_b = j
            if best_b == -1 or max_cov == 0:
                break
            res_blocks.append(best_b)
            for s in subsets_of_block[best_b]:
                if s in uncovered:
                    uncovered.remove(s)
        best_solution = res_blocks
        best_size = len(res_blocks)
        
    # 4. Post-processing: Deletion Pass
    if best_solution is not None:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 5:
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
            
    # 5. Local Search: Swap and Delete
    # Try to swap blocks to enable further deletions
    if best_solution is not None and time.time() < deadline - 5:
        # Precompute for each block, the set of t-subsets it covers
        # And for each t-subset, the list of blocks covering it (already have blocks_of_subset)
        
        # We will iterate and try to improve
        improved = True
        max_local_steps = 20
        step = 0
        
        while improved and step < max_local_steps and time.time() < deadline - 5:
            improved = False
            step += 1
            
            # Recalculate cover counts
            cover_count = np.zeros(n_t, dtype=np.int32)
            for b_idx in best_solution:
                for s_idx in subsets_of_block[b_idx]:
                    cover_count[s_idx] += 1
            
            # Identify blocks with unique coverage
            # For each block in solution, find t-subsets covered only by it
            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 5:
                    break
                
                unique_subs = []
                for s_idx in subsets_of_block[b_idx]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    # This block is redundant? 
                    # If so, deletion pass should have removed it. 
                    # But maybe it became redundant due to a previous swap in this iteration?
                    # Remove it.
                    for s_idx in subsets_of_block[b_idx]:
                        cover_count[s_idx] -= 1
                    best_solution.pop(i)
                    best_size -= 1
                    improved = True
                    break # Restart loop as indices changed
                
                # Check if we can swap this block
                # The union of points in unique_subs must be contained in any replacement block
                union_points = set()
                for s_idx in unique_subs:
                    union_points.update(t_subsets[s_idx])
                
                if len(union_points) > k:
                    continue # No single block can cover these unique subsets
                
                # Find candidate replacement blocks
                # A candidate block B_j must contain union_points.
                # It must not be the current block.
                # It should ideally cover other uncovered or singly-covered subsets to be "better", 
                # but primarily it must allow the current block to be removed.
                
                # Optimization: Iterate through all k_subsets is expensive (50k).
                # We can filter:
                # We need a block that contains all points in union_points.
                
                # Let's just iterate through the existing solution's neighbors or random samples?
                # No, we should check the global list, but smartly.
                
                # For v=19, k=7, n_k=50388. Iterating 50k blocks for each of ~150 blocks is 7.5M ops.
                # This is fast enough in Python if optimized, or we can use NumPy.
                
                # Let's use a simpler check: 
                # If we remove b_idx, the subsets in unique_subs become uncovered.
                # We need a block B_j that covers ALL of unique_subs.
                
                # We can precompute for each subset of points (of size <= k) the list of blocks containing them?
                # That's too much memory.
                
                # Alternative: Iterate through blocks_of_subset for the first unique subset, 
                # and check if they contain the other unique subsets.
                
                candidate_blocks = set(blocks_of_subset[unique_subs[0]])
                candidate_blocks.discard(b_idx)
                
                # If unique_subs is large, the intersection of blocks_of_subset for all of them is small.
                # We can intersect the sets.
                for s_idx in unique_subs[1:]:
                    candidate_blocks.intersection_update(blocks_of_subset[s_idx])
                    if not candidate_blocks:
                        break
                
                if not candidate_blocks:
                    continue
                
                # Try swapping with the first valid candidate
                # We want a candidate that covers *more* stuff or allows *more* deletions.
                # Let's just try one and see if it helps.
                # To be deterministic and fast, pick the one with highest "potential" (e.g., covers most total subsets).
                
                best_candidate = -1
                best_score = -1
                
                # Limit candidates to check for speed
                cand_list = list(candidate_blocks)
                random.shuffle(cand_list)
                for cand_idx in cand_list[:5]: # Check up to 5 candidates
                    # Simulate swap
                    # Remove b_idx, Add cand_idx
                    # Check if solution remains valid (it should, by construction of unique_subs)
                    # Then run deletion on the new solution? That's expensive.
                    
                    # Instead, just check if the swap reduces the "potential" for deletion.
                    # Actually, let's just do the swap and let the outer loop's deletion pass handle the rest.
                    # But we need to ensure we don't make things worse.
                    
                    # Heuristic: Prefer candidates that cover many other subsets.
                    score = len(subsets_of_block[cand_idx])
                    if score > best_score:
                        best_score = score
                        best_candidate = cand_idx
                
                if best_candidate != -1:
                    # Perform swap
                    best_solution[i] = best_candidate
                    improved = True
                    break # Restart loop to recalculate cover counts
        
    # Final Deletion Pass
    if best_solution is not None and time.time() < deadline - 2:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed and time.time() < deadline - 2:
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

    return [list(k_subsets[idx]) for idx in best_solution]
