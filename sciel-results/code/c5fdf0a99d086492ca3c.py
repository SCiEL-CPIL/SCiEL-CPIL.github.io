import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
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
    
    # 2. Primal-dual approximation algorithm with multiple restarts
    # This exploits submodularity of coverage function
    num_pd_restarts = 8
    
    for pd_restart in range(num_pd_restarts):
        if time.time() > deadline - 20:
            break
        
        # Initialize dual variables with random perturbation for diversity
        if pd_restart == 0:
            y = np.ones(n_t)
        else:
            y = np.random.uniform(0.5, 1.5, n_t)
        
        # Primal-dual algorithm:
        # 1. Start with all dual variables at 0
        # 2. Increase duals for uncovered subsets uniformly
        # 3. When a block becomes "tight" (sum of duals of its subsets >= cost), select it
        # 4. Mark all subsets in selected block as covered
        # 5. Repeat until all subsets covered
        
        y = np.zeros(n_t)
        covered = np.zeros(n_t, dtype=bool)
        selected_blocks = []
        block_cost = np.ones(n_k)  # Each block costs 1
        
        # Track which subsets are still uncovered
        uncovered_mask = np.ones(n_t, dtype=bool)
        n_uncovered = n_t
        
        # For efficiency, maintain a priority queue of blocks by their "tightness"
        # A block is tight when sum_{s in block} y[s] >= cost[block]
        
        # We'll use an iterative approach:
        # - Find the minimum delta to increase duals until some block becomes tight
        # - Select that block, update coverage
        # - Repeat
        
        while n_uncovered > 0 and time.time() < deadline - 15:
            # Compute current tightness for each block
            # tightness[j] = sum_{s in subsets_of_block[j]} y[s] - cost[j]
            # We want the block with minimum slack (cost - tightness_sum)
            
            # For speed, only consider blocks that cover at least one uncovered subset
            # and haven't been selected yet
            
            best_block = -1
            min_slack = float('inf')
            
            # Sample-based approach for speed: check a random subset of blocks
            if n_k > 5000:
                sample_size = min(5000, n_k)
                indices = np.random.choice(n_k, size=sample_size, replace=False)
            else:
                indices = range(n_k)
            
            for j in indices:
                if j in selected_blocks:
                    continue
                
                # Quick check: does this block cover any uncovered subset?
                has_uncovered = False
                for s_idx in subsets_of_block[j]:
                    if uncovered_mask[s_idx]:
                        has_uncovered = True
                        break
                
                if not has_uncovered:
                    continue
                
                # Compute slack = cost - sum of duals
                dual_sum = sum(y[s_idx] for s_idx in subsets_of_block[j])
                slack = block_cost[j] - dual_sum
                
                if slack < min_slack:
                    min_slack = slack
                    best_block = j
            
            if best_block == -1:
                # No valid block found; increase all duals slightly
                y += 0.01
                continue
            
            # Increase duals for uncovered subsets until best_block becomes tight
            if min_slack > 0:
                # Need to increase duals by min_slack distributed among uncovered subsets in best_block
                # For simplicity, increase all uncovered subsets uniformly
                # But we need to be careful not to overshoot for other blocks
                
                # Find the delta that makes best_block tight
                # delta = min_slack / (number of uncovered subsets in best_block)
                n_unc_in_block = sum(1 for s_idx in subsets_of_block[best_block] if uncovered_mask[s_idx])
                if n_unc_in_block > 0:
                    delta = min_slack / n_unc_in_block
                    # But we also need to check that no other block becomes tight earlier
                    # For now, use a conservative step
                    delta = min(delta, 0.1)
                    
                    for s_idx in range(n_t):
                        if uncovered_mask[s_idx]:
                            y[s_idx] += delta
            
            # Select the block
            selected_blocks.append(best_block)
            
            # Update coverage
            for s_idx in subsets_of_block[best_block]:
                if uncovered_mask[s_idx]:
                    uncovered_mask[s_idx] = False
                    n_uncovered -= 1
            
            # If we've covered everything, break
            if n_uncovered == 0:
                break
        
        if n_uncovered == 0 and len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_solution = selected_blocks.copy()
    
    # 3. If primal-dual didn't find a good solution, use LP-based greedy as fallback
    if best_solution is None or best_size > 200:
        # LP Relaxation for duals
        duals = None
        try:
            if time.time() > deadline - 30:
                raise TimeoutError()
                
            M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
            
            best_duals = None
            best_lp_obj = float('inf')
            
            for attempt in range(3):
                if time.time() > deadline - 30:
                    break
                c = np.ones(n_k)
                if attempt > 0:
                    c += np.random.uniform(0, 0.1, n_k)
                A_ub = -M.T.tocsr()
                b_ub = -np.ones(n_t)
                bounds = [(0, 1)] * n_k
                
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                
                if res.success:
                    duals_candidate = np.abs(res.ineqlin.marginals)
                    if np.sum(duals_candidate) > 0:
                        if best_duals is None or res.fun < best_lp_obj:
                            best_duals = duals_candidate
                            best_lp_obj = res.fun
        except Exception:
            best_duals = None
        
        if best_duals is not None:
            duals = best_duals
        else:
            duals = np.ones(n_t)
        
        # Multi-restart heap-based greedy with LP duals
        num_restarts = 6
        
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
                
            duals_variant = duals.copy()
            if restart % 3 == 1:
                for s_idx in range(n_t):
                    if len(blocks_of_subset[s_idx]) < 10:
                        duals_variant[s_idx] *= 2.0
            elif restart % 3 == 2:
                duals_variant += np.random.uniform(0, 0.05, n_t)
                
            block_weights = np.zeros(n_k)
            for j in range(n_k):
                if subsets_of_block[j]:
                    block_weights[j] = np.sum(duals_variant[subsets_of_block[j]])
                    
            if restart > 0:
                block_weights += np.random.uniform(0, 0.005 + 0.002 * restart, n_k)
                
            heap = [(-block_weights[j], j) for j in range(n_k)]
            heapq.heapify(heap)
            
            covered = np.zeros(n_t, dtype=bool)
            picked = np.zeros(n_k, dtype=bool)
            solution_blocks = []
            
            while heap:
                if time.time() > deadline - 15:
                    break
                    
                neg_w, j = heapq.heappop(heap)
                w = -neg_w
                
                if picked[j]:
                    continue
                if w > block_weights[j] + 1e-9:
                    continue
                if w <= 1e-9:
                    break
                    
                picked[j] = True
                solution_blocks.append(j)
                
                for s_idx in subsets_of_block[j]:
                    if not covered[s_idx]:
                        covered[s_idx] = True
                        y_s = duals_variant[s_idx]
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
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = sum(1 for s in subsets_of_block[j] if s in uncovered)
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
        best_size = len(best_solution)
    
    # 2. LP Relaxation for duals - solve multiple times with different objectives
    duals = None
    try:
        if time.time() > deadline - 5:
            raise TimeoutError()
            
        M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
        
        # Try multiple LP formulations to get better duals
        best_duals = None
        best_lp_obj = float('inf')
        
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            c = np.ones(n_k)
            if attempt > 0:
                c += np.random.uniform(0, 0.1, n_k)
            A_ub = -M.T.tocsr()
            b_ub = -np.ones(n_t)
            bounds = [(0, 1)] * n_k
            
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if res.success:
                duals_candidate = np.abs(res.ineqlin.marginals)
                if np.sum(duals_candidate) > 0:
                    if best_duals is None or res.fun < best_lp_obj:
                        best_duals = duals_candidate
                        best_lp_obj = res.fun
    except Exception:
        best_duals = None
    
    if best_duals is not None:
        duals = best_duals
    else:
        duals = np.ones(n_t)
    
    # 3. Multi-restart heap-based greedy with LP duals
    # Use different strategies for different restarts
    num_restarts = 12
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Vary the dual weights for diversity
        duals_variant = duals.copy()
        if restart % 3 == 1:
            # Emphasize rare subsets (those in fewer blocks)
            for s_idx in range(n_t):
                if len(blocks_of_subset[s_idx]) < 10:
                    duals_variant[s_idx] *= 2.0
        elif restart % 3 == 2:
            # Add random perturbation
            duals_variant += np.random.uniform(0, 0.05, n_t)
            
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals_variant[subsets_of_block[j]])
                
        if restart > 0:
            # Scale perturbation with restart number for more diversity
            block_weights += np.random.uniform(0, 0.005 + 0.002 * restart, n_k)
            
        heap = [(-block_weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        while heap:
            if time.time() > deadline - 10:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-9:
                break
                
            picked[j] = True
            solution_blocks.append(j)
            
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals_variant[s_idx]
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
        uncovered = set(range(n_t))
        res_blocks = []
        while uncovered:
            best_b = -1
            max_cov = -1
            for j in range(n_k):
                cnt = sum(1 for s in subsets_of_block[j] if s in uncovered)
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
        best_size = len(best_solution)
    
    # 4. Deletion Pass
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
    
    # 5. Advanced Local Search: 1-for-1, 2-for-1, and 2-for-2 swaps
    if best_solution is not None and time.time() < deadline - 8:
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
        
        # Build index: for each subset, list of blocks containing it (for fast lookup)
        # Already have blocks_of_subset
        
        improved = True
        max_iters = 20
        iter_count = 0
        while improved and time.time() < deadline - 8 and iter_count < max_iters:
            improved = False
            iter_count += 1
            new_solution = best_solution.copy()
            solution_set = set(best_solution)
            
            # Try 1-for-1 replacement: remove block i, find if another block can cover all its unique subsets
            for i, b_idx in enumerate(best_solution):
                if time.time() > deadline - 8:
                    break
                subs_b = subsets_of_block[b_idx]
                
                # Find subsets that would become uncovered if we remove b_idx
                unique_subs = [s for s in subs_b if cover_count[s] == 1]
                
                if not unique_subs:
                    # Can remove entirely
                    cover_count[subs_b] -= 1
                    new_solution.pop(i)
                    solution_set.discard(b_idx)
                    improved = True
                    # Recompute cover counts
                    cover_count = np.zeros(n_t, dtype=np.int32)
                    for b_idx2 in new_solution:
                        for s_idx in subsets_of_block[b_idx2]:
                            cover_count[s_idx] += 1
                    break
                else:
                    # Try to find a replacement block that covers all unique_subs
                    # Use the most constrained subset to guide search
                    # Sort unique_subs by number of covering blocks (ascending)
                    unique_subs_sorted = sorted(unique_subs, key=lambda s: len(blocks_of_subset[s]))
                    
                    best_replacement = -1
                    for b_new in range(n_k):
                        if b_new in solution_set:
                            continue
                        # Check if b_new covers all unique_subs
                        # Quick check: does b_new contain the most constrained subset?
                        if unique_subs_sorted:
                            most_constrained = unique_subs_sorted[0]
                            if most_constrained not in subsets_of_block[b_new]:
                                continue
                        new_subs_set = set(subsets_of_block[b_new])
                        if all(s in new_subs_set for s in unique_subs):
                            best_replacement = b_new
                            break
                    
                    if best_replacement != -1:
                        # Remove b_idx, add best_replacement
                        cover_count[subs_b] -= 1
                        new_solution[i] = best_replacement
                        solution_set.discard(b_idx)
                        solution_set.add(best_replacement)
                        for s_idx in subsets_of_block[best_replacement]:
                            cover_count[s_idx] += 1
                        improved = True
                        break
            
            # If no 1-for-1 found, try 2-for-1: remove two blocks, add one that covers all their unique subsets
            if not improved and len(best_solution) >= 2:
                for i in range(len(best_solution)):
                    if time.time() > deadline - 8:
                        break
                    b_i = best_solution[i]
                    subs_i = subsets_of_block[b_i]
                    unique_i = [s for s in subs_i if cover_count[s] == 1]
                    
                    for j in range(i + 1, len(best_solution)):
                        b_j = best_solution[j]
                        subs_j = subsets_of_block[b_j]
                        unique_j = [s for s in subs_j if cover_count[s] == 1]
                        
                        # Combined unique subsets
                        combined_unique = set(unique_i) | set(unique_j)
                        if not combined_unique:
                            continue
                        
                        # Find a block that covers all combined_unique
                        best_replacement = -1
                        for b_new in range(n_k):
                            if b_new in solution_set:
                                continue
                            new_subs_set = set(subsets_of_block[b_new])
                            if all(s in new_subs_set for s in combined_unique):
                                best_replacement = b_new
                                break
                        
                        if best_replacement != -1:
                            # Remove b_i and b_j, add best_replacement
                            cover_count[subs_i] -= 1
                            cover_count[subs_j] -= 1
                            # Remove both from new_solution
                            new_solution.pop(j)  # Remove j first (higher index)
                            new_solution.pop(i)  # Now remove i
                            solution_set.discard(b_i)
                            solution_set.discard(b_j)
                            new_solution.append(best_replacement)
                            solution_set.add(best_replacement)
                            for s_idx in subsets_of_block[best_replacement]:
                                cover_count[s_idx] += 1
                            improved = True
                            break
                    
                    if improved:
                        break
            
            if improved:
                best_solution = new_solution
                best_size = len(best_solution)
    
    # 6. Final deletion pass after swaps - multiple aggressive passes
    if best_solution is not None and time.time() < deadline - 2:
        for pass_num in range(5):
            if time.time() > deadline - 2:
                break
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
