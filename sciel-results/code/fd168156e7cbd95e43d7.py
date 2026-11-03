import itertools
import numpy as np
import time
import random
import heapq
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets and k-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Map each t-subset to the list of k-blocks that contain it
    blocks_of_subset = [[] for _ in range(n_t)]
    # Map each k-block to the list of t-subsets it contains
    subsets_of_block = [[] for _ in range(n_k)]
    
    # Build adjacency
    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            blocks_of_subset[i].append(j)
            subsets_of_block[j].append(i)
            
    # Convert to numpy arrays for faster operations where possible, 
    # but lists are often faster for small sizes in Python if we avoid overhead.
    # However, for intersection logic, we might need sets or sorted lists.
    # Let's keep them as lists for now, but maybe convert blocks_of_subset to sets for intersection?
    # No, lists are better for iteration. We'll use a helper for intersection.
    
    best_solution = None
    best_size = float('inf')
    
    # 2. LP Relaxation for initial weights (optional but helpful)
    duals = None
    try:
        if time.time() > deadline - 15:
            raise TimeoutError()
            
        # Build sparse matrix M (n_t x n_k)
        rows = []
        cols = []
        for j in range(n_k):
            for i in subsets_of_block[j]:
                rows.append(i)
                cols.append(j)
        
        M = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_t, n_k))
        
        c = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds = [(0, 1)] * n_k
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                     options={'presolve': True, 'time_limit': 8})
        
        if res.success and res.ineqlin is not None:
            duals = np.abs(res.ineqlin.marginals)
            if np.sum(duals) < 1e-10:
                duals = None
    except Exception:
        duals = None
        
    if duals is None:
        # Fallback weights: uniform or based on frequency
        duals = np.ones(n_t)
        
    # 3. Greedy Construction with Restarts
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Initialize weights
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals[subsets_of_block[j]])
                
        # Perturbation
        if restart > 0:
            # Add small random noise to break ties and explore different paths
            noise = np.random.uniform(0, 0.01 * np.max(block_weights), n_k)
            block_weights += noise
            
        # Greedy Selection
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        # Use a heap for efficient max-weight selection
        # We need to update weights dynamically. 
        # Instead of a heap with lazy deletion which is complex to update correctly for "weight reduction",
        # we can use a simple list scan if n_k is small enough, or a heap with re-insertion.
        # n_k for C(19,7) is 50388. A heap is better.
        
        heap = [(-block_weights[j], j) for j in range(n_k) if block_weights[j] > 1e-12]
        heapq.heapify(heap)
        
        while heap:
            if time.time() > deadline - 5:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
            if w > block_weights[j] + 1e-9:
                continue
            if w <= 1e-12:
                break
                
            # Select block j
            picked[j] = True
            solution_blocks.append(j)
            
            # Update weights of other blocks
            for s_idx in subsets_of_block[j]:
                if not covered[s_idx]:
                    covered[s_idx] = True
                    y_s = duals[s_idx]
                    if y_s > 1e-15:
                        for b_prime in blocks_of_subset[s_idx]:
                            if not picked[b_prime]:
                                block_weights[b_prime] -= y_s
                                if block_weights[b_prime] > 1e-12:
                                    heapq.heappush(heap, (-block_weights[b_prime], b_prime))
                                
            if np.all(covered):
                break
                
        if np.all(covered) and len(solution_blocks) < best_size:
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
    # Fallback if no solution found
    if best_solution is None:
        best_solution = []
        covered = np.zeros(n_t, dtype=bool)
        while not np.all(covered):
            best_b = -1
            max_new = -1
            for j in range(n_k):
                if not picked[j]: # Note: 'picked' from last restart is not valid here, use a new check
                    new_cov = sum(1 for s in subsets_of_block[j] if not covered[s])
                    if new_cov > max_new:
                        max_new = new_cov
                        best_b = j
            if best_b == -1:
                break
            best_solution.append(best_b)
            for s in subsets_of_block[best_b]:
                covered[s] = True
        best_size = len(best_solution)
        
    # 4. Local Search: Deletion and 2-Swap
    if best_solution:
        # Helper to calculate cover counts
        def calc_cover_counts(sol):
            counts = np.zeros(n_t, dtype=np.int32)
            for b_idx in sol:
                counts[subsets_of_block[b_idx]] += 1
            return counts
        
        # Deletion Pass
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            cover_count = calc_cover_counts(best_solution)
            new_sol = []
            for b_idx in best_solution:
                subs = subsets_of_block[b_idx]
                # If all subsets in this block are covered by at least one other block, we can remove it
                if np.all(cover_count[subs] > 1):
                    cover_count[subs] -= 1
                    changed = True
                else:
                    new_sol.append(b_idx)
            best_solution = new_sol
            best_size = len(best_solution)
            
        # 2-Swap Pass
        # For each pair (i, j), check if we can replace them with a single block k
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            cover_count = calc_cover_counts(best_solution)
            sol_set = set(best_solution)
            
            # To speed up, we can iterate over pairs. 
            # n_solution is around 150. Pairs ~ 11000. 
            # For each pair, we need to find if there's a block covering the "critical" subsets.
            
            for i in range(len(best_solution)):
                if time.time() > deadline - 5:
                    break
                b1 = best_solution[i]
                for j in range(i + 1, len(best_solution)):
                    if time.time() > deadline - 5:
                        break
                    b2 = best_solution[j]
                    
                    # Determine subsets that become uncovered if we remove b1 and b2
                    # These are subsets s where cover_count[s] == 2 and s is in b1 or b2
                    # AND s is in b1 XOR b2? No.
                    # If s is in both b1 and b2, and count is 2, removing both makes count 0.
                    # If s is in b1 only, and count is 1 (impossible if b2 also covers it? No, b2 doesn't cover it), 
                    # wait. 
                    
                    # Let's calculate the set of subsets that are covered ONLY by b1 and b2 (among the current solution).
                    # This is: (Subsets(b1) U Subsets(b2)) such that count(s) == 2 in current solution?
                    # No, if count(s) > 2, removing b1 and b2 still leaves count(s) >= 1.
                    # So we only care about subsets s where count(s) == 2 and s is in b1 or b2.
                    # Actually, if s is in b1 and b2, and count(s)=2, then removing both leaves 0.
                    # If s is in b1 but not b2, and count(s)=1, then removing b1 leaves 0. But count(s)=1 implies only b1 covers it.
                    # So we need to find all s such that count(s) - (1 if s in b1 else 0) - (1 if s in b2 else 0) == 0.
                    
                    critical_subsets = []
                    # Iterate through subsets of b1
                    for s in subsets_of_block[b1]:
                        if cover_count[s] == 2 and s in subsets_of_block[b2]:
                            # If s is in both, and count is 2, it becomes 0
                            critical_subsets.append(s)
                        elif cover_count[s] == 1:
                            # If s is only in b1 (and not b2, because if it was in b2 count would be at least 2? No, count is total. 
                            # If s is in b1 and b2, count includes both. 
                            # If cover_count[s] == 1, then s is in exactly one block in the solution.
                            # If s is in b1, it cannot be in b2 (else count >= 2).
                            # So if s in b1 and count==1, removing b1 makes it 0.
                            critical_subsets.append(s)
                            
                    # We also need to check subsets in b2 that are not in b1
                    # But we can just check all subsets in b2 as well to be safe, or optimize.
                    # Let's just collect all s in b1 U b2 where new_count == 0.
                    # new_count(s) = cover_count[s] - (s in b1) - (s in b2)
                    
                    # More efficient:
                    # Create a set of critical subsets
                    critical_set = set()
                    for s in subsets_of_block[b1]:
                        new_c = cover_count[s] - 1
                        if new_c == 0:
                            critical_set.add(s)
                        elif new_c == 1 and s in subsets_of_block[b2]:
                            # If removing b1 leaves 1, but b2 also covers it, then removing b2 leaves 0
                            critical_set.add(s)
                            
                    for s in subsets_of_block[b2]:
                        # Only need to check if s is NOT in b1, or if it was already handled
                        # Actually, the logic above handles s in b1. 
                        # If s is in b2 but not b1:
                        if s not in subsets_of_block[b1]:
                             if cover_count[s] == 1:
                                 critical_set.add(s)
                        else:
                            # s is in both. 
                            # If cover_count[s] == 2, then removing both leaves 0.
                            # This was handled in the first loop? 
                            # In first loop: new_c = 2-1=1. Then check if s in b2. Yes. So it was added.
                            # So we don't need to add it again.
                            pass

                    if not critical_set:
                        # We can remove both b1 and b2
                        new_sol = [b for idx, b in enumerate(best_solution) if idx not in (i, j)]
                        best_solution = new_sol
                        best_size = len(best_solution)
                        improved = True
                        break # Restart loop as solution changed
                    
                    # Check if there is a block k that covers all critical subsets
                    # Find intersection of blocks_of_subset[s] for s in critical_set
                    # Start with the smallest list to speed up intersection
                    lists = [blocks_of_subset[s] for s in critical_set]
                    if not lists:
                        continue
                    lists.sort(key=len)
                    candidates = set(lists[0])
                    for lst in lists[1:]:
                        candidates &= set(lst)
                        if not candidates:
                            break
                    
                    # Filter out blocks already in the solution
                    candidates -= sol_set
                    
                    if candidates:
                        # Pick the first candidate (or one with some heuristic)
                        k = next(iter(candidates))
                        # Replace b1, b2 with k
                        new_sol = [b for idx, b in enumerate(best_solution) if idx not in (i, j)]
                        new_sol.append(k)
                        best_solution = new_sol
                        best_size = len(best_solution)
                        sol_set = set(best_solution)
                        improved = True
                        break # Restart loop as solution changed
                        
                if improved:
                    break
            if improved:
                # Re-run deletion after swap
                changed = True
                while changed and time.time() < deadline - 5:
                    changed = False
                    cover_count = calc_cover_counts(best_solution)
                    new_sol = []
                    for b_idx in best_solution:
                        subs = subsets_of_block[b_idx]
                        if np.all(cover_count[subs] > 1):
                            cover_count[subs] -= 1
                            changed = True
                        else:
                            new_sol.append(b_idx)
                    best_solution = new_sol
                    best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
