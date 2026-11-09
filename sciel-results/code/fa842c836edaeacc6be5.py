import itertools
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # 5s safety margin
    
    points = list(range(v))
    
    # Precompute all blocks and t-subsets
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    
    # Map t-subsets to indices
    t_idx = {s: i for i, s in enumerate(all_t)}
    
    # Build incidence: for each block, which t-subsets it covers
    block_tsubs = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(t_idx[s])
        block_tsubs.append(subs)
    
    # Convert to numpy arrays for faster access
    block_tsubs_np = [np.array(subs, dtype=np.int32) for subs in block_tsubs]
    
    # Build reverse index: for each t-subset, which blocks cover it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs_np):
        for si in subs:
            tsub_blocks[si].append(i)
    
    def run_greedy(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        gains = np.array([len(subs) for subs in block_tsubs_np], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.bool_)
        selected_blocks = []
        uncovered_count = N_t
        
        while uncovered_count > 0:
            if time.time() > DEADLINE:
                break
                
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1 and random_seed is not None:
                best_idx = random.choice(candidates.tolist())
            else:
                best_idx = candidates[0]
                
            selected_blocks.append(all_blocks[best_idx])
            
            for si in block_tsubs_np[best_idx]:
                if not is_covered[si]:
                    is_covered[si] = True
                    uncovered_count -= 1
                    for b_idx in tsub_blocks[si]:
                        gains[b_idx] -= 1
                        
        return selected_blocks

    def try_reduce(blocks):
        """Remove redundant blocks (those with no unique t-subsets)."""
        if not blocks:
            return blocks
            
        n = len(blocks)
        cov_count = np.zeros(N_t, dtype=np.int32)
        block_tsubs_list = []
        for b in blocks:
            subs = []
            for s in itertools.combinations(b, t):
                subs.append(t_idx[s])
            block_tsubs_list.append(subs)
            for si in subs:
                cov_count[si] += 1
                
        removed = [False] * n
        changed = True
        while changed:
            if time.time() > DEADLINE:
                break
            changed = False
            for i in range(n):
                if removed[i]:
                    continue
                has_unique = False
                for si in block_tsubs_list[i]:
                    if cov_count[si] == 1:
                        has_unique = True
                        break
                if not has_unique:
                    removed[i] = True
                    changed = True
                    for si in block_tsubs_list[i]:
                        cov_count[si] -= 1
                        
        return [b for i, b in enumerate(blocks) if not removed[i]]

    def try_milp_refinement(blocks, time_budget=30.0):
        """
        Use MILP to find a smaller subset of blocks from the candidate pool
        that still covers all t-subsets.
        """
        if not blocks or time.time() > DEADLINE:
            return blocks
            
        n = len(blocks)
        if n < 2:
            return blocks
        
        # Build the candidate pool: all blocks in 'blocks' plus some additional candidates
        # For efficiency, we'll work with the blocks we already have
        # and try to minimize their selection.
        
        # Build incidence matrix: A[i, j] = 1 if block i covers t-subset j
        # This is sparse. We'll use scipy sparse.
        from scipy.sparse import csr_matrix
        
        # Precompute mapping from block tuple to index in all_blocks
        # Use a set-based lookup for O(1) access
        block_set = set(all_blocks)
        
        row_indices = []
        col_indices = []
        for i, b in enumerate(blocks):
            # Find index in all_blocks
            b_tuple = tuple(b)
            if b_tuple not in block_set:
                continue
            # Use bisect or precomputed dict - for now use a simpler approach
            # Actually, let's just compute the t-subsets directly
            for s in itertools.combinations(b, t):
                si = t_idx[s]
                row_indices.append(si)
                col_indices.append(i)
        
        if not row_indices:
            return blocks
            
        A = csr_matrix((np.ones(len(row_indices), dtype=np.int8), 
                        (row_indices, col_indices)), 
                       shape=(N_t, n))
        
        # Objective: minimize sum of x_i (select as few blocks as possible)
        c = np.ones(n, dtype=np.float64)
        
        # Constraints: for each t-subset j, sum_i A[i,j] * x_i >= 1
        # i.e., A @ x >= 1
        lower_bound = np.ones(N_t, dtype=np.float64)
        upper_bound = np.full(N_t, np.inf, dtype=np.float64)
        
        constraint = LinearConstraint(A, lower_bound, upper_bound)
        
        # x_i in {0, 1}
        integrality = np.ones(n, dtype=np.int8)
        bounds = Bounds(lb=0, ub=1)
        
        try:
            # Run MILP with a timeout by running in a thread
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def solve_milp():
                try:
                    res = milp(
                        c=c,
                        constraints=constraint,
                        integrality=integrality,
                        bounds=bounds
                    )
                    result_queue.put(res)
                except Exception as e:
                    result_queue.put(e)
            
            solver_thread = threading.Thread(target=solve_milp)
            solver_thread.start()
            solver_thread.join(timeout=time_budget)
            
            if solver_thread.is_alive():
                # Thread didn't finish in time; we can't easily kill it
                # Just return the original blocks
                return blocks
                
            res = result_queue.get()
            if isinstance(res, Exception):
                return blocks
            if not res.success:
                return blocks
            
            # Extract solution
            x = res.x
            if x is None:
                return blocks
                
            selected = [blocks[i] for i in range(n) if x[i] > 0.5]
            
            # Verify the solution is valid
            if len(selected) == 0:
                return blocks
                
            # Check coverage
            covered = set()
            for b in selected:
                for s in itertools.combinations(b, t):
                    covered.add(s)
                    
            if len(covered) == N_t:
                return selected
            else:
                return blocks
                
        except Exception:
            return blocks

    def local_search(blocks, time_budget=20.0):
        """
        Local search: try to remove blocks or swap pairs to reduce count.
        """
        if not blocks or time.time() > DEADLINE:
            return blocks
            
        blocks = list(blocks)
        n = len(blocks)
        
        # Build coverage counts
        cov_count = np.zeros(N_t, dtype=np.int32)
        block_subs = []
        for b in blocks:
            subs = []
            for s in itertools.combinations(b, t):
                si = t_idx[s]
                subs.append(si)
                cov_count[si] += 1
            block_subs.append(subs)
        
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            
            # Try removing each block
            for i in range(len(blocks)):
                if time.time() > DEADLINE:
                    return blocks
                
                # Check if block i has any unique coverage
                has_unique = False
                for si in block_subs[i]:
                    if cov_count[si] == 1:
                        has_unique = True
                        break
                
                if not has_unique:
                    # Remove block i
                    for si in block_subs[i]:
                        cov_count[si] -= 1
                    blocks.pop(i)
                    block_subs.pop(i)
                    improved = True
                    break  # Restart scan after removal
        
        # Try 1-for-1 swaps: remove block i, add block j not in set
        # This is expensive, so we'll do a limited version
        if time.time() < DEADLINE - 5.0:
            current_n = len(blocks)
            
            # Build a set of current blocks for fast lookup
            current_block_set = set(tuple(b) for b in blocks)
            
            # For each block, try to find a replacement that allows removal
            # We'll try: for each pair (i, j) where j is not in set,
            # check if removing i and adding j still covers everything
            # This is O(n * N_blocks) which is too slow, so we'll be selective
            
            # Instead, try a simpler approach: for each block i,
            # find t-subsets it uniquely covers, and look for blocks
            # that cover those t-subsets
            for i in range(len(blocks)):
                if time.time() > DEADLINE:
                    return blocks
                
                # Find t-subsets uniquely covered by block i
                unique_subs = []
                for si in block_subs[i]:
                    if cov_count[si] == 1:
                        unique_subs.append(si)
                
                if not unique_subs:
                    # Already handled by removal
                    continue
                
                # Look for a block that covers all unique_subs
                # and is not already in the set
                # This is expensive, so we'll skip for now
                pass
        
        return blocks

    best_solution = []
    best_size = float('inf')
    
    # Run multiple greedy restarts
    for i in range(15):
        if time.time() > DEADLINE:
            break
            
        if i == 0:
            sol = run_greedy(random_seed=None)
        else:
            sol = run_greedy(random_seed=i)
            
        if not sol:
            continue
            
        sol = try_reduce(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
    
    # If we have time, try local search then MILP refinement on the best solution
    if best_solution and time.time() < DEADLINE - 20.0:
        # First try local search (faster)
        time_budget_ls = min(15.0, DEADLINE - time.time() - 10.0)
        if time_budget_ls > 3.0:
            ls_refined = local_search(best_solution, time_budget=time_budget_ls)
            if ls_refined and len(ls_refined) < best_size:
                best_solution = ls_refined
                best_size = len(ls_refined)
    
    # Then try MILP refinement if time remains
    if best_solution and time.time() < DEADLINE - 15.0:
        time_budget = min(20.0, DEADLINE - time.time() - 5.0)
        if time_budget > 5.0:
            refined = try_milp_refinement(best_solution, time_budget=time_budget)
            if refined and len(refined) < best_size:
                best_solution = refined
                best_size = len(refined)
    
    # Final reduction pass
    if best_solution and time.time() < DEADLINE - 3.0:
        best_solution = try_reduce(best_solution)
        best_size = len(best_solution)
    
    return best_solution
