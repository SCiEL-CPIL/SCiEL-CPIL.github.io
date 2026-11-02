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
            
    # Convert lists to numpy arrays for faster access if needed, but lists are fine for iteration
    # subsets_of_block = [np.array(subs, dtype=np.int32) for subs in subsets_of_block]
    
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
            # res.ineqlin.marginals are the dual variables for the constraints
            # The constraint is -M.T @ x <= -1
            # Dual y >= 0 for the original >= 1 constraint.
            # The marginal for <= constraint is usually negative or zero in minimization?
            # Let's check signs.
            # Standard form: min c^T x s.t. A_ub x <= b_ub.
            # Dual: max b_ub^T y s.t. A_ub^T y >= c.
            # Here A_ub = -M.T, b_ub = -1.
            # Dual: max -1^T y s.t. -M y >= 1  => M y <= -1.
            # This seems tricky with signs.
            # SciPy linprog marginals for A_ub constraints (<=) are the dual values.
            # For a minimization problem, if the constraint is active, the marginal is the change in objective.
            # Let's just take absolute values or check if they are positive.
            # Usually, for set cover, duals are positive.
            duals = res.ineqlin.marginals
            # Ensure they are non-negative
            duals = np.abs(duals)
            
            # If duals are all zero, it's useless
            if np.sum(duals) == 0:
                duals = None
        else:
            duals = None
            
    except Exception:
        duals = None
        
    # 3. Dynamic Greedy
    # If LP failed, use uniform weights (standard greedy)
    if duals is None:
        duals = np.ones(n_t)
        
    # Run a few iterations with slight noise if time permits, or just one strong one
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize weights
        # Weight of block j = sum of duals of subsets in block j
        block_weights = np.zeros(n_k)
        for j in range(n_k):
            if subsets_of_block[j]:
                block_weights[j] = np.sum(duals[subsets_of_block[j]])
                
        # Add noise for diversification in restarts > 0
        if restart > 0:
            block_weights += np.random.uniform(0, 0.01, n_k)
            
        # Heap of (-weight, index)
        heap = [(-block_weights[j], j) for j in range(n_k)]
        heapq.heapify(heap)
        
        covered = np.zeros(n_t, dtype=bool)
        picked = np.zeros(n_k, dtype=bool)
        solution_blocks = []
        
        while heap:
            if time.time() > deadline - 2:
                break
                
            neg_w, j = heapq.heappop(heap)
            w = -neg_w
            
            if picked[j]:
                continue
                
            # Check if weight is still valid (lazy deletion)
            # Since weights only decrease, if w > block_weights[j] + eps, it's stale
            if w > block_weights[j] + 1e-9:
                continue
                
            if w <= 1e-9:
                # No more gain
                break
                
            # Select block j
            picked[j] = True
            solution_blocks.append(j)
            
            # Update weights for blocks sharing subsets with j
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
                
        # Check if valid
        if len(solution_blocks) < best_size and np.all(covered):
            best_size = len(solution_blocks)
            best_solution = solution_blocks.copy()
            
        # Early stop if we hit a very good number?
        # 151 is the target. If we get 151, we can keep searching for 150.
        
    # Fallback if no valid solution found (should not happen)
    if best_solution is None:
        # Simple greedy fallback
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
        best_size = len(best_solution)
        
    # 4. Post-processing: Deletion Pass
    if best_solution is not None:
        # Compute cover counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            for s_idx in subsets_of_block[b_idx]:
                cover_count[s_idx] += 1
                
        # Iterative deletion
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
            
    # 5. Optional: 2-for-1 Swap Search (if time permits)
    # This is more complex and might not be worth it if deletion is clean.
    # Given the time budget, we stick to deletion.
    
    # Construct output
    return [list(k_subsets[idx]) for idx in best_solution]
