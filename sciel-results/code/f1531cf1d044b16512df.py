import numpy as np
import itertools
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    # 1. Precompute t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # 2. Generate a pool of candidate blocks
    # We will run a fast greedy multiple times to collect blocks
    pool = set()
    
    def fast_greedy(seed):
        random.seed(seed)
        uncovered = [True] * n_t_subs
        blocks = []
        
        while any(uncovered):
            # Find first uncovered
            # To speed up, we can maintain a list of uncovered indices
            # But for simplicity in pool gen, we just scan
            # Optimization: Keep a set of uncovered indices
            # For now, let's just pick a random uncovered index
            uncov_indices = [i for i, u in enumerate(uncovered) if u]
            if not uncov_indices:
                break
            idx = random.choice(uncov_indices)
            seed_sub = all_t_subs[idx]
            
            # Build block
            block = list(seed_sub)
            block_set = set(block)
            
            # Greedily add points
            # To be fast, we don't calculate exact gain for all points.
            # We just add points that are in many uncovered t-subsets with the current block.
            # Simple heuristic: pick random points until k, then check coverage.
            # Better: pick points that appear in many uncovered t-subsets.
            
            # Let's stick to a simple randomized completion
            remaining_points = [x for x in range(v) if x not in block_set]
            random.shuffle(remaining_points)
            
            for x in remaining_points:
                if len(block) >= k:
                    break
                # Check if adding x helps
                # Count how many uncovered t-subsets S are such that S \ {x} subset block
                # This is expensive.
                # Let's just add it.
                block.append(x)
                block_set.add(x)
            
            if len(block) < k:
                 # Fill with random
                 for x in remaining_points:
                     if x not in block_set:
                         block.append(x)
                         block_set.add(x)
                         if len(block) == k: break
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered[sub_to_idx[sub]] = False
            
            if time.time() - start_time > time_limit * 0.5: # Stop generating pool early
                break
        
        return blocks

    # Generate pool from multiple seeds
    num_generations = 20
    for i in range(num_generations):
        if time.time() - start_time > time_limit * 0.6:
            break
        blocks = fast_greedy(i)
        for b in blocks:
            pool.add(b)
            
    pool = list(pool)
    
    if not pool:
        # Fallback to single greedy
        blocks = fast_greedy(0)
        return [list(b) for b in blocks]
        
    # 3. Formulate ILP
    # Variables: x_j for each block j in pool
    n_blocks = len(pool)
    
    # Objective: minimize sum x_j
    c = np.ones(n_blocks)
    
    # Constraints: for each t-subset i, sum_{j: i in block_j} x_j >= 1
    # A[i, j] = 1 if t-subset i is in block j
    
    # Build A matrix
    # This can be memory intensive if pool is large.
    # n_t_subs = 11628, n_blocks maybe ~1000. 11628 * 1000 = 11M entries.
    # Using sparse matrix is better.
    
    from scipy.sparse import csr_matrix
    
    rows = []
    cols = []
    data = []
    
    for j, block in enumerate(pool):
        for sub in itertools.combinations(block, t):
            i = sub_to_idx[sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = csr_matrix((data, (rows, cols)), shape=(n_t_subs, n_blocks))
    
    # Constraint: A x >= 1
    # milp expects A_ub x <= b_ub and A_eq x = b_eq
    # So -A x <= -1
    A_ub = -A
    b_ub = -np.ones(n_t_subs)
    
    # Bounds: 0 <= x_j <= 1
    bounds = Bounds(lb=np.zeros(n_blocks), ub=np.ones(n_blocks))
    
    # Integrality: x_j is integer
    integrality = np.ones(n_blocks)
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A_ub, -np.ones(n_t_subs), 0), 
                   integrality=integrality, bounds=bounds, 
                   options={"time_limit": time_limit - (time.time() - start_time) - 5})
        
        if res.status == 0: # Optimal or feasible
            # Extract solution
            selected_indices = np.where(res.x > 0.5)[0]
            selected_blocks = [list(pool[i]) for i in selected_indices]
            
            # Verify
            covered = set()
            for block in selected_blocks:
                for sub in itertools.combinations(block, t):
                    covered.add(tuple(sub))
            if len(covered) == n_t_subs:
                return selected_blocks
            else:
                # Infeasible or error, fallback to best greedy
                pass
        else:
            # If no feasible solution found, fallback
            pass
            
    except Exception as e:
        pass
        
    # Fallback: Use the best greedy result found during pool generation
    # We didn't store the best greedy result separately, so re-run one
    # Or just return the first greedy result
    blocks = fast_greedy(0)
    return [list(b) for b in blocks]
