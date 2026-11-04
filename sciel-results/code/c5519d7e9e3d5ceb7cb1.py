import itertools
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import sys

def construct(v, k, t):
    deadline = time.time() + 140  # 140s safety margin
    
    # Calculate sizes
    num_blocks = int(np.math.comb(v, k))
    num_subsets = int(np.math.comb(v, t))
    
    # If the problem is too large for ILP, fall back to greedy immediately
    # 50,000 variables is usually okay for HiGHS in 100s if sparse.
    # 100,000+ might be risky.
    use_ilp = num_blocks <= 80000 and num_subsets <= 10000
    
    if use_ilp:
        try:
            blocks = _solve_ilp(v, k, t, num_blocks, num_subsets, deadline)
            if blocks is not None:
                return blocks
        except Exception as e:
            # Fall through to greedy if ILP fails
            pass
    
    # Fallback: Optimized Greedy + Local Search
    return _greedy_with_refinement(v, k, t, deadline)

def _solve_ilp(v, k, t, num_blocks, num_subsets, deadline):
    # Precompute all blocks and subsets
    all_blocks = list(itertools.combinations(range(v), k))
    all_subsets = list(itertools.combinations(range(v), t))
    
    # Create mapping from subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Build incidence matrix A (num_blocks x num_subsets)
    # A[i, j] = 1 if subset j is in block i
    rows = []
    cols = []
    data = []
    
    for i, block in enumerate(all_blocks):
        block_set = set(block)
        for sub in itertools.combinations(block, t):
            j = subset_to_idx[sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = csr_matrix((data, (rows, cols)), shape=(num_blocks, num_subsets))
    
    # Objective: minimize sum(x_i)
    c = np.ones(num_blocks)
    
    # Constraints: A @ x >= 1
    constraint = LinearConstraint(A, lb=np.ones(num_subsets), ub=np.inf)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(lb=np.zeros(num_blocks), ub=np.ones(num_blocks))
    
    # Integrality: x_i must be integer
    integrality = np.ones(num_blocks)
    
    # Solve
    res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds, options={'time_limit': 100})
    
    if res.status == 0:  # Optimal
        # Extract solution
        selected_indices = np.where(res.x > 0.5)[0]
        blocks = [list(all_blocks[i]) for i in selected_indices]
        return blocks
    elif res.status == 2:  # Feasible (but maybe not optimal)
        selected_indices = np.where(res.x > 0.5)[0]
        if len(selected_indices) > 0:
            blocks = [list(all_blocks[i]) for i in selected_indices]
            # Verify validity just in case
            if _is_valid_covering(blocks, v, t):
                return blocks
    elif res.status == 3:  # Time limit reached, but might have feasible solution
        if res.x is not None:
            selected_indices = np.where(res.x > 0.5)[0]
            if len(selected_indices) > 0:
                blocks = [list(all_blocks[i]) for i in selected_indices]
                if _is_valid_covering(blocks, v, t):
                    return blocks
                    
    return None

def _is_valid_covering(blocks, v, t):
    covered = set()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            covered.add(sub)
    total_subsets = int(np.math.comb(v, t))
    return len(covered) == total_subsets

def _greedy_with_refinement(v, k, t, deadline):
    # This is the improved greedy from inspiration 1
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # For each point, precompute which t-subsets it participates in
    point_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for p in s:
            point_subsets[p].append(i)
            
    uncovered_set = set(range(n_subsets))
    uncovered_count = n_subsets
    blocks = []
    
    while uncovered_count > 0:
        if time.time() > deadline - 30:
            break
            
        # Find an uncovered subset to seed a new block
        seed_idx = next(iter(uncovered_set))
        seed = list(all_t_subsets[seed_idx])
        
        block = seed[:]
        
        while len(block) < k:
            if time.time() > deadline - 25:
                break
                
            best_point = None
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                    
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    idx = subset_to_idx[sub]
                    if idx in uncovered_set:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                    
            if best_point is None or best_gain <= 0:
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
                        
            if best_point is not None:
                block.append(best_point)
            else:
                break
        
        if len(block) < k:
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block:
                    block.append(x)
                    
        block.sort()
        blocks.append(block)
        
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            if idx in uncovered_set:
                uncovered_set.discard(idx)
                uncovered_count -= 1
                
        if uncovered_count == 0:
            break
            
    # Emergency fill if not complete
    if uncovered_count > 0:
        while uncovered_count > 0 and time.time() < deadline - 5:
            seed_idx = next(iter(uncovered_set))
            seed = list(all_t_subsets[seed_idx])
            block = seed[:]
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if idx in uncovered_set:
                    uncovered_set.discard(idx)
                    uncovered_count -= 1
                    
    # Local search: remove redundant blocks
    if time.time() < deadline - 10:
        blocks = _local_search_remove(blocks, v, t, deadline)
        
    return blocks

def _local_search_remove(blocks, v, t, deadline):
    subset_to_idx = {}
    all_t_subsets = list(itertools.combinations(range(v), t))
    for i, s in enumerate(all_t_subsets):
        subset_to_idx[s] = i
        
    from collections import defaultdict
    cover_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            cover_count[idx] += 1
            
    removed = set()
    for i in range(len(blocks)):
        if time.time() > deadline - 5:
            break
        if i in removed:
            continue
            
        is_critical = False
        for sub in itertools.combinations(blocks[i], t):
            idx = subset_to_idx[sub]
            if cover_count[idx] == 1:
                is_critical = True
                break
                
        if not is_critical:
            removed.add(i)
            for sub in itertools.combinations(blocks[i], t):
                idx = subset_to_idx[sub]
                cover_count[idx] -= 1
                
    if removed:
        blocks = [b for i, b in enumerate(blocks) if i not in removed]
        
    return blocks
