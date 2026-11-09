import itertools
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    # Try MILP approach first
    try:
        sol = solve_milp(v, k, t)
        if sol is not None:
            return sol
    except Exception:
        pass
    
    # Fallback to Greedy + Prune (same as original)
    return greedy_with_prune(v, k, t)

def solve_milp(v, k, t):
    points = range(v)
    k_subs = list(itertools.combinations(points, k))
    num_blocks = len(k_subs)
    
    t_subs = list(itertools.combinations(points, t))
    num_t_subs = len(t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    rows = []
    cols = []
    data = []
    
    # Optimization: Pre-compute t-subset indices for each block?
    # Just loop. 346k * 35 is ~12M ops. Should be < 5s.
    for j, block in enumerate(k_subs):
        for t_sub in itertools.combinations(block, t):
            i = t_sub_to_idx[t_sub]
            rows.append(i)
            cols.append(j)
            data.append(1)
            
    A = sp.csr_matrix((data, (rows, cols)), shape=(num_t_subs, num_blocks))
    
    c = np.ones(num_blocks)
    constraint = LinearConstraint(A, 1, np.inf)
    bounds = Bounds(0, 1)
    integrality = np.ones(num_blocks)
    
    res = milp(
        c=c,
        constraints=constraint,
        integrality=integrality,
        bounds=bounds,
        options={'time_limit': 100, 'mip_rel_gap': 0.0} # 0.0 means optimal, but time_limit cuts it.
    )
    
    if res.success and res.x is not None:
        x = np.round(res.x).astype(int)
        solution = [list(k_subs[j]) for j in range(num_blocks) if x[j] == 1]
        
        # Quick validation
        if is_valid_covering(solution, v, k, t):
            return solution
            
    return None

def is_valid_covering(blocks, v, k, t):
    covered = set()
    for block in blocks:
        for t_sub in itertools.combinations(block, t):
            covered.add(t_sub)
    all_t_subs = set(itertools.combinations(range(v), t))
    return all_t_subs.issubset(covered)

def greedy_with_prune(v, k, t):
    # Copy of the original logic to ensure baseline performance
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    k_subs = list(itertools.combinations(range(v), k))
    block_data = []
    for block in k_subs:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_data.append((block, mask))
        
    uncovered_mask = (1 << num_t_subs) - 1
    solution = []
    solution_masks = []
    
    while uncovered_mask:
        best_gain = 0
        best_block = None
        best_mask = None
        
        for block, mask in block_data:
            gain = (uncovered_mask & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_mask = mask
                
        if best_gain == 0:
            break
            
        solution.append(list(best_block))
        solution_masks.append(best_mask)
        uncovered_mask &= ~best_mask
        
    # Prune
    counts = [0] * num_t_subs
    for block in solution:
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            counts[idx] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(solution)):
            block = solution[i]
            for t_sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[t_sub]
                counts[idx] -= 1
            if min(counts) > 0:
                solution.pop(i)
                changed = True
                break
            else:
                for t_sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[t_sub]
                    counts[idx] += 1
                    
    return solution
