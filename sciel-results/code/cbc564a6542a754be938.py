import itertools
import numpy as np
import random

def construct(v, k, t):
    t_subsets = list(itertools.combinations(range(v), t))
    k_subsets = list(itertools.combinations(range(v), k))
    n_t = len(t_subsets)
    n_k = len(k_subsets)
    t_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Attempt MILP Solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        rows = []
        cols = []
        data = []
        
        for j, k_sub in enumerate(k_subsets):
            for t_sub in itertools.combinations(k_sub, t):
                i = t_idx[t_sub]
                rows.append(i)
                cols.append(j)
                data.append(1)
                
        A = sp.csr_matrix((data, (rows, cols)), shape=(n_t, n_k))
        
        c = np.ones(n_k)
        constraints = LinearConstraint(A, lb=1, ub=np.inf)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_k)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x
            blocks = [list(k_subsets[j]) for j in range(n_k) if x[j] > 0.5]
            return blocks
    except Exception:
        pass
        
    # Fallback: Randomized Greedy (Different from deterministic seed)
    # Precompute masks for speed
    block_masks = [0] * n_k
    for i, k_sub in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    uncovered_mask = (1 << n_t) - 1
    blocks = []
    
    while uncovered_mask:
        # Calculate coverage for all blocks
        coverages = []
        for i in range(n_k):
            cov = (uncovered_mask & block_masks[i]).bit_count()
            if cov > 0:
                coverages.append((cov, i))
                
        if not coverages:
            break
            
        # Sort by coverage descending
        coverages.sort(reverse=True)
        
        # Randomly pick from top candidates (e.g., top 5 or proportional to coverage)
        # To make it distinct from deterministic greedy, we pick randomly from the top K
        top_k = min(10, len(coverages))
        # Weighted random choice or just uniform from top?
        # Uniform from top 10 is a good stochastic hill-climbing step.
        chosen_idx = random.choice(coverages[:top_k])[1]
        
        blocks.append(list(k_subsets[chosen_idx]))
        uncovered_mask &= ~block_masks[chosen_idx]
        
    return blocks
