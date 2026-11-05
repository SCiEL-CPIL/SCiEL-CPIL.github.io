import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Points are 0..11
    points = range(12)
    
    # Generate all possible 6-subsets (blocks)
    all_blocks = list(itertools.combinations(points, 6))
    num_blocks = len(all_blocks) # 924
    
    # Generate all possible 5-subsets (constraints)
    all_5_subsets = list(itertools.combinations(points, 5))
    num_constraints = len(all_5_subsets) # 792
    
    # Try to use scipy.optimize.milp for an exact solution
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import scipy.sparse as sp
        
        # Build incidence matrix A (num_constraints x num_blocks)
        # A[i, j] = 1 if 5-subset i is contained in 6-subset j
        # We can build this by checking membership
        # For efficiency, we can use a dictionary of sets
        block_sets = [set(b) for b in all_blocks]
        subset_sets = [set(s) for s in all_5_subsets]
        
        # Create sparse matrix
        rows = []
        cols = []
        data = []
        
        for i, s in enumerate(subset_sets):
            for j, b in enumerate(block_sets):
                if s.issubset(b):
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
        
        A = sp.csr_matrix((data, (rows, cols)), shape=(num_constraints, num_blocks))
        
        # Objective: maximize sum(x_j) -> minimize -sum(x_j)
        c = -np.ones(num_blocks)
        
        # Constraints: A x <= 1
        constraint = LinearConstraint(A, ub=np.ones(num_constraints))
        
        # Bounds: 0 <= x_j <= 1
        bounds = Bounds(lb=np.zeros(num_blocks), ub=np.ones(num_blocks))
        
        # Solve
        res = milp(c=c, constraints=constraint, bounds=bounds, integrality=np.ones(num_blocks))
        
        if res.success:
            # Extract solution
            x = res.x
            blocks = [list(all_blocks[i]) for i in range(num_blocks) if x[i] > 0.5]
            return blocks
        else:
            # Fallback if solver failed
            pass
            
    except Exception:
        pass

    # Fallback: Greedy with randomization and restarts
    # This is less likely to find 132, but better than nothing
    # However, since the ILP is very likely to succeed, this is just a safety net.
    # A better fallback is a hardcoded construction if we had it, but we don't.
    # Let's do a simple greedy first as a baseline fallback.
    covered = set()
    blocks = []
    for b in all_blocks:
        subsets = list(itertools.combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
            
    return blocks
