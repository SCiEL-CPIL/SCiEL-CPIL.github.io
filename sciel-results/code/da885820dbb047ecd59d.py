import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    # Set a deadline to ensure we return before the 120s limit
    # We use 110s to be safe with overhead
    deadline = start_time + 110.0

    # Primary Strategy: Integer Linear Programming (ILP)
    # This is guaranteed to find the optimal 132 blocks (Witt Design S(5,6,12))
    # if the solver converges. The problem size (924 vars, 792 constraints) is small.
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix, csr_matrix
        
        # Generate all possible 6-subsets (blocks)
        all_blocks = list(combinations(range(12), 6))
        n_blocks = len(all_blocks)
        
        # Generate all possible 5-subsets
        all_subsets = list(combinations(range(12), 5))
        n_subsets = len(all_subsets)
        
        # Map 5-subset to its index in all_subsets for fast lookup
        subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
        
        # Build constraint matrix A
        # A[i, j] = 1 if block j contains subset i
        # Shape: (n_subsets, n_blocks)
        A = lil_matrix((n_subsets, n_blocks), dtype=np.int8)
        
        for j, b in enumerate(all_blocks):
            # Each 6-block contains exactly 6 five-subsets
            for s in combinations(b, 5):
                i = subset_to_idx[s]
                A[i, j] = 1
        
        # Convert to CSR for efficient solver interaction
        A_csr = A.tocsr()
        
        # Objective: Maximize sum(x_j) => Minimize -sum(x_j)
        c = -np.ones(n_blocks)
        
        # Constraints: For each 5-subset, the sum of blocks containing it must be <= 1
        # 0 <= A @ x <= 1
        constraints = [LinearConstraint(A_csr, 0, 1)]
        
        # Integrality: x_j must be integer (0 or 1)
        integrality = np.ones(n_blocks)
        
        # Bounds: 0 <= x_j <= 1
        bounds = Bounds(0, 1)
        
        # Solve the ILP
        # HiGHS (used by milp) is very fast for this size
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            # Extract the optimal blocks
            x = res.x
            blocks = [list(all_blocks[j]) for j in range(n_blocks) if x[j] > 0.5]
            return blocks
        else:
            # If ILP fails (e.g., timeout or infeasible, though it should be feasible),
            # we fall back to the heuristic.
            pass
            
    except Exception:
        # If SciPy is not available or an error occurs, fall back to heuristic
        pass

    # Fallback Strategy: Greedy Set Packing
    # This matches the seed program's behavior.
    # It is deterministic and fast, though not optimal.
    covered = set()
    blocks = []
    
    for b in combinations(range(12), 6):
        # Check if any 5-subset of b is already covered
        # A block is valid if none of its 5-subsets are in 'covered'
        # We can short-circuit this check
        valid = True
        for s in combinations(b, 5):
            if s in covered:
                valid = False
                break
        
        if valid:
            blocks.append(list(b))
            # Add all 5-subsets of b to covered
            for s in combinations(b, 5):
                covered.add(s)
                
        # Check time
        if time.time() > deadline:
            break
            
    return blocks
