import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Precompute conflict graph
    # conflict_mask[i] is a bitset where bit j is set if i and j+1 conflict
    # We use 0-indexed internally, so vertex i corresponds to number i+1
    conflict_mask = [0] * N
    conflict_pairs = []
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            if (a*b) % (a+b) == 0:
                conflict_mask[a-1] |= (1 << (b-1))
                conflict_mask[b-1] |= (1 << (a-1))
                conflict_pairs.append((a-1, b-1))
    
    # Attempt to solve using MILP
    # Variables: x_i in {0, 1}
    # Maximize sum(x_i)
    # Subject to x_i + x_j <= 1 for all (i,j) in conflict_pairs
    #            x_i >= 0
    
    if time.time() > deadline - 10:
        # Not enough time for MILP setup/solve, use heuristic
        return greedy_mis(conflict_mask, N)
        
    try:
        c = np.ones(N)  # Coefficients for objective function (maximize)
        
        # Constraints: x_i + x_j <= 1
        # Number of constraints is len(conflict_pairs)
        M = len(conflict_pairs)
        if M == 0:
            # No conflicts, take all
            return list(range(1, N+1))
            
        # Build A matrix sparsely? scipy.milp takes dense or sparse.
        # For N=256 and potentially ~1000-5000 conflicts, dense is fine but sparse is safer.
        # However, scipy.milp accepts LinearConstraint which can take sparse A.
        # Let's use sparse matrix for A to be efficient.
        from scipy.sparse import lil_matrix
        A = lil_matrix((M, N), dtype=int)
        
        for k, (i, j) in enumerate(conflict_pairs):
            A[k, i] = 1
            A[k, j] = 1
        A = A.tocsr()
        
        # LinearConstraint(A, lb, ub)
        # We want A @ x <= 1, so lb = -inf, ub = 1
        # Actually, LinearConstraint(A, lb, ub) enforces lb <= A @ x <= ub
        # So lb = -np.inf, ub = np.ones(M)
        
        constraints = LinearConstraint(A, -np.inf, np.ones(M))
        
        # Bounds for x: 0 <= x_i <= 1
        bounds = Bounds(0, 1)
        
        # Integrality: 1 for binary variables
        integrality = np.ones(N)
        
        # Solve
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if time.time() > deadline - 2:
             # Check if we have a solution before returning
             if res.status == 0:
                 x = res.x
                 solution_indices = [i for i in range(N) if x[i] > 0.5]
                 if solution_indices:
                     return [idx + 1 for idx in solution_indices]
             return greedy_mis(conflict_mask, N)

        if res.status == 0:
            x = res.x
            solution_indices = [i for i in range(N) if x[i] > 0.5]
            if solution_indices:
                # Verify solution just in case (MILP might have numerical issues)
                valid = True
                sol_set = set(solution_indices)
                for i in solution_indices:
                    # Check against all j in sol_set where j > i
                    mask_i = conflict_mask[i]
                    # Check if any other selected vertex is in the conflict mask
                    # We can just check if (mask_i & selection_bitset) != 0
                    # But simpler: iterate
                    pass 
                
                # Let's do a quick verification
                for i_idx in range(len(solution_indices)):
                    for j_idx in range(i_idx+1, len(solution_indices)):
                        u = solution_indices[i_idx]
                        v = solution_indices[j_idx]
                        if (u < v and (conflict_mask[u] & (1 << v))) or (v < u and (conflict_mask[v] & (1 << u))):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    return [idx + 1 for idx in solution_indices]
                else:
                    # Fallback to greedy if MILP result is invalid
                    return greedy_mis(conflict_mask, N)
            else:
                return greedy_mis(conflict_mask, N)
        else:
            # Solver failed, fallback to greedy
            return greedy_mis(conflict_mask, N)
            
    except Exception:
        return greedy_mis(conflict_mask, N)

def greedy_mis(conflict_mask, N):
    """
    Greedy heuristic for MIS:
    1. Sort vertices by degree (ascending) - pick low degree first?
       Or descending? 
       Standard greedy MIS: Pick vertex with lowest degree in current graph, add to set, remove neighbors.
       Or: Pick vertex with highest degree, add to set, remove neighbors.
       Let's try: Iterate vertices in order of increasing degree in the original graph.
       If vertex is not excluded, add it and exclude its neighbors.
    """
    # Calculate degrees
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x])
    
    selected = set()
    excluded = set()
    
    for v in order:
        if v in excluded:
            continue
        if v in selected:
            continue
        # Check if v conflicts with any already selected? 
        # Since we remove neighbors when we select, if v is not excluded, it shouldn't conflict with selected.
        # But wait, "excluded" is updated when we select a node.
        # So if v is not in excluded, it is safe to select.
        selected.add(v)
        # Exclude neighbors
        mask = conflict_mask[v]
        # Iterate bits in mask
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            excluded.add(idx)
            m ^= lb
            
    if not selected:
        return [1]
        
    return [v + 1 for v in sorted(selected)]
