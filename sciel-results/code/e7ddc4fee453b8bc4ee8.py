import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    # Vertices are 2 to 256
    vertices = list(range(2, 257))
    n = len(vertices)
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Build conflict graph
    # x_i + x_j <= 1 for all pairs (i, j) where lcm(vertices[i], vertices[j]) <= 256
    edges = []
    
    # Optimization: Only check pairs where product is small or gcd is large
    # lcm(a, b) <= 256  =>  a*b / gcd(a, b) <= 256
    # We can iterate i < j
    for i in range(n):
        a = vertices[i]
        # If a > 256, no conflicts (but a is max 256)
        # For a fixed a, b must satisfy b <= 256 * gcd(a, b) / a
        # Since gcd(a, b) <= a, b <= 256.
        # A simple O(n^2) check is 255^2 / 2 ~ 32k checks. Very fast.
        for j in range(i + 1, n):
            b = vertices[j]
            # Quick rejection: if a*b > 256 * min(a,b) then lcm > 256?
            # lcm = a*b/g. g <= min(a,b). So lcm >= a*b/min(a,b) = max(a,b).
            # This doesn't help much for rejection.
            # Just compute lcm.
            g = math.gcd(a, b)
            l = a // g * b
            if l <= 256:
                edges.append((i, j))
                
    if time.time() > deadline:
        # Fallback if even graph construction is slow (unlikely)
        return [256]

    # Objective: Maximize sum(1/v * x_v)
    # milp minimizes. So minimize -sum(1/v * x_v)
    c = np.array([-1.0 / v for v in vertices])
    
    # Constraints: x_i + x_j <= 1 for each edge
    if edges:
        A_rows = []
        A_cols = []
        A_data = []
        for i, j in edges:
            A_rows.append(len(A_rows)) # row index
            A_cols.append(i)
            A_data.append(1.0)
            A_rows.append(len(A_rows))
            A_cols.append(j)
            A_data.append(1.0)
        
        # The row index logic above is wrong for sparse matrix construction directly.
        # Let's construct properly.
        row_ind = []
        col_ind = []
        data = []
        
        for edge_idx, (i, j) in enumerate(edges):
            row_ind.append(edge_idx)
            col_ind.append(i)
            data.append(1.0)
            
            row_ind.append(edge_idx)
            col_ind.append(j)
            data.append(1.0)
            
        A = csc_matrix((data, (row_ind, col_ind)), shape=(len(edges), n))
        b = np.ones(len(edges))
        
        constraints = LinearConstraint(A, ub=b)
    else:
        constraints = LinearConstraint(np.zeros((0, n)), ub=np.zeros(0))

    # Variables are binary
    integrality = np.ones(n)
    
    # Bounds [0, 1]
    bounds = Bounds(lb=0, ub=1)
    
    try:
        # Run MIP solver
        # Time limit for solver itself
        solver_time_limit = max(1.0, deadline - (time.time() - start_time) - 5.0)
        
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": solver_time_limit, "presolve": True}
        )
        
        if res.status == 0 or res.status == 2: # 0: optimal, 2: feasible (might be incumbent)
            if res.x is not None:
                # res.x might be float, need to round
                x = np.round(res.x)
                selected = [vertices[i] for i in range(n) if x[i] > 0.5]
                if selected:
                    return selected
        
        # If MIP failed or timed out without a good result, use a heuristic
    except Exception as e:
        pass
    
    # Heuristic Fallback: Greedy by weight (descending 1/v)
    # Sort vertices by 1/v descending (i.e., v ascending)
    # Wait, 1/v is largest for small v.
    # So we should try to include small numbers if possible.
    # But small numbers conflict with many others.
    # Greedy: Sort by 1/v descending. Add if no conflict.
    
    sorted_vertices = sorted(vertices, key=lambda v: 1.0/v, reverse=True)
    selected = []
    selected_set = set()
    
    for v in sorted_vertices:
        # Check if v conflicts with any in selected
        conflict = False
        for u in selected:
            if math.lcm(v, u) <= 256:
                conflict = True
                break
        if not conflict:
            selected.append(v)
            selected_set.add(v)
            
    if selected:
        return selected
    
    # Final fallback: just return a valid single element
    return [256]
