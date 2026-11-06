import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    # Generate candidate points in Hex(3)
    candidates = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            # Hex distance formula
            d = (abs(a) + abs(b) + abs(a+b)) // 2
            if d <= 3:
                candidates.append([a, b])
    
    n = len(candidates) # Should be 37
    
    # Find unit distance pairs
    edges = []
    for i in range(n):
        a1, b1 = candidates[i]
        for j in range(i+1, n):
            a2, b2 = candidates[j]
            da = a1 - a2
            db = b1 - b2
            dist_sq = da*da + db*db + da*db
            if dist_sq == 1:
                edges.append((i, j))
                
    # MILP setup
    # Variables: x_i (binary) for each candidate
    # Objective: Max sum z_ij
    # z_ij <= x_i
    # z_ij <= x_j
    # z_ij >= x_i + x_j - 1
    # Sum x_i = 32
    
    num_edges = len(edges)
    # Total variables: n + num_edges
    # Indices 0..n-1 are x
    # Indices n..n+num_edges-1 are z
    
    # Objective c: maximize sum z -> minimize -sum z
    c = np.zeros(n + num_edges)
    c[n:] = -1.0 # Minimize -z is Maximize z
    
    # Bounds
    # x_i in [0, 1]
    # z_ij in [0, 1]
    lb = np.zeros(n + num_edges)
    ub = np.ones(n + num_edges)
    bounds = Bounds(lb, ub)
    
    # Integrality
    integrality = np.zeros(n + num_edges)
    integrality[:n] = 1.0 # x are integer (binary)
    # z can be continuous, but since x are binary and constraints are tight, z will be binary.
    # Actually, to be safe, we can keep z continuous.
    
    # Constraints
    # 1. Sum x_i = 32
    # 2. z_ij - x_i <= 0
    # 3. z_ij - x_j <= 0
    # 4. z_ij - x_i - x_j >= -1  => -z_ij + x_i + x_j <= 1
    
    # We will build the A matrix for LinearConstraint(A @ v <= b)
    # Or use sparse matrix for efficiency, but n is small.
    
    # Constraint 1: Sum x = 32
    # This is equality. LinearConstraint supports lower and upper bounds.
    # A_row for sum x: [1, 1, ..., 1, 0, ..., 0]
    
    # Let's count rows
    # 1 row for sum
    # 3 rows per edge
    num_rows = 1 + 3 * num_edges
    
    A = np.zeros((num_rows, n + num_edges))
    b_lower = np.zeros(num_rows)
    b_upper = np.zeros(num_rows)
    
    # Row 0: Sum x = 32
    A[0, :n] = 1.0
    b_lower[0] = 32
    b_upper[0] = 32
    
    row_idx = 1
    for k, (i, j) in enumerate(edges):
        z_idx = n + k
        
        # z_ij - x_i <= 0
        A[row_idx, z_idx] = 1.0
        A[row_idx, i] = -1.0
        b_upper[row_idx] = 0.0
        b_lower[row_idx] = -np.inf
        row_idx += 1
        
        # z_ij - x_j <= 0
        A[row_idx, z_idx] = 1.0
        A[row_idx, j] = -1.0
        b_upper[row_idx] = 0.0
        b_lower[row_idx] = -np.inf
        row_idx += 1
        
        # -z_ij + x_i + x_j <= 1
        A[row_idx, z_idx] = -1.0
        A[row_idx, i] = 1.0
        A[row_idx, j] = 1.0
        b_upper[row_idx] = 1.0
        b_lower[row_idx] = -np.inf
        row_idx += 1
        
    constraints = LinearConstraint(A, b_lower, b_upper)
    
    # Solve
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        x_vals = res.x[:n]
        selected_indices = [i for i in range(n) if x_vals[i] > 0.5]
        points = [candidates[i] for i in selected_indices]
        return points
    else:
        # Fallback to spiral if MILP fails
        # ... (existing spiral code)
        pass
