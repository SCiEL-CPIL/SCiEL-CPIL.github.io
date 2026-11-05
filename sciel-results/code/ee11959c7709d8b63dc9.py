import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
def construct():
    n = 16
    target = 65
    deadline = time.time() + 118.0  # safety margin

    # 1. Precompute all possible edges and triangles
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Incidence matrix: A[i, e] = 1 if edge e is incident to vertex i
    A = np.zeros((n, m), dtype=int)
    for i, (u, v) in enumerate(all_edges):
        A[u, i] = 1
        A[v, i] = 1

    # Find all triangles
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            # Common neighbors w > v to avoid duplicates and ensure unique triangle
            # Actually, standard triangle enumeration:
            # For each edge (u, v), find common neighbors w.
            # To be efficient, we can just iterate all triples u < v < w
            for w in range(v + 1, n):
                if (u, v) in edge_to_idx and (u, w) in edge_to_idx and (v, w) in edge_to_idx:
                    triangles.append((u, v, w))
    
    t_count = len(triangles)
    
    # Precompute triangle edge indices
    # For each triangle, we need the indices of its 3 edges
    tri_edge_indices = []
    for u, v, w in triangles:
        tri_edge_indices.append([
            edge_to_idx[(u, v)],
            edge_to_idx[(u, w)],
            edge_to_idx[(v, w)]
        ])
    
    # 2. Define ILP Variables
    # x[e] for e in range(m): 1 if edge is present
    # y[u] for u in range(n): degree of vertex u
    # T: max triangle degree sum
    # Total variables: m + n + 1
    
    # Objective: minimize T
    c = np.zeros(m + n + 1)
    c[m + n] = 1.0  # Minimize T
    
    # Constraints:
    # 1. Exactly 65 edges: sum(x) = 65
    # 2. Degree consistency: y[u] = sum(A[u, e] * x[e]) for all u
    #    => y[u] - sum(A[u, e] * x[e]) = 0
    # 3. Triangle constraint: For each triangle k with vertices (u, v, w):
    #    y[u] + y[v] + y[w] <= T
    #    => y[u] + y[v] + y[w] - T <= 0
    
    # Build constraint matrix
    # Rows:
    # Row 0: Sum of edges = 65
    # Rows 1..n: Degree constraints
    # Rows n+1..n+t_count: Triangle constraints
    
    num_rows = 1 + n + t_count
    num_vars = m + n + 1
    
    # We will use LinearConstraint with lb and ub arrays
    lb = np.zeros(num_rows)
    ub = np.zeros(num_rows)
    
    # Constraint 0: Sum x = 65
    lb[0] = 65
    ub[0] = 65
    
    # Constraint 1..n: y[u] - sum(A[u, :]*x) = 0
    # This is equality, so lb == ub == 0
    # Matrix structure:
    # For row r (1 <= r <= n), corresponding to vertex u = r-1:
    # Coeffs for x: -A[u, :]
    # Coeff for y[u]: 1
    # Coeff for T: 0
    
    # Constraint n+1..: y[u] + y[v] + y[w] - T <= 0
    # lb = -inf, ub = 0
    
    # Instead of building a huge sparse matrix manually with potential errors,
    # let's construct it carefully.
    
    # Since m=120, n=16, t_count is likely small (maybe < 100), dense is fine.
    constraint_matrix = np.zeros((num_rows, num_vars))
    
    # Row 0: Sum x = 65
    constraint_matrix[0, :m] = 1.0
    
    # Rows 1..n: Degree constraints
    for u in range(n):
        row_idx = 1 + u
        constraint_matrix[row_idx, :m] = -A[u, :]  # -sum(A[u, e]*x[e])
        constraint_matrix[row_idx, m + u] = 1.0    # + y[u]
        # lb[row_idx] = 0, ub[row_idx] = 0 (default)
        
    # Rows n+1..: Triangle constraints
    for k in range(t_count):
        row_idx = 1 + n + k
        u, v, w = triangles[k]
        constraint_matrix[row_idx, m + u] = 1.0
        constraint_matrix[row_idx, m + v] = 1.0
        constraint_matrix[row_idx, m + w] = 1.0
        constraint_matrix[row_idx, m + n] = -1.0  # -T
        ub[row_idx] = 0.0
        lb[row_idx] = -np.inf
        
    # Variable bounds
    # x: 0 <= x <= 1
    # y: 0 <= y <= n-1 (max degree)
    # T: 0 <= T <= 3*(n-1) (max sum)
    
    lb_vars = np.zeros(num_vars)
    ub_vars = np.zeros(num_vars)
    ub_vars[:m] = 1.0
    ub_vars[m:m+n] = n - 1
    ub_vars[m+n] = 3 * (n - 1)
    
    integrality = np.zeros(num_vars)
    integrality[:m] = 1.0  # x must be integer
    # y and T can be continuous in the relaxation, but y is determined by x so it's effectively integer.
    # For strict ILP, we can force y to be integer too, but it's redundant if x is.
    # Let's keep y continuous for the solver to be faster, as y = A x will be integer if x is.
    
    constraints = LinearConstraint(constraint_matrix, lb, ub)
    bounds = Bounds(lb_vars, ub_vars)
    
    # Solve ILP
    # We might need to run multiple times with different initial bounds or just one strong run.
    # SciPy's milp uses HiGHS.
    
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0: # Optimal
            x_sol = res.x[:m]
            edges = []
            for i in range(m):
                if x_sol[i] > 0.5:
                    edges.append(list(all_edges[i]))
            
            # Verify and return
            if len(edges) == target:
                return {'n': n, 'edges': edges}
                
    except Exception as e:
        print(f"ILP failed: {e}", file=sys.stderr)
        
    # Fallback: Use the seed construction if ILP fails or times out
    edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
