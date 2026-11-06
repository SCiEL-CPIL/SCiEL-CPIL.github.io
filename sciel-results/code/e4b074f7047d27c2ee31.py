"""Optimized K2,3-free graph construction on 20 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    
    # We will use MILP to solve this exactly.
    # Variables:
    # x[i, j] for i < j, index 0 to M-1 where M = n*(n-1)//2
    # z[i, j, k] for i < j and k != i, k != j.
    # We need to map these to a flat vector.
    
    M = n * (n - 1) // 2
    
    # Map (u, v) with u < v to index
    edge_idx = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx[(u, v)] = idx
            idx += 1
    
    # Map (u, v, w) with u < v, w != u, w != v to index
    # z_uv_w represents x_uw * x_vw
    z_idx = {}
    idx = M
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(n):
                if w == u or w == v:
                    continue
                z_idx[(u, v, w)] = idx
                idx += 1
    
    total_vars = idx
    
    # Objective: maximize sum of x
    # milp minimizes c^T x, so we use -1
    c = np.zeros(total_vars)
    for i in range(M):
        c[i] = -1.0
    
    # Constraints
    # 1. z_uv_w <= x_uw
    # 2. z_uv_w <= x_vw
    # 3. z_uv_w >= x_uw + x_vw - 1  =>  -z_uv_w + x_uw + x_vw >= 1  =>  z_uv_w - x_uw - x_vw <= -1
    # 4. For each pair (u, v), sum_{w != u, v} z_uv_w <= 2
    
    # We will build the constraint matrix A and bounds b.
    # Number of constraints:
    # 3 * len(z_idx) + M
    num_constraints = 3 * len(z_idx) + M
    
    A = np.zeros((num_constraints, total_vars))
    lb = np.zeros(num_constraints)
    ub = np.zeros(num_constraints)
    
    row = 0
    for (u, v, w), z_i in z_idx.items():
        # Get x indices
        # x_uw: ensure u < w or w < u
        if u < w:
            x_uw_i = edge_idx[(u, w)]
        else:
            x_uw_i = edge_idx[(w, u)]
        
        if v < w:
            x_vw_i = edge_idx[(v, w)]
        else:
            x_vw_i = edge_idx[(w, v)]
        
        # Constraint 1: z <= x_uw  =>  z - x_uw <= 0
        A[row, z_i] = 1.0
        A[row, x_uw_i] = -1.0
        lb[row] = -np.inf
        ub[row] = 0.0
        row += 1
        
        # Constraint 2: z <= x_vw  =>  z - x_vw <= 0
        A[row, z_i] = 1.0
        A[row, x_vw_i] = -1.0
        lb[row] = -np.inf
        ub[row] = 0.0
        row += 1
        
        # Constraint 3: z >= x_uw + x_vw - 1  =>  z - x_uw - x_vw <= -1
        A[row, z_i] = 1.0
        A[row, x_uw_i] = -1.0
        A[row, x_vw_i] = -1.0
        lb[row] = -np.inf
        ub[row] = -1.0
        row += 1
    
    # Constraint 4: Sum z <= 2 for each pair (u, v)
    for u in range(n):
        for v in range(u + 1, n):
            # Find all w != u, v
            # We stored z with u < v, so key is (u, v, w)
            for w in range(n):
                if w == u or w == v:
                    continue
                z_i = z_idx[(u, v, w)]
                A[row, z_i] = 1.0
            
            lb[row] = -np.inf
            ub[row] = 2.0
            row += 1
    
    # Variable bounds
    # x in {0, 1}, z in {0, 1}
    # milp uses integrality vector
    integrality = np.ones(total_vars)
    
    # Bounds for variables
    # x and z are binary, so 0 to 1
    var_bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
    
    # Solve
    try:
        res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=var_bounds)
        
        if res.status == 0:  # Optimal
            x_vals = res.x[:M]
            edges = []
            for (u, v), i in edge_idx.items():
                if x_vals[i] > 0.5:
                    edges.append([u, v])
            return edges
        else:
            # Fallback to heuristic if MILP fails or times out
            pass
            
    except Exception:
        pass
    
    # Fallback heuristic if MILP fails
    # (Keep the old logic as a safety net, but it should rarely be used)
    def popcount(x):
        return bin(x).count('1')
    
    def can_add_edge(adj, u, v):
        if u == v:
            return False
        if (adj[u] >> v) & 1:
            return False
        if popcount(adj[u] & adj[v]) >= 3:
            return False
        for y in range(n):
            if y == u or y == v:
                continue
            if (adj[u] >> y) & 1:
                if popcount(adj[v] & adj[y]) >= 2:
                    return False
        for x in range(n):
            if x == u or x == v:
                continue
            if (adj[v] >> x) & 1:
                if popcount(adj[u] & adj[x]) >= 2:
                    return False
        return True

    adj = [0] * n
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    for u, v in all_edges:
        if can_add_edge(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
