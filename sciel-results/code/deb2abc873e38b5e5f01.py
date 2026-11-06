"""Find Hamilton cycle with minimal max edge arrival index."""
# EVOLVE-BLOCK-START
import time
import sys
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

N = 24
TIME_LIMIT = 85  # seconds, leaving margin for evaluator overhead
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Build adjacency: for each edge in order, assign position (1-based)
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

def solve_hamilton_cycle_milp(K, deadline):
    """
    Check if a Hamilton cycle exists in G_K using MILP.
    Returns a list of vertices representing the cycle, or None.
    """
    # Build list of available directed arcs
    arcs = []
    arc_index = {}
    for u in range(N):
        for v in range(N):
            if u == v:
                continue
            if (u, v) in edge_pos and edge_pos[(u, v)] <= K:
                idx = len(arcs)
                arcs.append((u, v))
                arc_index[(u, v)] = idx
    
    if not arcs:
        return None
        
    num_arcs = len(arcs)
    # Variables: x_0...x_{num_arcs-1} (binary), u_1...u_{N-1} (integer)
    # Total variables: num_arcs + (N-1)
    num_vars = num_arcs + (N - 1)
    
    # Objective: minimize 0 (feasibility)
    c = np.zeros(num_vars)
    
    # Integrality: x are binary (0/1), u are integer (1..N-1)
    integrality = np.zeros(num_vars)
    integrality[:num_arcs] = 1  # x are binary
    integrality[num_arcs:] = 1  # u are integer
    
    # Bounds for u: 1 <= u_i <= N-1
    lb = np.zeros(num_vars)
    ub = np.zeros(num_vars)
    lb[:num_arcs] = 0
    ub[:num_arcs] = 1
    lb[num_arcs:] = 1
    ub[num_arcs:] = N - 1
    
    # Constraints
    A_rows = []
    lb_rows = []
    ub_rows = []
    
    # 1. Flow conservation: sum_out(i) = 1, sum_in(i) = 1
    for i in range(N):
        # Outgoing
        row_out = np.zeros(num_vars)
        for idx, (u, v) in enumerate(arcs):
            if u == i:
                row_out[idx] = 1
        if np.any(row_out):
            A_rows.append(row_out)
            lb_rows.append(1)
            ub_rows.append(1)
            
        # Incoming
        row_in = np.zeros(num_vars)
        for idx, (u, v) in enumerate(arcs):
            if v == i:
                row_in[idx] = 1
        if np.any(row_in):
            A_rows.append(row_in)
            lb_rows.append(1)
            ub_rows.append(1)
            
    # 2. MTZ constraints: u_i - u_j + N * x_ij <= N - 1
    # for all i, j in 1..N-1, i != j
    # Note: x_ij is the variable for arc (i, j)
    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                continue
            # Check if arc (i, j) exists
            if (i, j) not in arc_index:
                continue
            idx = arc_index[(i, j)]
            
            row = np.zeros(num_vars)
            # u_i is at index num_arcs + (i-1)
            row[num_arcs + (i-1)] = 1
            # u_j is at index num_arcs + (j-1)
            row[num_arcs + (j-1)] = -1
            # x_ij is at index idx
            row[idx] = N
            
            A_rows.append(row)
            lb_rows.append(-np.inf)
            ub_rows.append(N - 1)
            
    if not A_rows:
        # No constraints? Impossible for N>1
        return None
        
    A = np.array(A_rows)
    lb_arr = np.array(lb_rows)
    ub_arr = np.array(ub_rows)
    
    constraints = LinearConstraint(A, lb_arr, ub_arr)
    bounds = Bounds(lb, ub)
    
    # Solve
    start_time = time.time()
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": max(1, deadline - time.time())})
    except Exception:
        return None
        
    if res.status != 0:
        return None
        
    # Extract solution
    x_vals = res.x[:num_arcs]
    cycle_edges = []
    for idx, (u, v) in enumerate(arcs):
        if x_vals[idx] > 0.5:
            cycle_edges.append((u, v))
            
    # Reconstruct cycle
    if len(cycle_edges) != N:
        return None
        
    # Build adjacency from cycle edges
    next_node = {}
    for u, v in cycle_edges:
        next_node[u] = v
        
    # Start at 0
    current = 0
    cycle = [0]
    for _ in range(N):
        current = next_node[current]
        cycle.append(current)
        
    # Verify cycle
    if cycle[0] != cycle[-1]:
        return None
    if len(set(cycle[:-1])) != N:
        return None
        
    return cycle[:-1]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - 5  # safety margin
    
    # Binary search for minimum K
    lo, hi = 24, len(EDGE_ORDER)
    best_cycle = None
    
    while lo <= hi:
        if time.time() > deadline - 3:
            break
        mid = (lo + hi) // 2
        cycle = solve_hamilton_cycle_milp(mid, deadline)
        if cycle is not None:
            best_cycle = cycle
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is not None:
        return best_cycle
        
    # Fallback: try larger K values directly if binary search failed to find anything
    # (should not happen if graph is complete enough, but just in case)
    for K in range(50, len(EDGE_ORDER) + 1, 10):
        if time.time() > deadline - 2:
            break
        cycle = solve_hamilton_cycle_milp(K, deadline)
        if cycle is not None:
            best_cycle = cycle
            break
            
    if best_cycle is not None:
        return best_cycle
        
    # Last resort: identity permutation
    return list(range(N))
# EVOLVE-BLOCK-END
