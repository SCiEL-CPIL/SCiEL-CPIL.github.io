"""Optimized Hamilton cycle finder using ILP (scipy.optimize.milp)."""
import sys
sys.setrecursionlimit(10000)
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_order = P['edge_order']

# Map edges to indices
edge_pos = {}
for i, (u, v) in enumerate(edge_order):
    if u > v:
        u, v = v, u
    edge_pos[(u, v)] = i + 1

# List of all edges for ILP variables
# We need to map variable index to edge (u, v)
# Number of edges is N*(N-1)/2
edges = []
for u in range(N):
    for v in range(u + 1, N):
        edges.append((u, v))

M = len(edges)

# Precompute edge positions for all edges
edge_indices = []
for u, v in edges:
    edge_indices.append(edge_pos[(u, v)])

def solve_ilp(K):
    """
    Check if a Hamiltonian cycle exists using only edges with index <= K.
    Returns the cycle as a list of vertices if found, else None.
    """
    # Variables:
    # x_i for i in range(M) : binary, indicates if edge i is in the cycle
    # u_i for i in range(N) : integer, MTZ potentials
    
    # Total variables: M + N
    n_vars = M + N
    
    # Objective: minimize 0 (feasibility problem)
    c = np.zeros(n_vars)
    
    # Bounds
    # x_i in [0, 1]
    # u_0 = 0
    # u_i in [1, N-1] for i > 0
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)
    
    for i in range(M):
        lb[i] = 0
        ub[i] = 1
        
    # u_0
    lb[M] = 0
    ub[M] = 0
    
    # u_i for i > 0
    for i in range(1, N):
        lb[M + i] = 1
        ub[M + i] = N - 1
        
    integrality = np.zeros(n_vars)
    integrality[:M] = 1  # x are binary
    integrality[M:] = 1  # u are integer
    
    # Constraints
    # 1. Degree constraints: sum of x for edges incident to node i is 2
    # For each node i, find indices of edges connected to i
    node_edges = [[] for _ in range(N)]
    for idx, (u, v) in enumerate(edges):
        node_edges[u].append(idx)
        node_edges[v].append(idx)
        
    A_list = []
    lb_list = []
    ub_list = []
    
    for i in range(N):
        row = np.zeros(n_vars)
        for e_idx in node_edges[i]:
            row[e_idx] = 1
        A_list.append(row)
        lb_list.append(2)
        ub_list.append(2)
        
    # 2. MTZ constraints: u_i - u_j + N * x_ij <= N - 1
    # For each pair (i, j) with i != j, we have a constraint.
    # But we only have x variables for i < j.
    # If edge (i, j) is in the cycle, x_ij = 1.
    # The constraint is: u_i - u_j + N * x_ij <= N - 1
    # And: u_j - u_i + N * x_ij <= N - 1
    
    # We iterate over all edges (i, j) with i < j.
    # For each such edge, we add two constraints.
    
    for e_idx, (i, j) in enumerate(edges):
        # Constraint 1: u_i - u_j + N * x_e <= N - 1
        row1 = np.zeros(n_vars)
        row1[M + i] = 1
        row1[M + j] = -1
        row1[e_idx] = N
        A_list.append(row1)
        lb_list.append(-np.inf)
        ub_list.append(N - 1)
        
        # Constraint 2: u_j - u_i + N * x_e <= N - 1
        row2 = np.zeros(n_vars)
        row2[M + i] = -1
        row2[M + j] = 1
        row2[e_idx] = N
        A_list.append(row2)
        lb_list.append(-np.inf)
        ub_list.append(N - 1)

    # 3. Edge availability: x_e = 0 if edge_pos[e] > K
    # We can handle this by setting ub[e_idx] = 0 if not available.
    # We already set ub[e_idx] = 1. We need to override for unavailable edges.
    for e_idx, pos in enumerate(edge_indices):
        if pos > K:
            ub[e_idx] = 0

    A = np.array(A_list)
    lb_con = np.array(lb_list)
    ub_con = np.array(ub_list)
    
    constraints = LinearConstraint(A, lb_con, ub_con)
    bounds = Bounds(lb, ub)
    
    res = milp(c, integrality=integrality, constraints=constraints, bounds=bounds)
    
    if res.status == 0:
        # Extract solution
        x_sol = res.x[:M]
        # Build adjacency
        adj = [[] for _ in range(N)]
        for e_idx, (u, v) in enumerate(edges):
            if x_sol[e_idx] > 0.5:
                adj[u].append(v)
                adj[v].append(u)
        
        # Trace cycle starting from 0
        cycle = [0]
        visited = {0}
        curr = 0
        for _ in range(N):
            nexts = adj[curr]
            if not nexts:
                return None # Should not happen if ILP is correct
            # Pick next unvisited
            nxt = None
            for n in nexts:
                if n not in visited:
                    nxt = n
                    break
            if nxt is None:
                # If all neighbors visited, must be closing the loop
                # But we only close after N steps.
                # If we are at step < N-1 and no unvisited neighbor, it's a subtour or error.
                # However, MTZ prevents subtours.
                # If curr != 0 and no unvisited neighbor, it might be the last node connecting to 0.
                # But we check closing at the end.
                if len(cycle) == N:
                    break
                else:
                    return None
            
            cycle.append(nxt)
            visited.add(nxt)
            curr = nxt
        
        # Check if it's a valid Hamiltonian cycle
        if len(cycle) != N:
            return None
        # Check closing edge
        last = cycle[-1]
        if 0 not in adj[last]:
            return None
            
        return cycle
    else:
        return None

# Binary search for minimum K
lo, hi = 1, len(edge_order)
best_cycle = None

while lo <= hi:
    mid = (lo + hi) // 2
    cycle = solve_ilp(mid)
    if cycle is not None:
        best_cycle = cycle
        hi = mid - 1
    else:
        lo = mid + 1

def construct():
    if best_cycle is not None:
        return best_cycle
    # Fallback: identity permutation (might be invalid, but better than exception)
    return list(range(N))
# EVOLVE-BLOCK-END
