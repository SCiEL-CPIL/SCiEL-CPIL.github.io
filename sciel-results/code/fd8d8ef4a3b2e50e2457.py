import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_INDEX = {}
for i, edge in enumerate(P['edge_order']):
    u, v = edge
    if u > v:
        u, v = v, u
    EDGE_INDEX[(u, v)] = i + 1

def get_edge_index(u, v):
    if u > v:
        u, v = v, u
    return EDGE_INDEX.get((u, v), 10**9)

def check_feasibility(S):
    """
    Checks if there is a Hamilton cycle using only edges with index <= S.
    Returns the cycle as a list of vertices if found, else None.
    """
    # Variables: x_{uv} for each edge, w_u for each vertex (MTZ)
    # Total variables: N*N + N = 576 + 24 = 600
    # But we only need symmetric x_{uv} for u < v, so N*(N-1)/2 = 276 variables for x.
    # Let's use symmetric representation to reduce size.
    
    edges = []
    edge_map = {}
    idx = 0
    for u in range(N):
        for v in range(u+1, N):
            if get_edge_index(u, v) <= S:
                edge_map[(u, v)] = idx
                edges.append((u, v))
                idx += 1
    
    if len(edges) < N:
        return None

    M = len(edges)
    # Variables: x_0 ... x_{M-1}, w_0 ... w_{N-1}
    num_vars = M + N
    c = np.zeros(num_vars) # Minimize 0
    
    # Constraints:
    # 1. Degree constraint: sum of incident edges = 2 for each node
    # 2. MTZ subtour elimination: w_u - w_v + N * x_{uv} <= N - 1
    #    w_u - w_v + N * x_{vu} <= N - 1
    #    w_u, w_v in [1, N]
    
    A = []
    b_lower = []
    b_upper = []
    
    # Degree constraints
    for node in range(N):
        row = np.zeros(num_vars)
        for i, (u, v) in enumerate(edges):
            if u == node:
                row[i] = 1
            if v == node:
                row[i] = 1
        A.append(row)
        b_lower.append(2)
        b_upper.append(2)
        
    # MTZ constraints
    # w_u - w_v + N * x_{uv} <= N - 1
    # -w_u + w_v + N * x_{uv} >= 1 - N  (since x_{uv} is 0/1, if x=1, -w_u + w_v <= N-1 => w_v - w_u <= N-1)
    # Actually, standard MTZ is:
    # w_u - w_v + N x_{uv} <= N - 1
    # w_v - w_u + N x_{uv} <= N - 1
    # This is redundant. Just use one direction? No, we need to prevent cycles in both directions.
    # Since graph is undirected, we can fix a root node (e.g., 0) and say w_0 = 1.
    # Then for all u, v:
    # w_u - w_v + N x_{uv} <= N - 1
    # w_v - w_u + N x_{uv} <= N - 1
    
    # Let's add constraints for all pairs u, v in edges.
    for i, (u, v) in enumerate(edges):
        # w_u - w_v + N * x_i <= N - 1
        row1 = np.zeros(num_vars)
        row1[i] = N
        row1[M + u] = 1
        row1[M + v] = -1
        A.append(row1)
        b_lower.append(-np.inf)
        b_upper.append(N - 1)
        
        # w_v - w_u + N * x_i <= N - 1
        row2 = np.zeros(num_vars)
        row2[i] = N
        row2[M + v] = 1
        row2[M + u] = -1
        A.append(row2)
        b_lower.append(-np.inf)
        b_upper.append(N - 1)

    A = np.array(A)
    b_lower = np.array(b_lower)
    b_upper = np.array(b_upper)
    
    # Bounds
    # x_i in [0, 1]
    # w_i in [1, N]
    bounds = Bounds(
        lb=[0] * M + [1] * N,
        ub=[1] * M + [N] * N
    )
    
    # Integrality
    integrality = [1] * M + [0] * N
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A, b_lower, b_upper), 
                   integrality=integrality, bounds=bounds, 
                   options={"time_limit": 5})
        if res.status == 0:
            x_vals = res.x[:M]
            # Extract cycle
            adj = {i: [] for i in range(N)}
            for i, (u, v) in enumerate(edges):
                if x_vals[i] > 0.5:
                    adj[u].append(v)
                    adj[v].append(u)
            
            # Verify it's a single cycle of length N
            if all(len(nei) == 2 for nei in adj.values()):
                # Traverse
                cycle = [0]
                prev = -1
                curr = 0
                for _ in range(N):
                    nexts = adj[curr]
                    next_node = nexts[0] if nexts[0] != prev else nexts[1]
                    if next_node == 0 and _ < N - 1:
                        return None # Closed too early
                    if _ == N - 1 and next_node != 0:
                        return None # Didn't close to 0
                    cycle.append(next_node)
                    prev, curr = curr, next_node
                if cycle[0] == 0 and cycle[-1] == 0:
                    return cycle[:-1]
            return None
        else:
            return None
    except Exception:
        return None

def construct():
    start_time = time.time()
    max_time = 55.0
    
    # Binary search on S
    lo, hi = 1, 276
    best_cycle = None
    best_S = 276
    
    while lo <= hi and time.time() - start_time < max_time:
        mid = (lo + hi) // 2
        cycle = check_feasibility(mid)
        if cycle is not None:
            best_cycle = cycle
            best_S = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_cycle is None:
        # Fallback to a simple heuristic if MILP fails or times out
        # Just return a random permutation, score will be high but valid
        return list(range(N))
        
    return best_cycle
# EVOLVE-BLOCK-END
