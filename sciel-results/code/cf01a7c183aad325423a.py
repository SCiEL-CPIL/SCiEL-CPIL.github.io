import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_list = P['edge_order']
edge_to_index = {}
for i, e in enumerate(edge_list):
    u, v = e
    if u > v:
        u, v = v, u
    edge_to_index[(u, v)] = i + 1  # 1-based index

def get_cycle_from_solution(y, N):
    # y is a numpy array of shape (N, N)
    # Reconstruct cycle
    cycle = [0]
    current = 0
    for _ in range(N - 1):
        next_node = None
        for j in range(N):
            if j != current and y[current, j] > 0.5:
                next_node = j
                break
        if next_node is None:
            return None
        cycle.append(next_node)
        current = next_node
    return cycle

def solve_for_K(K):
    # Build ILP for edges with index <= K
    # Variables:
    # y[i][j] for i != j, if edge {i,j} exists in E_K
    # u[i] for i in 1..N-1
    
    # Determine which edges are allowed
    allowed_edges = []
    for (u, v), idx in edge_to_index.items():
        if idx <= K:
            allowed_edges.append((u, v))
    
    # If number of allowed edges is less than N, impossible
    if len(allowed_edges) < N:
        return None

    # Create mapping from (i, j) to variable index
    var_map = {}
    var_idx = 0
    for u in range(N):
        for v in range(N):
            if u == v:
                continue
            if (u, v) in edge_to_index and edge_to_index[(u, v)] <= K:
                var_map[(u, v)] = var_idx
                var_idx += 1
            elif (v, u) in edge_to_index and edge_to_index[(v, u)] <= K:
                # We need both directions if the undirected edge is present
                # Wait, my var_map logic above only adds (u,v) if (u,v) is in edge_to_index.
                # edge_to_index keys are sorted (min, max).
                # So I should check if the undirected edge is allowed.
                pass
            
    # Let's redo variable creation properly
    var_map = {}
    var_idx = 0
    for u in range(N):
        for v in range(N):
            if u == v:
                continue
            key = (min(u, v), max(u, v))
            if key in edge_to_index and edge_to_index[key] <= K:
                var_map[(u, v)] = var_idx
                var_idx += 1
                
    M = var_idx
    if M == 0:
        return None

    # Total variables: M (for y) + N-1 (for u)
    n_vars = M + N - 1
    
    # Objective: minimize 0
    c = np.zeros(n_vars)
    
    # Constraints
    A_rows = []
    lb_list = []
    ub_list = []
    
    # 1. Flow conservation: out-degree = 1, in-degree = 1
    for i in range(N):
        # Out-degree
        row = np.zeros(n_vars)
        for j in range(N):
            if i == j: continue
            if (i, j) in var_map:
                row[var_map[(i, j)]] = 1
        A_rows.append(row)
        lb_list.append(1)
        ub_list.append(1)
        
        # In-degree
        row = np.zeros(n_vars)
        for j in range(N):
            if i == j: continue
            if (j, i) in var_map:
                row[var_map[(j, i)]] = 1
        A_rows.append(row)
        lb_list.append(1)
        ub_list.append(1)
        
    # 2. MTZ constraints: u_i - u_j + N * y_ij <= N - 1
    # u_0 is fixed to 0. So we don't have a variable for u_0.
    # If i=0, u_0=0. If j=0, u_j is a variable.
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if (i, j) not in var_map:
                continue
            
            row = np.zeros(n_vars)
            y_idx = var_map[(i, j)]
            row[y_idx] = N
            
            if i != 0:
                u_i_idx = M + i - 1
                row[u_i_idx] = 1
            # else u_i is 0
            
            if j != 0:
                u_j_idx = M + j - 1
                row[u_j_idx] = -1
            # else u_j is 0
            
            A_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(N - 1)
            
    A = np.array(A_rows)
    lb = np.array(lb_list)
    ub = np.array(ub_list)
    
    # Bounds
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    # y variables are binary (0 or 1)
    # u variables are continuous [0, N-1]
    integrality = np.zeros(n_vars)
    integrality[:M] = 1  # y variables are integer
    integrality[M:] = 0  # u variables are continuous
    
    # Adjust bounds for u variables
    bounds_ub = bounds.ub.copy()
    bounds_ub[M:] = N - 1
    
    constraints = LinearConstraint(A, lb, ub)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(lb=bounds.lb, ub=bounds_ub))
    except Exception as e:
        return None
        
    if res.status != 0:
        return None
        
    y_sol = res.x[:M]
    # Reconstruct y matrix
    Y = np.zeros((N, N))
    for (i, j), idx in var_map.items():
        Y[i, j] = y_sol[idx]
        
    cycle = get_cycle_from_solution(Y, N)
    if cycle is not None:
        return cycle
    return None

def construct():
    start_time = time.time()
    deadline = 90  # seconds
    
    # 1. Get initial upper bound using a greedy heuristic
    # Greedy: Start at 0, always go to the unvisited neighbor with the smallest edge index
    def greedy_cycle():
        visited = [False] * N
        visited[0] = True
        current = 0
        cycle = [0]
        max_idx = 0
        
        for _ in range(N - 1):
            best_next = -1
            best_idx = float('inf')
            for j in range(N):
                if not visited[j]:
                    key = (min(current, j), max(current, j))
                    if key in edge_to_index:
                        idx = edge_to_index[key]
                        if idx < best_idx:
                            best_idx = idx
                            best_next = j
            if best_next == -1:
                return None
            visited[best_next] = True
            cycle.append(best_next)
            current = best_next
            max_idx = max(max_idx, best_idx)
            
        # Check closing edge
        key = (min(0, current), max(0, current))
        if key in edge_to_index:
            max_idx = max(max_idx, edge_to_index[key])
        else:
            return None
            
        return cycle, max_idx

    greedy_res = greedy_cycle()
    if greedy_res is None:
        # Fallback to identity if greedy fails (should not happen for complete graph)
        return list(range(N))
        
    best_cycle, K_best = greedy_res
    K_min = N  # A cycle must have at least N edges, so max index >= N? Not necessarily, but K must be at least the N-th smallest edge index in the graph? No, K is the max index in the cycle.
    # Actually, K_min can be 1. But we know K >= N is a safe lower bound for the *value* of the max index?
    # The indices are 1..276.
    # Let's just start from 1.
    
    # Binary search for minimal K
    lo = 1
    hi = K_best
    
    last_feasible_cycle = best_cycle
    last_feasible_K = K_best
    
    while lo < hi:
        if time.time() - start_time > deadline - 5:
            break
            
        mid = (lo + hi) // 2
        res = solve_for_K(mid)
        if res is not None:
            last_feasible_cycle = res
            last_feasible_K = mid
            hi = mid
        else:
            lo = mid + 1
            
    # Verify the found cycle
    # Calculate its score
    max_idx = 0
    for i in range(N):
        u = last_feasible_cycle[i]
        v = last_feasible_cycle[(i + 1) % N]
        key = (min(u, v), max(u, v))
        if key in edge_to_index:
            max_idx = max(max_idx, edge_to_index[key])
            
    # It's possible that the binary search found a K, but the cycle returned has a max index > K?
    # No, solve_for_K only allows edges with index <= K. So the cycle's max index is <= K.
    
    return last_feasible_cycle
# EVOLVE-BLOCK-END
