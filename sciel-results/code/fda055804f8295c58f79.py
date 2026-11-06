"""Find Hamilton cycle with minimal max edge arrival index using MILP + local search."""
import time
import sys
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

# Precompute adjacency masks for each K (for quick feasibility checks)
def precompute_masks():
    masks_list = []
    for K in range(N, len(EDGE_ORDER) + 1):
        masks = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        masks_list.append(masks)
    return masks_list

MASKS = precompute_masks()

def check_connectivity(masks):
    visited = [False] * N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        neighbors = masks[u]
        while neighbors:
            lsb = neighbors & (-neighbors)
            nb = lsb.bit_length() - 1
            neighbors ^= lsb
            if not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    return count == N

def milp_find_cycle(K, deadline):
    """Use MILP to find a Hamilton cycle in the graph with edges up to position K."""
    # Build adjacency list for edges with position <= K
    adj = [[] for _ in range(N)]
    for i in range(K):
        u, v = EDGE_ORDER[i]
        adj[u].append(v)
        adj[v].append(u)
    
    # Check basic feasibility: all vertices must have degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    if not check_connectivity(MASKS[K - N]):
        return None
    
    # MILP formulation:
    # Variables: x_u_v for each undirected edge (u,v) with u < v, x_u_v in {0,1}
    # Objective: minimize sum x_u_v (we just want any Hamilton cycle)
    # Constraints:
    # 1. For each vertex v: sum of x over edges incident to v = 2
    # 2. Subtour elimination: for each subset S of vertices with 2 <= |S| <= N-2,
    #    sum of x over edges within S <= |S| - 1
    #    (This is exponential, so we use a simplified version or rely on solver)
    
    # For N=24, the subtour elimination constraints are too many to enumerate.
    # Instead, we use a different approach: use the MILP to find a 2-factor first,
    # then check if it's a single cycle.
    
    # Build edge list for the subgraph
    edges_in_subgraph = []
    for i in range(K):
        u, v = EDGE_ORDER[i]
        if u < v:
            edges_in_subgraph.append((u, v))
    
    m = len(edges_in_subgraph)
    if m < N:
        return None
    
    # Decision variables: x_j for j = 0..m-1
    c = np.ones(m)  # minimize number of edges selected
    
    # Degree constraints: for each vertex, exactly 2 edges selected
    # A matrix with 2N rows (one per vertex-direction) and m columns
    # Actually, we need N constraints (one per vertex)
    A = np.zeros((N, m))
    for j, (u, v) in enumerate(edges_in_subgraph):
        A[u, j] = 1
        A[v, j] = 1
    
    # Constraints: A @ x = 2 for all vertices
    # LinearConstraint(A, 2, 2)
    
    # For subtour elimination, we add a few key constraints:
    # For each pair of vertices, ensure they can't form a 2-cycle (trivially satisfied since no multi-edges)
    # For small subsets, we can add some constraints
    
    # Let's add constraints for subsets of size 3, 4, 5 (sampled)
    # For a subset S of size k, sum of x over edges within S <= k - 1
    
    # This is still complex. Let's try a simpler approach:
    # Use the MILP without subtour elimination, get a 2-factor, then check if it's Hamiltonian.
    # If not, add subtour constraints iteratively.
    
    constraints = [LinearConstraint(A, 2, 2)]
    
    # Add some subtour elimination constraints for small subsets
    # For each vertex v, and each pair of neighbors u, w of v,
    # we can add: x_{u,v} + x_{v,w} + sum of edges in {u,v,w} \ {(u,v),(v,w)} <= 2
    # This prevents triangles from being isolated
    
    # Actually, let's just solve without subtour elimination first and see
    # if we get a single cycle. If not, we'll need to add constraints.
    
    integrality = np.ones(m)
    bounds = Bounds(0, 1)
    
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    except Exception:
        return None
    
    if res.status != 0:
        return None
    
    x = res.x
    selected_edges = []
    for j, (u, v) in enumerate(edges_in_subgraph):
        if x[j] > 0.5:
            selected_edges.append((u, v))
    
    # Build the cycle from selected edges
    if len(selected_edges) != N:
        return None
    
    # Check if it forms a single cycle
    adj_sel = [[] for _ in range(N)]
    for u, v in selected_edges:
        adj_sel[u].append(v)
        adj_sel[v].append(u)
    
    for v in range(N):
        if len(adj_sel[v]) != 2:
            return None
    
    # Traverse the cycle
    visited = set()
    path = [0]
    visited.add(0)
    curr = 0
    for _ in range(N - 1):
        nxts = adj_sel[curr]
        if not nxts:
            return None
        nxt = nxts[0] if nxts[0] not in visited else (nxts[1] if len(nxts) > 1 else None)
        if nxt is None or nxt in visited:
            # Check if we've completed the cycle
            if len(path) == N:
                break
            return None
        path.append(nxt)
        visited.add(nxt)
        curr = nxt
    
    if len(path) != N:
        return None
    
    # Check if the last vertex connects back to 0
    if 0 not in adj_sel[curr]:
        return None
    
    return path


def dfs_find_cycle(masks, deadline):
    """DFS with Warnsdorff heuristic to find Hamilton cycle."""
    sys.setrecursionlimit(10000)
    
    def dfs(v, visited_mask, path):
        if len(path) == N:
            if masks[v] & (1 << 0):
                return path[:]
            return None
        
        available = masks[v] & ~visited_mask
        if available == 0:
            return None
        
        neighbors = []
        temp = available
        while temp:
            lsb = temp & (-temp)
            nb = lsb.bit_length() - 1
            temp ^= lsb
            if len(path) == N - 1:
                onward = 1 if (masks[nb] & (1 << 0)) else 0
            else:
                onward = bin(masks[nb] & ~visited_mask & ~(1 << v)).count('1')
            neighbors.append((onward, nb))
        
        neighbors.sort(key=lambda x: x[0])
        
        from collections import defaultdict
        deg_map = defaultdict(list)
        for deg, nb in neighbors:
            deg_map[deg].append(nb)
        
        for deg in sorted(deg_map.keys()):
            nbs = deg_map[deg]
            random.shuffle(nbs)
            for nb in nbs:
                result = dfs(nb, visited_mask | (1 << nb), path + [nb])
                if result is not None:
                    return result
        return None
    
    for attempt in range(20):
        if time.time() > deadline - 2:
            break
        rng = random.Random(attempt * 7919 + 12345)
        cycle = dfs(0, 1 << 0, [0])
        if cycle is not None:
            return cycle
    
    return None


def two_opt_polish(cycle, edge_pos, deadline):
    """Apply 2-opt to minimize max edge position."""
    if cycle is None:
        return None
    
    def cycle_score(cyc):
        max_pos = 0
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            pos = edge_pos[(u, v)]
            if pos > max_pos:
                max_pos = pos
        return max_pos
    
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        n = len(best_cycle)
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < best_score:
                    best_cycle = new_cycle
                    best_score = new_score
                    improved = True
    
    return best_cycle


def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    best_cycle = None
    best_K = len(EDGE_ORDER)
    
    # Binary search for minimum K
    lo, hi = N, len(EDGE_ORDER)
    
    while lo <= hi and time.time() < deadline - 20:
        mid = (lo + hi) // 2
        masks = MASKS[mid - N]
        
        valid = True
        for v in range(N):
            if bin(masks[v]).count('1') < 2:
                valid = False
                break
        if not valid:
            lo = mid + 1
            continue
        
        if not check_connectivity(masks):
            lo = mid + 1
            continue
        
        # Try MILP first
        cycle = milp_find_cycle(mid, deadline)
        
        if cycle is None:
            # Fall back to DFS
            cycle = dfs_find_cycle(masks, deadline)
        
        if cycle is not None:
            best_cycle = cycle
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Polish with 2-opt
    if best_cycle is not None and time.time() < deadline - 5:
        polished = two_opt_polish(best_cycle, edge_pos, deadline)
        if polished is not None:
            best_cycle = polished
    
    # Fallback: try higher K values
    if best_cycle is None:
        for K in range(N, len(EDGE_ORDER) + 1, 5):
            if time.time() > deadline - 5:
                break
            masks = MASKS[K - N]
            
            valid = True
            for v in range(N):
                if bin(masks[v]).count('1') < 2:
                    valid = False
                    break
            if not valid:
                continue
            
            if not check_connectivity(masks):
                continue
            
            cycle = milp_find_cycle(K, deadline)
            if cycle is None:
                cycle = dfs_find_cycle(masks, deadline)
            if cycle is not None:
                best_cycle = cycle
                break
    
    if best_cycle is not None:
        return best_cycle
    
    return list(range(N))
