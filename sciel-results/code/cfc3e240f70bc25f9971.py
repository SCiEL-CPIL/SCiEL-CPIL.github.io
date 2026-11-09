"""Find the minimum prefix index for a Hamilton cycle in a fixed edge permutation."""
import time
import sys

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute adjacency lists for each prefix
def build_adjacency(prefix_len):
    """Build adjacency list for the first prefix_len edges."""
    adj = [set() for _ in range(N)]
    for i in range(prefix_len):
        u, v = EDGE_ORDER[i]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def has_hamiltonian_cycle(adj, deadline):
    """Check if the graph with adjacency list adj has a Hamiltonian cycle using DFS with pruning."""
    # Use vertex 0 as the start to break symmetry
    start = 0
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(N)]
    
    # If any vertex has degree < 2, no Hamiltonian cycle
    for i in range(N):
        if degrees[i] < 2:
            return False
    
    # DFS with backtracking
    # visited: bitmask of visited vertices
    # path: current path
    # We need to visit all N vertices and return to start
    
    # Use iterative approach with pruning
    # For speed, we use a recursive DFS with aggressive pruning
    
    sys.setrecursionlimit(10000)
    
    visited = 1 << start
    path = [start]
    remaining = N - 1
    
    # Precompute neighbor lists for faster iteration
    neigh = [list(adj[i]) for i in range(N)]
    
    # Sort neighbors by degree (least connected first) for better pruning
    for i in range(N):
        neigh[i].sort(key=lambda x: degrees[x])
    
    def dfs(current, visited_mask, path_len):
        nonlocal deadline
        if time.time() > deadline:
            return False
        
        if path_len == N:
            # Check if we can return to start
            return start in adj[current]
        
        # Pruning: check if all unvisited vertices are still reachable
        # Simple pruning: if the current vertex's unvisited neighbors
        # don't include any vertex that needs to be visited, fail
        
        for nxt in neigh[current]:
            if visited_mask & (1 << nxt):
                continue
            if path_len == N - 1:
                # Last vertex must connect back to start
                if nxt in adj[start]:
                    return True
                continue
            
            # Pruning: check if removing nxt would disconnect any unvisited vertex
            # This is expensive, so use a lighter check
            new_mask = visited_mask | (1 << nxt)
            if dfs(nxt, new_mask, path_len + 1):
                return True
        
        return False
    
    return dfs(start, visited, 1)

def find_min_prefix():
    """Binary search for the minimum prefix length that contains a Hamiltonian cycle."""
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    # First, verify that the full graph has a Hamiltonian cycle
    # (It should, since it's a complete graph)
    
    # Binary search
    lo, hi = 23, N * (N - 1) // 2  # Minimum edges for a cycle is N, max is all edges
    
    # Actually, minimum possible is N edges (the cycle itself), so lo = N
    lo = N
    hi = N * (N - 1) // 2
    
    # Check if hi works (should always work for complete graph)
    adj_hi = build_adjacency(hi)
    if not has_hamiltonian_cycle(adj_hi, deadline):
        # Shouldn't happen for complete graph, but handle gracefully
        return list(range(N))
    
    best = hi
    while lo < hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        adj_mid = build_adjacency(mid)
        if has_hamiltonian_cycle(adj_mid, deadline):
            best = mid
            hi = mid
        else:
            lo = mid + 1
    
    return best

def construct():
    """Find and return a Hamiltonian cycle that minimizes the maximum edge arrival index."""
    start_time = time.time()
    deadline = start_time + 80
    
    min_prefix = find_min_prefix()
    
    # Now we need to actually find a Hamiltonian cycle using only edges within the first min_prefix edges
    adj = build_adjacency(min_prefix)
    
    # Find the actual cycle
    sys.setrecursionlimit(10000)
    start = 0
    visited = 1 << start
    path = [start]
    remaining = N - 1
    
    neigh = [list(adj[i]) for i in range(N)]
    degrees = [len(adj[i]) for i in range(N)]
    for i in range(N):
        neigh[i].sort(key=lambda x: degrees[x])
    
    def dfs_find(current, visited_mask, path_list):
        nonlocal deadline
        if time.time() > deadline:
            return False
        
        if len(path_list) == N:
            return start in adj[current]
        
        for nxt in neigh[current]:
            if visited_mask & (1 << nxt):
                continue
            if len(path_list) == N - 1:
                if nxt in adj[start]:
                    path_list.append(nxt)
                    return True
                continue
            
            path_list.append(nxt)
            if dfs_find(nxt, visited_mask | (1 << nxt), path_list):
                return True
            path_list.pop()
        
        return False
    
    if dfs_find(start, visited, path):
        # path now contains the Hamiltonian cycle (without the closing edge back to start)
        # The permutation is path[0], path[1], ..., path[N-1]
        # The implicit closing edge is from path[N-1] to path[0]
        return path
    
    # Fallback: return identity permutation (shouldn't happen)
    return list(range(N))
