"""Optimized Hamiltonian Cycle search using Warnsdorff's rule and bitmask DP/DFS."""
import time
import sys

sys.setrecursionlimit(100000)

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)

# Precompute edge indices for scoring
EDGE_IDX = {}
for i, (a, b) in enumerate(EDGE_ORDER):
    EDGE_IDX[(min(a, b), max(a, b))] = i + 1

def _build_adj(K):
    adj = [0] * N
    for i in range(K):
        a, b = EDGE_ORDER[i]
        adj[a] |= 1 << b
        adj[b] |= 1 << a
    return adj

def _find_cycle(adj):
    """
    Finds a Hamiltonian cycle in the graph defined by adj bitmasks.
    Uses DFS with Warnsdorff's rule and connectivity pruning.
    Returns a list of vertices representing the cycle, or None.
    """
    start = 0
    start_bit = 1 << start
    full_mask = (1 << N) - 1
    
    # We need to visit all vertices and return to start.
    # Fix start=0. The last vertex must be a neighbor of 0.
    # We can try all possible end vertices (neighbors of 0) or just let DFS find it.
    # To be fast, we fix the end vertex to the neighbor of 0 with the smallest index?
    # No, that might be too restrictive if that edge is the only one but the path doesn't exist.
    # However, for a dense graph, any neighbor works.
    # Let's just do a standard DFS from 0, ensuring we don't close the loop until the end.
    
    path = [start]
    visited = start_bit
    
    def dfs():
        if len(path) == N:
            # Check if last vertex connects to start
            last = path[-1]
            if adj[last] & start_bit:
                return True
            return False
        
        cur = path[-1]
        # Candidates: unvisited neighbors of cur
        cands = adj[cur] & ~visited
        
        # Heuristic: Sort candidates by the number of their unvisited neighbors (Warnsdorff)
        # Fewest unvisited neighbors first.
        cand_list = []
        temp_cands = cands
        while temp_cands:
            lsb = temp_cands & -temp_cands
            temp_cands &= ~lsb
            u = lsb.bit_length() - 1
            # Count unvisited neighbors of u
            # Note: we can't just use adj[u] & ~visited because we might close the loop early?
            # No, we only close at the end.
            # However, if u is the last vertex to be visited, it must connect to start.
            # If len(path) == N-1, u must connect to start.
            if len(path) == N - 1:
                if not (adj[u] & start_bit):
                    continue
            deg = (adj[u] & ~visited).bit_count()
            cand_list.append((deg, u))
        
        if not cand_list:
            return False
            
        cand_list.sort()
        
        for deg, u in cand_list:
            # Prune: Check connectivity of remaining vertices
            # The unvisited vertices (including u) must form a connected component 
            # that includes u? No, u is being added.
            # The set of unvisited vertices AFTER adding u must be connected 
            # to the current path? 
            # Simpler prune: The set of unvisited vertices (before adding u) 
            # must be connected to u?
            # Actually, the standard prune is: 
            # Let R be the set of unvisited vertices.
            # If we pick u, the remaining set is R \ {u}.
            # R \ {u} must be connected to u (via edges in the graph) 
            # OR if R \ {u} is empty, it's fine.
            # Also, if R \ {u} is not empty, it must be connected.
            
            new_visited = visited | (1 << u)
            remaining = full_mask & ~new_visited
            
            if remaining:
                # Check if 'remaining' is connected to u
                # BFS from u within (remaining | {u})?
                # Actually, we just need to ensure that the subgraph induced by 
                # (remaining | {u}) is connected.
                # Since u is connected to cur (in path), and we are adding u,
                # we need 'remaining' to be connected to u.
                
                # Fast check: 
                # 1. Is 'remaining' connected?
                # 2. Does 'remaining' have an edge to u?
                
                # Let's do a BFS from u in the graph induced by (remaining | {u}).
                # If we can reach all vertices in 'remaining', then it's connected.
                
                comp = 1 << u
                frontier = 1 << u
                allowed = remaining | (1 << u)
                
                while frontier:
                    lsb = frontier & -frontier
                    frontier &= ~lsb
                    v_idx = lsb.bit_length() - 1
                    # Neighbors in allowed
                    nb = adj[v_idx] & allowed & ~comp
                    comp |= nb
                    frontier |= nb
                
                if comp != allowed:
                    continue # Prune: disconnected
                
            path.append(u)
            if dfs():
                return True
            path.pop()
            
        return False

    if dfs():
        return path
    return None

def _score_of(cycle):
    if not cycle:
        return M + 1
    n = len(cycle)
    worst = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        key = (min(a, b), max(a, b))
        val = EDGE_IDX.get(key, M + 1)
        if val > worst:
            worst = val
    return worst

def construct():
    deadline = time.time() + 85.0
    
    # Binary search for the smallest K
    lo, hi = N, M
    best_cycle = None
    
    # Ensure hi is feasible (complete graph always has cycle)
    adj_hi = _build_adj(M)
    cyc = _find_cycle(adj_hi)
    if cyc is None:
        return list(range(N))
    best_cycle = cyc
    best_score = _score_of(cyc)
    hi = M
    
    # Binary search
    while lo < hi and time.time() < deadline - 5.0:
        mid = (lo + hi) // 2
        if mid < N:
            lo = mid + 1
            continue
            
        adj_mid = _build_adj(mid)
        cyc = _find_cycle(adj_mid)
        
        if cyc is not None:
            hi = mid
            best_cycle = cyc
            # We don't strictly need to update best_score here if we track hi,
            # but it's good for final verification.
        else:
            lo = mid + 1
            
    # Final check at hi
    if time.time() < deadline - 2.0:
        adj_final = _build_adj(hi)
        cyc = _find_cycle(adj_final)
        if cyc is not None:
            best_cycle = cyc
            
    return best_cycle if best_cycle is not None else list(range(N))
# EVOLVE-BLOCK-END
