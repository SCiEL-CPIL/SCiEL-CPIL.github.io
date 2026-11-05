"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_index = {}
for idx, (u, v) in enumerate(P['edge_order'], 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = idx

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Precompute max edge index
    max_idx = max(edge_index.values())
    
    # Binary search for the smallest K such that a Hamilton cycle exists
    # using only edges with index <= K
    
    # Helper: Check if a Hamilton cycle exists with max edge index <= K
    def exists_hamiltonian_cycle(K):
        # Build adjacency list of allowed edges
        adj = [[] for _ in range(N)]
        allowed = [[False] * N for _ in range(N)]
        
        for (u, v), idx in edge_index.items():
            if idx <= K:
                adj[u].append(v)
                adj[v].append(u)
                allowed[u][v] = True
                allowed[v][u] = True
        
        # Sort neighbors by some heuristic? 
        # For DFS, order doesn't matter for correctness, but for speed.
        # We can try to visit neighbors with fewer options first (Warnsdorff-like)
        
        # Pre-check: Every vertex must have degree >= 2
        for i in range(N):
            if len(adj[i]) < 2:
                return None
        
        # DFS
        # State: current node, visited mask, path
        # We fix start node as 0 to reduce symmetry
        start = 0
        visited_mask = 1 << start
        path = [start]
        
        # Pruning function
        def can_complete(u, mask):
            # If mask is full, check if u connects to start
            if mask == (1 << N) - 1:
                return allowed[u][start]
            
            # Check unused vertices
            unused_vertices = []
            for v in range(N):
                if not (mask & (1 << v)):
                    unused_vertices.append(v)
            
            # If no unused vertices, handled above
            
            # Pruning 1: Any unused vertex must have at least 2 allowed neighbors
            # in the context of the final cycle.
            # However, during construction, an unused vertex v needs to connect to:
            # 1. Two other vertices.
            # These can be:
            # - Other unused vertices
            # - The current end u
            # - The start vertex (only if v is the last one added)
            
            # A simpler and stronger pruning:
            # If there are > 1 unused vertices, every unused vertex v must have
            # at least one allowed edge to another unused vertex.
            # Why? Because the path from u to start must traverse all unused vertices.
            # The subgraph of unused vertices in the path must be connected (as a set of edges in the path).
            # Actually, the unused vertices form a path segment $u \to v_1 \to \dots \to v_k \to start$.
            # So $v_1$ connects to $u$, $v_k$ connects to $start$.
            # Intermediate $v_i$ connect to $v_{i-1}$ and $v_{i+1}$.
            # So every $v_i$ (except possibly $v_1, v_k$) must connect to another unused vertex.
            # $v_1$ must connect to $u$ and $v_2$ (if $k>1$) or $start$ (if $k=1$).
            # $v_k$ must connect to $v_{k-1}$ (if $k>1$) or $u$ (if $k=1$) and $start$.
            
            # Let's check:
            if len(unused_vertices) > 1:
                for v in unused_vertices:
                    has_unused_neighbor = False
                    for w in adj[v]:
                        if not (mask & (1 << w)):
                            has_unused_neighbor = True
                            break
                    if not has_unused_neighbor:
                        # v must connect to u and start?
                        # If v is the only one connecting to u and start, it must be the ONLY unused vertex.
                        # But we are in the case len > 1.
                        # So if v has no unused neighbors, it cannot be part of a path of length > 1 of unused vertices.
                        # It would have to be adjacent to u and start directly.
                        # But there are other unused vertices. They must be visited.
                        # The path is u -> ... -> v -> ... -> start.
                        # If v has no unused neighbors, it must be adjacent to u and start.
                        # This implies v is an endpoint of the unused segment.
                        # But there are other unused vertices. They must be in the segment.
                        # So the segment is u -> v -> ... -> start? No, v has no unused neighbors.
                        # So v cannot be in the middle.
                        # If v is at the end (next to u), then u->v->... but v has no unused neighbors, so v must go to start.
                        # Then the path is u->v->start. But there are other unused vertices. Impossible.
                        return False
            
            # If len(unused_vertices) == 1:
            # The single unused vertex v must be adjacent to u and start.
            elif len(unused_vertices) == 1:
                v = unused_vertices[0]
                if not (allowed[u][v] and allowed[v][start]):
                    return False
            
            return True
        
        def dfs(u, mask, path):
            if time.time() > deadline:
                return None
            
            if not can_complete(u, mask):
                return None
            
            if mask == (1 << N) - 1:
                if allowed[u][start]:
                    return path[:]
                return None
            
            # Get neighbors
            # Heuristic: Try neighbors that are "more constrained" first?
            # Or just iterate.
            for v in adj[u]:
                if not (mask & (1 << v)):
                    new_mask = mask | (1 << v)
                    path.append(v)
                    res = dfs(v, new_mask, path)
                    if res is not None:
                        return res
                    path.pop()
            
            return None
        
        result = dfs(start, visited_mask, path)
        return result

    low = 0
    high = max_idx
    best_cycle = None
    
    # We can also just iterate K from low to high if max_idx is small?
    # max_idx is 276. Binary search is ~8 steps.
    
    while low <= high:
        mid = (low + high) // 2
        cycle = exists_hamiltonian_cycle(mid)
        if cycle is not None:
            best_cycle = cycle
            high = mid - 1
        else:
            low = mid + 1
            
    if best_cycle is None:
        # Fallback to identity if something went wrong
        return list(range(N))
        
    return best_cycle

# EVOLVE-BLOCK-END
