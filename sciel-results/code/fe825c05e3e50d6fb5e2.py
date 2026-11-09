"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

N = 24

# Generate the pinned edge list with random.Random(614)
def _build_edge_order():
    rng = random.Random(614)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    rng.shuffle(edges)
    return edges

EDGE_ORDER = _build_edge_order()
EDGE_INDEX = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(u, v)] = idx
    EDGE_INDEX[(v, u)] = idx

def _edge_score(u, v):
    return EDGE_INDEX[(u, v)]

def construct():
    # We want to find a Hamiltonian cycle that minimizes the maximum edge arrival index.
    # This is equivalent to finding the smallest T such that there exists a Hamiltonian cycle
    # using only edges with arrival index <= T.
    
    # Strategy:
    # 1. Binary search on T.
    # 2. For a given T, check if a Hamiltonian cycle exists in the subgraph of edges with index <= T.
    # 3. Use DFS with Warnsdorff's rule (heuristic) to find the cycle efficiently.
    
    def find_cycle_with_max_idx(max_allowed):
        """Find a Hamiltonian cycle using only edges with index <= max_allowed."""
        # Build adjacency list for edges with index <= max_allowed
        adj = [[] for _ in range(N)]
        for u in range(N):
            for v in range(u + 1, N):
                if _edge_score(u, v) <= max_allowed:
                    adj[u].append(v)
                    adj[v].append(u)
        
        # Check if each vertex has degree >= 2 (necessary for Hamiltonian cycle)
        for u in range(N):
            if len(adj[u]) < 2:
                return None
        
        # Sort neighbors by degree (Warnsdorff's rule) to prune search space
        # We re-sort dynamically in the DFS step, but initial sort helps
        # Actually, dynamic sorting is better. We'll do that inside DFS.
        
        visited = [False] * N
        path = [0]
        visited[0] = True
        
        # Use iterative DFS or recursive with sys.setrecursionlimit
        sys.setrecursionlimit(10000)
        
        def dfs():
            if len(path) == N:
                # Check if we can close the cycle
                if 0 in adj[path[-1]]:
                    return True
                return False
            
            last = path[-1]
            
            # Get unvisited neighbors
            candidates = []
            for v in adj[last]:
                if not visited[v]:
                    # We cannot go to 0 until the very end
                    if v == 0:
                        continue
                    candidates.append(v)
            
            if not candidates:
                return False
            
            # Heuristic: Sort candidates by the number of their unvisited neighbors (Warnsdorff)
            # Choosing the vertex with fewest onward moves first often leads to a solution
            # or fails faster if a dead end is reached.
            def degree_key(v):
                deg = 0
                for w in adj[v]:
                    if not visited[w] and w != 0: # 0 is only valid at the end
                        deg += 1
                # Also consider if v is the last one, it must connect to 0
                # But for intermediate steps, just count unvisited neighbors
                return deg
            
            candidates.sort(key=degree_key)
            
            for v in candidates:
                visited[v] = True
                path.append(v)
                
                if dfs():
                    return True
                
                path.pop()
                visited[v] = False
            
            return False
        
        if dfs():
            return path[:]
        return None
    
    # Determine the range for binary search
    # The maximum possible index is the total number of edges
    total_edges = N * (N - 1) // 2
    hi = total_edges
    lo = N # Minimum number of edges in a cycle is N
    
    best_cycle = None
    best_max_idx = hi
    
    # Binary search for the smallest T where a Hamiltonian cycle exists
    while lo < hi:
        mid = (lo + hi) // 2
        cycle = find_cycle_with_max_idx(mid)
        if cycle is not None:
            hi = mid
            best_cycle = cycle
            best_max_idx = mid
        else:
            lo = mid + 1
    
    # If no cycle found in the binary search range (should not happen as hi is total edges),
    # or if best_cycle is None, we need a fallback.
    if best_cycle is None:
        # Fallback: Find any Hamiltonian cycle using all edges
        # This should always exist for N=24 complete graph
        # Use the DFS with max_allowed = total_edges
        best_cycle = find_cycle_with_max_idx(total_edges)
        if best_cycle is None:
             # Extremely unlikely, but return identity as last resort
             best_cycle = list(range(N))
             best_max_idx = max(_edge_score(best_cycle[i], best_cycle[i+1]) for i in range(N-1))
             best_max_idx = max(best_max_idx, _edge_score(best_cycle[-1], best_cycle[0]))
        else:
             best_max_idx = total_edges
    else:
        # Verify the max index of the found cycle is indeed <= best_max_idx
        # (It should be by construction, but good to be sure)
        actual_max = max(_edge_score(best_cycle[i], best_cycle[i+1]) for i in range(N-1))
        actual_max = max(actual_max, _edge_score(best_cycle[-1], best_cycle[0]))
        if actual_max < best_max_idx:
            best_max_idx = actual_max

    return best_cycle

# EVOLVE-BLOCK-END
