"""Find a Hamilton cycle with minimum maximum edge arrival index."""
import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGES = P['edge_order']

def _find_hamilton_cycle(adj, n, time_limit=5.0):
    """Find a Hamilton cycle in graph defined by adj (list of sets). Returns list of vertices or None."""
    start = time.time()
    # Start from vertex 0 to break symmetry
    path = [0]
    visited = [False] * n
    visited[0] = True
    
    def backtrack():
        if time.time() - start > time_limit:
            return False
        if len(path) == n:
            # Check if last vertex connects back to first
            if 0 in adj[path[-1]]:
                return True
            return False
        
        current = path[-1]
        # Get unvisited neighbors, sorted by degree for better pruning
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return False
        
        # Sort by degree (fewest options first) for better pruning
        neighbors.sort(key=lambda v: len([u for u in adj[v] if not visited[u] or u == path[0]]))
        
        for v in neighbors:
            # Pruning: check if v has at least one unvisited neighbor (or is the last vertex connecting to 0)
            unvisited_neighbors = [u for u in adj[v] if not visited[u]]
            if len(path) + 1 < n and not unvisited_neighbors:
                continue
            if len(path) + 1 == n and 0 not in adj[v]:
                continue
            
            # Pruning: check connectivity of unvisited vertices
            # This is expensive, so only do it occasionally
            if len(path) % 4 == 0 and len(path) > 2:
                unvisited = [u for u in range(n) if not visited[u]]
                if unvisited:
                    # BFS from first unvisited
                    seen = set()
                    stack = [unvisited[0]]
                    seen.add(unvisited[0])
                    while stack:
                        node = stack.pop()
                        for nb in adj[node]:
                            if nb in unvisited and nb not in seen:
                                seen.add(nb)
                                stack.append(nb)
                    if len(seen) != len(unvisited):
                        continue
            
            visited[v] = True
            path.append(v)
            if backtrack():
                return True
            path.pop()
            visited[v] = False
        
        return False
    
    if backtrack():
        return path
    return None

def _check_prefix(k, time_limit=5.0):
    """Check if first k edges contain a Hamilton cycle. Returns cycle or None."""
    adj = [set() for _ in range(N)]
    for i in range(min(k, len(EDGES))):
        u, v = EDGES[i]
        adj[u].add(v)
        adj[v].add(u)
    
    # Quick check: all vertices must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return None
    
    # Quick check: graph must be connected
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                count += 1
                stack.append(nb)
    if count != N:
        return None
    
    cycle = _find_hamilton_cycle(adj, N, time_limit=time_limit)
    return cycle

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    # Minimum edges needed for a Hamilton cycle is N
    best_cycle = None
    best_k = None
    
    # Try increasing k from N upward
    for k in range(N, len(EDGES) + 1):
        if time.time() > deadline - 10.0:
            break
        
        remaining_time = deadline - time.time() - 5.0
        if remaining_time < 0.5:
            break
        
        # Use less time for early checks, more for later
        tlim = min(remaining_time * 0.3, 8.0)
        if tlim < 0.1:
            tlim = 0.1
        
        cycle = _check_prefix(k, time_limit=tlim)
        if cycle is not None:
            best_cycle = cycle
            best_k = k
            break
    
    if best_cycle is not None:
        return best_cycle
    else:
        # Fallback: return a simple cycle
        return list(range(N))
# EVOLVE-BLOCK-END
