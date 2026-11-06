"""Find minimum-threshold Hamiltonian cycle in edge process."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Precompute edge position (1-based index)
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    edge_pos[(u, v)] = i + 1
    edge_pos[(v, u)] = i + 1


def _build_adj_matrix(max_pos):
    """Build adjacency matrix for edges with position <= max_pos."""
    adj = [[False] * N for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER):
        if i + 1 > max_pos:
            break
        adj[u][v] = True
        adj[v][u] = True
    return adj


def _check_connected(adj):
    """Check if graph is connected."""
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        u = stack.pop()
        for v in range(N):
            if adj[u][v] and not visited[v]:
                visited[v] = True
                count += 1
                stack.append(v)
    return count == N


def _find_hamiltonian_cycle(adj, deadline, rng):
    """
    Find a Hamiltonian cycle in the graph defined by adj matrix.
    Uses randomized DFS with backtracking and pruning.
    Returns list of vertices forming the cycle, or None.
    """
    n = N
    degrees = [sum(adj[i]) for i in range(n)]
    
    # Check min degree
    if any(d < 2 for d in degrees):
        return None
    
    # Check connectivity
    if not _check_connected(adj):
        return None
    
    # Use backtracking with randomized neighbor ordering
    # Try multiple starting points and orderings
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Randomize the search by shuffling neighbor order
        start = 0  # Always start from 0 for consistency
        
        path = [start]
        used = [False] * n
        used[start] = True
        
        # Build neighbor lists with randomization
        neighbors = []
        for i in range(n):
            nbrs = [j for j in range(n) if adj[i][j]]
            rng.shuffle(nbrs)
            neighbors.append(nbrs)
        
        def dfs(current, depth):
            if time.time() > deadline:
                return False
            
            if depth == n:
                return adj[current][start]
            
            # Get unvisited neighbors
            unvisited = [w for w in neighbors[current] if not used[w]]
            
            # Pruning: for each unvisited vertex, check if it has enough 
            # unvisited neighbors to be reachable
            # Also check: the last vertex must connect back to start
            
            # Sort by degree (fewest options first)
            unvisited.sort(key=lambda w: sum(1 for x in neighbors[w] if not used[x]))
            
            for w in unvisited:
                # Pruning: check if w can still connect to all remaining unvisited
                remaining = n - depth - 1
                if remaining > 0:
                    # Count how many unvisited vertices w connects to
                    conn_count = sum(1 for x in neighbors[w] if not used[x])
                    if conn_count < remaining:
                        continue
                
                # Additional pruning: check if any unvisited vertex becomes isolated
                # This is expensive; skip for now
                
                used[w] = True
                path.append(w)
                if dfs(w, depth + 1):
                    return True
                path.pop()
                used[w] = False
            
            return False
        
        if dfs(start, 1):
            return path[:]
    
    return None


def _cycle_score(cycle):
    """Compute the max edge position in the cycle."""
    n = len(cycle)
    return max(edge_pos[(cycle[i], cycle[(i+1) % n])] for i in range(n))


def _two_opt_improve(cycle, edge_pos, deadline):
    """Improve cycle using 2-opt to reduce max edge position."""
    if cycle is None:
        return None
    
    best_cycle = list(cycle)
    best_score = _cycle_score(best_cycle)
    n = len(best_cycle)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+2, n):  # j > i+1 to avoid trivial
                if time.time() > deadline:
                    break
                if i == 0 and j == n-1:
                    continue  # Would just reverse the whole cycle
                
                # Try reversing segment i+1..j
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = _cycle_score(new_cycle)
                if new_score < best_score:
                    best_score = new_score
                    best_cycle = new_cycle
                    improved = True
    
    return best_cycle


def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    rng = random.Random(614)
    
    # Strategy: Binary search for minimum K, then improve the cycle
    # First, find a feasible upper bound
    
    best_cycle = None
    best_K = len(EDGE_ORDER)
    
    # Try to find a cycle at various K values
    # Start with a reasonable guess and adjust
    
    # First, try K = 60, 80, 100, ... to find a feasible one
    test_Ks = [48, 60, 72, 84, 96, 120, 150, 200, 250, 300, 360]
    for K in test_Ks:
        if K > len(EDGE_ORDER):
            break
        if time.time() > deadline - 15:
            break
        adj = _build_adj_matrix(K)
        cycle = _find_hamiltonian_cycle(adj, deadline - 10, rng)
        if cycle is not None:
            best_K = K
            best_cycle = cycle
            break
    
    if best_cycle is None:
        # Try from the top down
        for K in range(min(len(EDGE_ORDER), 360), 47, -10):
            if time.time() > deadline - 10:
                break
            adj = _build_adj_matrix(K)
            cycle = _find_hamiltonian_cycle(adj, deadline - 5, rng)
            if cycle is not None:
                best_K = K
                best_cycle = cycle
                break
    
    if best_cycle is None:
        # Fallback: return identity
        return list(range(N))
    
    # Binary search to narrow down K
    lo, hi = max(24, best_K - 30), best_K
    while lo < hi and time.time() < deadline - 20:
        mid = (lo + hi) // 2
        adj = _build_adj_matrix(mid)
        cycle = _find_hamiltonian_cycle(adj, deadline - 15, rng)
        if cycle is not None:
            hi = mid
            best_cycle = cycle
            best_K = mid
        else:
            lo = mid + 1
    
    # Improve the cycle using 2-opt
    if time.time() < deadline - 5:
        improved = _two_opt_improve(best_cycle, edge_pos, deadline - 2)
        if improved is not None:
            best_cycle = improved
    
    return best_cycle
# EVOLVE-BLOCK-END
