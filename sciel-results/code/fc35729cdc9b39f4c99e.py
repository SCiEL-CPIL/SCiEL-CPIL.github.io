"""Search for a Hamilton cycle minimizing the max edge arrival index in the pinned edge order."""
import time
import random
from itertools import combinations

# Build the edge order and position lookup
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# pos[(u,v)] = 1-based index where u < v
pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    if u > v:
        u, v = v, u
    pos[(u, v)] = i

def edge_pos(u, v):
    if u > v:
        u, v = v, u
    return pos[(u, v)]

def cycle_score(cycle):
    """Compute max edge arrival index for a Hamilton cycle."""
    n = len(cycle)
    max_pos = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        p = edge_pos(u, v)
        if p > max_pos:
            max_pos = p
    return max_pos

def has_hamilton_cycle_in_prefix(K):
    """Check if there's a Hamilton cycle using only edges with position <= K."""
    # Build adjacency: for each vertex, list of neighbors connected by edges with pos <= K
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER[:K], 1):
        adj[u].append(v)
        adj[v].append(u)
    
    # Check minimum degree >= 2
    for v in range(N):
        if len(adj[v]) < 2:
            return False
    
    # Backtracking with pruning
    # Fix vertex 0 as start to break symmetry
    start = 0
    visited = [False] * N
    visited[start] = True
    path = [start]
    
    def dfs():
        if len(path) == N:
            # Check closing edge
            last = path[-1]
            first = path[0]
            if last in adj[first]:
                return True
            return False
        
        current = path[-1]
        # Try neighbors, ordered by fewest remaining options (MRV heuristic)
        candidates = []
        for nb in adj[current]:
            if not visited[nb]:
                # Count unvisited neighbors of nb
                cnt = sum(1 for x in adj[nb] if not visited[x])
                candidates.append((cnt, nb))
        
        candidates.sort()
        
        for cnt, nb in candidates:
            if nb == start and len(path) < N - 1:
                continue  # Can't go back to start until the end
            
            visited[nb] = True
            path.append(nb)
            
            if dfs():
                return True
            
            path.pop()
            visited[nb] = False
        
        return False
    
    return dfs()

def find_cycle_in_prefix(K):
    """Find a specific Hamilton cycle using only edges with position <= K."""
    adj = [[] for _ in range(N)]
    for i, (u, v) in enumerate(EDGE_ORDER[:K], 1):
        adj[u].append(v)
        adj[v].append(u)
    
    for v in range(N):
        if len(adj[v]) < 2:
            return None
    
    start = 0
    visited = [False] * N
    visited[start] = True
    path = [start]
    
    def dfs():
        if len(path) == N:
            last = path[-1]
            first = path[0]
            if last in adj[first]:
                return list(path)
            return None
        
        current = path[-1]
        candidates = []
        for nb in adj[current]:
            if not visited[nb]:
                cnt = sum(1 for x in adj[nb] if not visited[x])
                candidates.append((cnt, nb))
        
        candidates.sort()
        
        for cnt, nb in candidates:
            if nb == start and len(path) < N - 1:
                continue
            
            visited[nb] = True
            path.append(nb)
            
            result = dfs()
            if result is not None:
                return result
            
            path.pop()
            visited[nb] = False
        
        return None
    
    return dfs()

def greedy_cycle():
    """Greedy construction: always pick the earliest available edge."""
    used_edges = set()
    cycle = [0]
    current = 0
    remaining = set(range(1, N))
    
    for _ in range(N - 1):
        best_nb = None
        best_pos = float('inf')
        for nb in remaining:
            p = edge_pos(current, nb)
            if p < best_pos:
                best_pos = p
                best_nb = nb
        cycle.append(best_nb)
        remaining.remove(best_nb)
        current = best_nb
    
    # Check closing edge
    return cycle

def local_search(cycle, max_iters=10000):
    """Improve cycle via 2-opt style moves."""
    best_score = cycle_score(cycle)
    best_cycle = list(cycle)
    current_cycle = list(cycle)
    
    for _ in range(max_iters):
        improved = False
        n = len(current_cycle)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # Same cycle reversed
                # Try reversing segment i..j
                new_cycle = current_cycle[:i] + current_cycle[i:j+1][::-1] + current_cycle[j+1:]
                s = cycle_score(new_cycle)
                if s < best_score:
                    best_score = s
                    best_cycle = new_cycle
                    current_cycle = new_cycle
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_cycle

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Start with greedy
    best_cycle = greedy_cycle()
    best_score = cycle_score(best_cycle)
    
    # Binary search for minimum K using backtracking
    lo, hi = 24, N * (N - 1) // 2  # 276
    # Find smallest K where a Hamilton cycle exists
    while lo < hi and time.time() < deadline - 30:
        mid = (lo + hi) // 2
        if has_hamilton_cycle_in_prefix(mid):
            hi = mid
        else:
            lo = mid + 1
    
    min_K = lo
    
    # Try to find the actual cycle at min_K
    if time.time() < deadline - 5:
        cycle = find_cycle_in_prefix(min_K)
        if cycle is not None:
            best_cycle = cycle
            best_score = cycle_score(best_cycle)
    
    # Local search to improve
    if time.time() < deadline - 5:
        best_cycle = local_search(best_cycle)
        best_score = cycle_score(best_cycle)
    
    # Additional random restarts if time permits
    rng = random.Random(614)
    while time.time() < deadline - 5:
        # Random cycle
        rand_cycle = list(range(N))
        rng.shuffle(rand_cycle)
        rand_cycle = local_search(rand_cycle, max_iters=500)
        s = cycle_score(rand_cycle)
        if s < best_score:
            best_score = s
            best_cycle = rand_cycle
    
    return best_cycle
