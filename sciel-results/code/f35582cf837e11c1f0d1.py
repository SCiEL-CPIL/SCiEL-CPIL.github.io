import time
import heapq
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

n = P['n']
edge_order = P['edge_order']
m = len(edge_order)

# Arrival index (0-based) for each unordered edge
arrival = {}
for idx, (u, v) in enumerate(edge_order):
    arrival[(min(u, v), max(u, v))] = idx

def edge_arrival(u, v):
    if u > v:
        u, v = v, u
    return arrival[(u, v)]

def cycle_score(cycle):
    s = 0
    L = len(cycle)
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        ar = edge_arrival(a, b) + 1
        if ar > s:
            s = ar
    return s

def is_hamiltonian(cycle):
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for v in range(n):
        if v not in set(cycle):
            return False
    return True

# Precompute adjacency with arrival costs for fast lookup
adj_arr = [[] for _ in range(n)]
for idx, (u, v) in enumerate(edge_order):
    adj_arr[u].append((idx, v))
    adj_arr[v].append((idx, u))
for v in range(n):
    adj_arr[v].sort()

def dijkstra_bottleneck(start, deadline, initial_upper_bound=None):
    """Find the minimum-bottleneck Hamiltonian cycle using Dijkstra on (last, mask) states."""
    full_mask = (1 << n) - 1
    dist = {}
    pq = [(0, start, 1 << start)]
    dist[(start, 1 << start)] = 0
    
    best_cycle = None
    best_bottleneck = initial_upper_bound if initial_upper_bound is not None else m + 1
    
    parent = {}
    
    steps = 0
    while pq:
        if time.time() > deadline:
            break
        cost, last, mask = heapq.heappop(pq)
        
        if dist.get((last, mask), m + 1) < cost:
            continue
        
        if mask == full_mask:
            close_cost = edge_arrival(last, start)
            total_bottleneck = max(cost, close_cost)
            if total_bottleneck < best_bottleneck:
                best_bottleneck = total_bottleneck
                path = []
                cur = (last, mask)
                while cur is not None:
                    path.append(cur[0])
                    cur = parent.get(cur)
                path.reverse()
                best_cycle = path
            if cost >= best_bottleneck:
                continue
        
        for arr_idx, nb in adj_arr[last]:
            nb_bit = 1 << nb
            if mask & nb_bit:
                continue
            new_mask = mask | nb_bit
            new_cost = max(cost, arr_idx)
            
            if best_cycle is not None and new_cost >= best_bottleneck:
                continue
            
            key = (nb, new_mask)
            if new_cost < dist.get(key, m + 1):
                dist[key] = new_cost
                parent[key] = (last, mask)
                heapq.heappush(pq, (new_cost, nb, new_mask))
        
        steps += 1
        if steps % 50000 == 0 and time.time() > deadline:
            break
    
    return best_cycle, (best_bottleneck + 1 if best_cycle else m + 1)

def find_hc_prefix_dfs(K, deadline, start=0):
    """Find Hamilton cycle using only edges with arrival <= K (0-based)."""
    adj = [set() for _ in range(n)]
    for idx in range(K + 1):
        u, v = edge_order[idx]
        adj[u].add(v)
        adj[v].add(u)
    for v in range(n):
        if len(adj[v]) < 2:
            return None
    adj_list = []
    for v in range(n):
        nb = list(adj[v])
        nb.sort(key=lambda x: edge_arrival(v, x))
        adj_list.append(nb)
    
    visited = [False] * n
    path = [start]
    visited[start] = True
    found = [None]
    unvisited_mask = ((1 << n) - 1) ^ (1 << start)
    
    def connected_component_ok(cur):
        seen = 1 << cur
        dq = deque([cur])
        while dq:
            u = dq.popleft()
            for w in adj_list[u]:
                bit = 1 << w
                if (unvisited_mask & bit) and not (seen & bit):
                    seen |= bit
                    dq.append(w)
        return (unvisited_mask | (1 << cur)) == seen
    
    def backtrack():
        nonlocal unvisited_mask
        if time.time() > deadline:
            return False
        if len(path) == n:
            last = path[-1]
            if edge_arrival(last, start) <= K:
                found[0] = path[:]
                return True
            return False
        cur = path[-1]
        remaining_unvisited = n - len(path)
        candidates = []
        for nb in adj_list[cur]:
            if visited[nb]:
                continue
            if remaining_unvisited == 1:
                if edge_arrival(nb, start) > K:
                    continue
            onward = 0
            for w in adj_list[nb]:
                if not visited[w]:
                    onward += 1
            if onward < 1 and remaining_unvisited > 1:
                continue
            candidates.append((onward, nb))
        candidates.sort()
        for _, nb in candidates:
            visited[nb] = True
            path.append(nb)
            unvisited_mask &= ~(1 << nb)
            if len(path) < n and not connected_component_ok(nb):
                path.pop()
                visited[nb] = False
                unvisited_mask |= (1 << nb)
                continue
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
            unvisited_mask |= (1 << nb)
        return False
    
    if backtrack():
        return found[0]
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    default_cycle = list(range(n))
    best_cycle = default_cycle
    best_score = cycle_score(default_cycle)
    
    # Strategy: Use Dijkstra bottleneck search from multiple start vertices
    # with an initial upper bound from the default cycle
    dj_deadline = deadline - 5.0
    
    # Try Dijkstra from vertex 0 first
    dj_cycle, dj_score = dijkstra_bottleneck(0, dj_deadline, initial_upper_bound=best_score - 1)
    if dj_cycle is not None and dj_score < best_score:
        best_score = dj_score
        best_cycle = dj_cycle
    
    # Try Dijkstra from other promising start vertices if time permits
    if time.time() < deadline - 3.0:
        for start_v in [1, 23, 12, 5, 9]:
            if time.time() > deadline - 3.0:
                break
            dj2_cycle, dj2_score = dijkstra_bottleneck(start_v, deadline - 2.0, initial_upper_bound=best_score - 1)
            if dj2_cycle is not None and dj2_score < best_score:
                best_score = dj2_score
                best_cycle = dj2_cycle
    
    # If Dijkstra didn't find a good cycle, try DFS with binary search on K
    if time.time() < deadline - 3.0:
        lo, hi = 0, best_score - 1
        search_start = max(20, best_score - 30)
        for K in range(search_start, min(hi + 1, m)):
            if time.time() > deadline - 3.0:
                break
            hc = find_hc_prefix_dfs(K, time.time() + 1.0, start=0)
            if hc is not None:
                sc = cycle_score(hc)
                if sc < best_score:
                    best_score = sc
                    best_cycle = hc
                break
    
    return best_cycle
# EVOLVE-BLOCK-END
