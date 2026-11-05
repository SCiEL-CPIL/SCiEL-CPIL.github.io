"""Rewritten program: exact longest simple path via component decomposition."""
import time

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Build adjacency
ADJ = [set() for _ in range(N)]
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Find connected components
def get_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in ADJ[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
    return comps

def is_tree(comp):
    """Check if component is a tree: edges = vertices - 1"""
    vset = set(comp)
    ecount = 0
    for u in comp:
        for v in ADJ[u]:
            if v in vset and u < v:
                ecount += 1
    return ecount == len(comp) - 1

def tree_diameter(comp):
    """Find diameter of a tree using two BFS."""
    # BFS from arbitrary node to find farthest
    def bfs(start):
        dist = {start: 0}
        parent = {start: -1}
        queue = [start]
        head = 0
        farthest = start
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in ADJ[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                    if dist[v] > dist[farthest]:
                        farthest = v
        return farthest, dist, parent

    a, _, _ = bfs(comp[0])
    b, dist, parent = bfs(a)
    # Reconstruct path from b to a
    path = [b]
    cur = b
    while cur != a:
        cur = parent[cur]
        path.append(cur)
    return path

def longest_path_small(comp, deadline):
    """Exhaustive DFS with branch-and-bound for small components."""
    n = len(comp)
    if n <= 1:
        return comp[:]
    
    # Map to local indices for efficiency
    idx_map = {v: i for i, v in enumerate(comp)}
    adj_local = [set() for _ in range(n)]
    for u in comp:
        for v in ADJ[u]:
            if v in idx_map:
                adj_local[idx_map[u]].add(idx_map[v])
    
    best_path = []
    best_len = 0
    
    # Try each starting vertex
    for start in range(n):
        # If even the best possible from here can't beat current, skip
        # Upper bound: n (can't exceed total vertices)
        if n <= best_len:
            break
        
        visited = [False] * n
        visited[start] = True
        path = [start]
        
        def dfs():
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            
            # Pruning: if we can't beat best even by visiting all remaining
            remaining = n - len(path)
            if len(path) + remaining <= best_len:
                return
            
            last = path[-1]
            # Sort neighbors by fewest unvisited neighbors first (greedy hint)
            neighbors = sorted(adj_local[last], key=lambda x: (
                sum(1 for y in adj_local[x] if not visited[y]), x
            ))
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    path.append(v)
                    dfs()
                    path.pop()
                    visited[v] = False
                    if time.time() > deadline:
                        return
        
        dfs()
        if time.time() > deadline:
            break
    
    if not best_path:
        return comp[:]
    
    # Convert back to original vertex indices
    return [comp[i] for i in best_path]

def longest_path_large(comp, deadline):
    """For larger components with cycles: iterative deepening + greedy initialization."""
    n = len(comp)
    idx_map = {v: i for i, v in enumerate(comp)}
    adj_local = [set() for _ in range(n)]
    for u in comp:
        for v in ADJ[u]:
            if v in idx_map:
                adj_local[idx_map[u]].add(idx_map[v])
    
    # Greedy initialization: try multiple greedy strategies
    def greedy_path(start):
        visited = [False] * n
        visited[start] = True
        path = [start]
        while True:
            last = path[-1]
            unvisited_neighbors = [v for v in adj_local[last] if not visited[v]]
            if not unvisited_neighbors:
                break
            # Pick neighbor with fewest unvisited neighbors (Warnsdorff-like)
            v = min(unvisited_neighbors, key=lambda x: (
                sum(1 for y in adj_local[x] if not visited[y]), x
            ))
            visited[v] = True
            path.append(v)
        return path
    
    best_path = []
    best_len = 0
    for start in range(n):
        p = greedy_path(start)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p[:]
    
    # Iterative deepening: try to extend beyond best_len
    # Use DFS with target = best_len + 1, best_len + 2, ...
    # But with strict time limit
    target = best_len + 1
    max_target = n
    
    while target <= max_target and time.time() < deadline:
        found = False
        for start in range(n):
            if time.time() > deadline:
                break
            visited = [False] * n
            visited[start] = True
            path = [start]
            
            def dfs_deep():
                nonlocal found
                if time.time() > deadline:
                    return True  # signal stop
                if len(path) >= target:
                    found = True
                    return True
                last = path[-1]
                # Prune: count unvisited reachable
                unvisited = [v for v in adj_local[last] if not visited[v]]
                if not unvisited:
                    return False
                # Sort by fewest unvisited neighbors
                unvisited.sort(key=lambda x: (
                    sum(1 for y in adj_local[x] if not visited[y]), x
                ))
                for v in unvisited:
                    visited[v] = True
                    path.append(v)
                    if dfs_deep():
                        path.pop()
                        visited[v] = False
                        return True
                    path.pop()
                    visited[v] = False
                return False
            
            if dfs_deep():
                break
            if found:
                break
        if found:
            target += 1
        else:
            break
    
    if not best_path:
        return comp[:]
    return [comp[i] for i in best_path]

def construct():
    deadline = time.time() + 50  # 50s safety margin within 60s (90s kill)
    
    comps = get_components()
    
    best_path = []
    best_len = 0
    
    # Sort components by size descending (larger components likely have longer paths)
    comps.sort(key=len, reverse=True)
    
    for comp in comps:
        if time.time() > deadline:
            break
        n = len(comp)
        
        # Upper bound: can't beat current best if component is too small
        if n <= best_len:
            continue
        
        if n == 1:
            if 1 > best_len:
                best_path = comp[:]
                best_len = 1
            continue
        
        if is_tree(comp):
            path = tree_diameter(comp)
            if len(path) > best_len:
                best_path = path
                best_len = len(path)
        elif n <= 22:
            path = longest_path_small(comp, deadline)
            if len(path) > best_len:
                best_path = path
                best_len = len(path)
        else:
            path = longest_path_large(comp, deadline)
            if len(path) > best_len:
                best_path = path
                best_len = len(path)
    
    if not best_path:
        # Fallback: return any single vertex
        return [0]
    
    return best_path
