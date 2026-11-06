"""Long path finder for G(100, 1.8/100) with seed 748."""
import time
import random
import numpy as np

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80  # 80s deadline with 10s margin
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Get degree
    deg = [len(adj[i]) for i in range(n)]
    
    # Find connected components
    def get_components():
        visited = [False] * n
        comps = []
        for i in range(n):
            if not visited[i]:
                comp = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)
                    for nb in adj[node]:
                        if not visited[nb]:
                            visited[nb] = True
                            stack.append(nb)
                comps.append(comp)
        return comps
    
    components = get_components()
    largest_comp = max(components, key=len)
    
    # Strategy 1: Bipartite matching to identify core vertices
    # Use Hopcroft-Karp for bipartite matching
    def hopcroft_karp(adj_list, n_left, n_right):
        """Bipartite matching using Hopcroft-Karp algorithm."""
        match_l = [-1] * n_left
        match_r = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            from collections import deque
            queue = deque()
            for u in range(n_left):
                if match_l[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj_list[u]:
                    if v >= n_right:
                        continue
                    if match_r[v] == -1:
                        found = True
                    elif dist[match_r[v]] == float('inf'):
                        dist[match_r[v]] = dist[u] + 1
                        queue.append(match_r[v])
            return found
        
        def dfs(u):
            for v in adj_list[u]:
                if v >= n_right:
                    continue
                if match_r[v] == -1 or (dist[match_r[v]] == dist[u] + 1 and dfs(match_r[v])):
                    match_l[u] = v
                    match_r[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching = 0
        while bfs():
            for u in range(n_left):
                if match_l[u] == -1:
                    if dfs(u):
                        matching += 1
        return matching, match_l, match_r
    
    # Create bipartite graph: left = vertices, right = vertices (for path finding)
    # We'll use a different approach: find maximum matching in the graph
    # to identify vertices that can be part of long paths
    
    # Strategy 2: Greedy path building with backtracking
    def greedy_path(start, seen, priority_fn):
        path = [start]
        seen = set(seen) if seen else set()
        seen.add(start)
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v not in seen]
            if not candidates:
                break
            # Choose next vertex using priority function
            best_v = min(candidates, key=priority_fn)
            path.append(best_v)
            seen.add(best_v)
        return path
    
    # Priority: prefer vertices with fewer unvisited neighbors (dead-end avoidance)
    def dead_end_priority(v, seen):
        unvisited_neighbors = len(adj[v] - seen)
        return (unvisited_neighbors, v)
    
    # Strategy 3: DFS with backtracking for longer paths
    def dfs_long_path(start, max_depth=5000):
        best_path = [start]
        path = [start]
        seen = {start}
        
        def dfs(current, depth):
            nonlocal best_path
            if time.time() > deadline:
                return
            if len(path) > len(best_path):
                best_path = path[:]
            if depth >= max_depth:
                return
            
            current = path[-1]
            candidates = sorted([v for v in adj[current] if v not in seen],
                               key=lambda v: (len(adj[v] - seen), v))
            
            for v in candidates:
                if time.time() > deadline:
                    return
                path.append(v)
                seen.add(v)
                dfs(v, depth + 1)
                path.pop()
                seen.remove(v)
        
        dfs(start, 0)
        return best_path
    
    # Strategy 4: Local improvement - try to extend or rearrange path
    def improve_path(path):
        if not path:
            return path
        path = list(path)
        improved = True
        while improved:
            improved = False
            # Try to extend from both ends
            for end in [0, -1]:
                current = path[end]
                for v in adj[current]:
                    if v not in path:
                        if end == 0:
                            path.insert(0, v)
                        else:
                            path.append(v)
                        improved = True
                        break
            # Try to insert unvisited vertices in the middle
            visited = set(path)
            for i in range(len(path) - 1):
                for v in adj[path[i]]:
                    if v not in visited and v in adj[path[i+1]]:
                        path.insert(i + 1, v)
                        visited.add(v)
                        improved = True
                        break
                if improved:
                    break
        return path
    
    # Find best path using multiple strategies
    best_path = []
    
    # Try greedy from each vertex in largest component
    for start in largest_comp:
        if time.time() > deadline:
            break
        path = greedy_path(start, set(), lambda v: dead_end_priority(v, set()))
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Try DFS from promising starts (high degree vertices)
    sorted_starts = sorted(largest_comp, key=lambda v: -deg[v])
    for start in sorted_starts[:20]:
        if time.time() > deadline:
            break
        path = dfs_long_path(start, max_depth=3000)
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Try multiple random restarts with different heuristics
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        start = rng.choice(largest_comp)
        # Randomized greedy
        path = [start]
        seen = {start}
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v not in seen]
            if not candidates:
                break
            # Random choice weighted by remaining degree
            weights = [1.0 / max(1, len(adj[v] - seen)) for v in candidates]
            total = sum(weights)
            r = rng.random() * total
            cumulative = 0
            chosen = candidates[0]
            for v, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    chosen = v
                    break
            path.append(chosen)
            seen.add(chosen)
        
        if len(path) > len(best_path):
            best_path = path[:]
    
    # Final improvement pass
    best_path = improve_path(best_path)
    
    # Verify the path is valid
    for i in range(len(best_path) - 1):
        if best_path[i + 1] not in adj[best_path[i]]:
            # Fallback to simple valid path
            best_path = [0]
            break
    
    if not best_path:
        best_path = [0]
    
    return best_path
