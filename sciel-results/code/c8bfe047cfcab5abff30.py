import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))

    def bfs_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_d:
                    max_d = d
        return max_d

    def is_triangle_free(adj):
        for u in range(n):
            neighbors = list(adj[u])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        return False
        return True

    def check_solution(added_edges):
        adj = [set(s) for s in base_adj]
        for u, v in added_edges:
            a, b = min(u, v), max(u, v)
            if u == v:
                return False
            if (a, b) in existing_edges:
                return False
            if (a, b) in [(min(x, y), max(x, y)) for x, y in added_edges if x != u and y != v]:
                # Check for duplicates among added edges
                pass
            adj[u].add(v)
            adj[v].add(u)
        
        # Check for duplicate added edges
        seen = set()
        for u, v in added_edges:
            a, b = min(u, v), max(u, v)
            if a in seen or b in seen:
                pass
            if (a, b) in seen:
                return False
            seen.add((a, b))
        
        if not is_triangle_free(adj):
            return False
        diam = bfs_diameter(adj)
        if diam > 4:
            return False
        return True

    # Precompute candidate edges
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in existing_edges:
                candidates.append((u, v))

    # Try small k first with backtracking
    # For k=1,2,3: exhaustive
    # For k=4,5,6: backtracking with pruning
    
    deadline = time.time() + 50.0

    # Try k=1
    for edge in candidates:
        if time.time() > deadline:
            break
        if check_solution([edge]):
            return [[int(edge[0]), int(edge[1])]]

    # Try k=2
    for i in range(len(candidates)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(candidates)):
            if check_solution([candidates[i], candidates[j]]):
                return [[int(candidates[i][0]), int(candidates[i][1])], [int(candidates[j][0]), int(candidates[j][1])]]

    # Try k=3
    for i in range(len(candidates)):
        if time.time() > deadline:
            break
        for j in range(i + 1, len(candidates)):
            if time.time() > deadline:
                break
            for k in range(j + 1, len(candidates)):
                if check_solution([candidates[i], candidates[j], candidates[k]]):
                    return [[int(candidates[i][0]), int(candidates[i][1])], 
                            [int(candidates[j][0]), int(candidates[j][1])], 
                            [int(candidates[k][0]), int(candidates[k][1])]]

    # For k>=4, use a smarter approach
    # Since the path has 24 vertices, we need to "shorten" the diameter significantly.
    # A known good solution is 6 edges. Let's try to find 4, 5, or 6.
    
    # Try some structured patterns for k=4,5,6
    patterns = [
        # k=4
        [(0, 6), (6, 12), (12, 18), (18, 23)],
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23)],  # k=5
        [(0, 4), (4, 9), (9, 14), (14, 19), (19, 23)],  # k=5
        [(0, 6), (6, 11), (11, 17), (17, 23), (6, 16), (16, 22)],  # k=6
        [(0, 5), (5, 10), (10, 15), (15, 20), (20, 23), (0, 10)],  # k=6
    ]
    
    for pattern in patterns:
        if time.time() > deadline:
            break
        if check_solution(pattern):
            return [[int(u), int(v)] for u, v in pattern]

    # Greedy approach for larger k
    def greedy_search():
        adj = [set(s) for s in base_adj]
        added_edges = []
        added_set = set()
        
        def compute_diam():
            return bfs_diameter(adj)
        
        def is_valid_edge(u, v):
            if u == v:
                return False
            a, b = min(u, v), max(u, v)
            if (a, b) in existing_edges:
                return False
            if (a, b) in added_set:
                return False
            # Check triangle-free
            if adj[u] & adj[v]:
                return False
            return True
        
        while time.time() < deadline:
            diam = compute_diam()
            if diam <= 4:
                break
            
            best_edge = None
            best_new_diam = diam + 1
            
            for u in range(n):
                for v in range(u + 1, n):
                    if not is_valid_edge(u, v):
                        continue
                    adj[u].add(v)
                    adj[v].add(u)
                    new_diam = compute_diam()
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
                    if new_diam < best_new_diam:
                        best_new_diam = new_diam
                        best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u].add(v)
            adj[v].add(u)
            added_set.add((min(u, v), max(u, v)))
            added_edges.append([u, v])
        
        return added_edges

    result = greedy_search()
    if result:
        return [[int(u), int(v)] for u, v in result]
    
    return []
