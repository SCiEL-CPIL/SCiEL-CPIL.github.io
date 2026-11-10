"""Rewritten program for large clique subdivisions in a pinned graph."""
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_path(s, t, blocked):
        if s == t:
            return [s]
        parent = {s: None}
        queue = deque([s])
        while queue:
            u = queue.popleft()
            if u == t:
                path = []
                cur = t
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for w in adj[u]:
                if w not in parent and w not in blocked:
                    parent[w] = u
                    queue.append(w)
        return None
    
    def find_disjoint_paths(branches):
        branches_set = set(branches)
        blocked = set()
        paths = {}
        pairs = list(combinations(branches, 2))
        
        # Order: direct edges first, then by BFS distance
        direct = [(u,v) for u,v in pairs if v in adj[u]]
        non_direct = [(u,v) for u,v in pairs if v not in adj[u]]
        
        def bfs_dist(u, v):
            if u == v: return 0
            visited = {u}
            queue = deque([(u, 0)])
            while queue:
                node, d = queue.popleft()
                if node == v: return d
                for w in adj[node]:
                    if w not in visited:
                        visited.add(w)
                        queue.append((w, d+1))
            return 999
        
        non_direct.sort(key=lambda p: bfs_dist(p[0], p[1]))
        ordered = direct + non_direct
        
        for u, v in ordered:
            path = find_path(u, v, blocked)
            if path is None:
                return None
            for w in path[1:-1]:
                blocked.add(w)
            paths[(u, v)] = path
        return paths
    
    best_result = None
    
    # Strategy: Greedy expansion from triangles
    # Find all triangles
    triangles = []
    for i in range(n):
        for j in range(i+1, n):
            if j in adj[i]:
                for k in range(j+1, n):
                    if k in adj[i] and k in adj[j]:
                        triangles.append((i, j, k))
    
    # Sort triangles by "potential" - how many other vertices connect to all three
    def triangle_potential(tri):
        a, b, c = tri
        count = 0
        for v in range(n):
            if v not in tri and v in adj[a] and v in adj[b] and v in adj[c]:
                count += 1
        return count
    
    triangles.sort(key=triangle_potential, reverse=True)
    
    for tri in triangles:
        if time.time() > deadline:
            break
        branches = list(tri)
        paths = find_disjoint_paths(branches)
        if paths is None:
            continue
        if best_result is None or len(branches) > len(best_result[0]):
            best_result = (branches, paths)
        
        # Greedy expansion
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            # Try adding each non-branch vertex
            candidates = [v for v in range(n) if v not in branches]
            # Sort by connectivity to existing branches
            candidates.sort(key=lambda v: -sum(1 for b in branches if b in adj[v]))
            
            for v in candidates:
                if time.time() > deadline:
                    break
                new_branches = branches + [v]
                new_paths = find_disjoint_paths(new_branches)
                if new_paths is not None:
                    branches = new_branches
                    paths = new_paths
                    if best_result is None or len(branches) > len(best_result[0]):
                        best_result = (branches, paths)
                    changed = True
                    break
    
    if best_result is None:
        u, v = P['edges'][0]
        best_result = ([u, v], {(u, v): [u, v]})
    
    branches, paths = best_result
    branch_list = list(branches)
    path_list = []
    for u, v in combinations(branch_list, 2):
        p = paths.get((u, v))
        if p is None:
            p = [u, v]
        path_list.append(list(p))
    
    return {'branches': branch_list, 'paths': path_list}
