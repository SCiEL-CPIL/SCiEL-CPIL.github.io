import time
import random
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    
    best_path = []
    
    # Function to find diameter of a tree component
    def get_tree_diameter(comp):
        if len(comp) == 0:
            return []
        if len(comp) == 1:
            return comp
        
        # Check if it's a tree: edges in component should be len(comp) - 1
        # We can just run BFS twice if we assume it's a tree or tree-like.
        # But if it has cycles, BFS diameter is not the longest path.
        # So we only use this if we verify it's a tree.
        
        # Let's verify if it's a tree
        edge_count = 0
        comp_set = set(comp)
        for u in comp:
            for v in adj[u]:
                if v in comp_set and u < v:
                    edge_count += 1
        
        if edge_count != len(comp) - 1:
            return None # Not a tree
        
        # BFS to find farthest node from arbitrary start
        def bfs(start):
            dist = {start: 0}
            parent = {start: None}
            q = deque([start])
            farthest = start
            max_dist = 0
            while q:
                node = q.popleft()
                if dist[node] > max_dist:
                    max_dist = dist[node]
                    farthest = node
                for neighbor in adj[node]:
                    if neighbor in comp_set and neighbor not in dist:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        q.append(neighbor)
            return farthest, dist, parent
        
        # First BFS
        start = comp[0]
        far1, dist1, parent1 = bfs(start)
        
        # Second BFS
        far2, dist2, parent2 = bfs(far1)
        
        # Reconstruct path
        path = []
        curr = far2
        while curr is not None:
            path.append(curr)
            curr = parent2[curr]
        path.reverse()
        return path

    # Function to find longest path in a general component using DFS
    def longest_path_dfs(comp, time_limit):
        comp_set = set(comp)
        best_local = []
        
        # Heuristic: Start from vertices with low degree
        degrees = {v: len([u for u in adj[v] if u in comp_set]) for v in comp}
        sorted_starts = sorted(comp, key=lambda x: (degrees[x], x))
        
        def dfs(start, time_limit):
            nonlocal best_local
            path = [start]
            visited = {start}
            
            def _dfs():
                nonlocal best_local
                if time.time() > time_limit:
                    return
                
                if len(path) > len(best_local):
                    best_local = list(path)
                
                current = path[-1]
                # Get unvisited neighbors in the component
                unvisited_neighbors = [v for v in adj[current] if v in comp_set and v not in visited]
                
                if not unvisited_neighbors:
                    return
                
                # Warnsdorff heuristic: sort by number of unvisited neighbors
                # Fewer unvisited neighbors first
                def unvisited_count(v):
                    return len([u for u in adj[v] if u in comp_set and u not in visited])
                
                unvisited_neighbors.sort(key=lambda v: (unvisited_count(v), v))
                
                for v in unvisited_neighbors:
                    path.append(v)
                    visited.add(v)
                    _dfs()
                    path.pop()
                    visited.remove(v)
            
            _dfs()
            return best_local
        
        # Try starting from a few low-degree vertices
        # To save time, limit the number of starts if component is large
        max_starts = 10
        for start in sorted_starts[:max_starts]:
            if time.time() > time_limit:
                break
            res = dfs(start, time_limit)
            if len(res) > len(best_local):
                best_local = res
        
        return best_local

    # Process each component
    deadline = time.time() + 85
    
    for comp in components:
        if time.time() > deadline:
            break
            
        # Try tree diameter first
        tree_path = get_tree_diameter(comp)
        if tree_path:
            if len(tree_path) > len(best_path):
                best_path = tree_path
        else:
            # Use DFS for cyclic components
            # Allocate remaining time
            remaining = deadline - time.time()
            if remaining > 1:
                path = longest_path_dfs(comp, time.time() + remaining - 1)
                if len(path) > len(best_path):
                    best_path = path

    if not best_path:
        # Fallback: return any single vertex or edge if found
        if P['edges']:
            best_path = [P['edges'][0][0], P['edges'][0][1]]
        else:
            best_path = [0]
            
    return best_path
