import random
import numpy as np

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
        
    potential = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                potential.append((u, v))
                
    def get_diameter(edges_set):
        # BFS from each node
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v, w in edges_set:
                    if (u == v and dist[w] == -1) or (u == w and dist[v] == -1):
                        # Need adjacency list for speed
                        pass
            # This is slow. Let's use adjacency list.
        # Re-implement with adj list
        adj = {i: [] for i in range(n)}
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
        
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            max_d = max(max_d, max(dist))
        return max_d

    best_solution = None
    best_size = float('inf')
    
    for _ in range(50): # 50 restarts
        random.shuffle(potential)
        current_edges = set(base_edges)
        # Greedy
        while True:
            d = get_diameter(current_edges)
            if d <= 4:
                break
            best_edge = None
            best_d = float('inf')
            for e in potential:
                if e in current_edges:
                    continue
                current_edges.add(e)
                new_d = get_diameter(current_edges)
                if new_d < best_d:
                    best_d = new_d
                    best_edge = e
                current_edges.remove(e)
            if best_edge is None:
                break
            current_edges.add(best_edge)
            
        added = [e for e in current_edges if e not in base_edges]
        if len(added) < best_size:
            best_size = len(added)
            best_solution = added
            
        # Try to remove
        for e in added[:]:
            if e in current_edges:
                current_edges.remove(e)
                if get_diameter(current_edges) <= 4:
                    best_solution = [x for x in best_solution if x != e]
                    best_size -= 1
                else:
                    current_edges.add(e)
                    
    # Convert to list of lists
    result = []
    for u, v in best_solution:
        result.append([u, v])
    return result
