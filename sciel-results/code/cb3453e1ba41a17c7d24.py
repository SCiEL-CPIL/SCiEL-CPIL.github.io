import time
import sys
from collections import deque
import numpy as np

sys.setrecursionlimit(10000)

# P is the pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build adjacency lists
    for i in range(n):
        adj_list[i] = list(adj[i])
    
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            stack = [i]
            visited_comp[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    # Sort components by size descending
    components.sort(key=len, reverse=True)
    
    best_path = [components[0][0]] if components else [0]
    best_len = len(best_path)
    
    if not components:
        return best_path
    
    largest_comp = components[0]
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # If the largest component is small enough, we might be able to do more exhaustive search
    # But for now, let's use a smarter heuristic-based DFS
    
    # Compute degrees within the component
    deg_in_comp = [0] * n
    for u in largest_comp:
        for v in adj[u]:
            if v in comp_set:
                deg_in_comp[u] += 1
    
    # Identify "core" vertices (high degree) and "periphery" (low degree)
    # In a sparse graph, the longest path likely goes through the core and ends in the periphery
    
    # Strategy: 
    # 1. First, try to find a long path using a greedy heuristic with backtracking
    # 2. Use Warnsdorff's rule: prefer moves that leave fewer options for the next step
    
    def count_unvisited_neighbors(v, visited):
        count = 0
        for w in adj_list[v]:
            if w not in visited:
                count += 1
        return count
    
    def dfs(path, current, visited, depth):
        nonlocal best_path, best_len
        
        # Update best
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        # Check time
        if time.time() >= deadline:
            return
        
        # Pruning: if we can't beat best, return
        # Quick upper bound: current path length + number of unvisited nodes in component
        # But more tightly: number of unvisited nodes reachable from current
        if len(path) + (comp_size - len(visited)) <= best_len:
            return
        
        # Get unvisited neighbors
        unvisited_neighbors = [v for v in adj_list[current] if v not in visited]
        
        if not unvisited_neighbors:
            return
        
        # Heuristic: sort by number of unvisited neighbors (Warnsdorff)
        # Vertices with fewer unvisited neighbors are "tighter" and should be visited earlier
        # to avoid trapping them
        unvisited_neighbors.sort(key=lambda v: count_unvisited_neighbors(v, visited))
        
        for v in unvisited_neighbors:
            if time.time() >= deadline:
                return
            
            path.append(v)
            visited.add(v)
            dfs(path, v, visited, depth + 1)
            path.pop()
            visited.discard(v)
            
            # Early termination if we found a Hamiltonian path
            if best_len == comp_size:
                return
    
    # Try starting from different vertices
    # Prioritize starting from vertices with low degree (leaves of the "tree-like" structure)
    # as they are more likely to be endpoints of a long path
    
    # Sort starting vertices by degree (ascending) - low degree first
    start_vertices = sorted(largest_comp, key=lambda v: deg_in_comp[v])
    
    # Also try starting from high-degree vertices
    start_vertices_high = sorted(largest_comp, key=lambda v: deg_in_comp[v], reverse=True)
    
    # Combine: alternate between low and high degree starts
    all_starts = []
    i, j = 0, 0
    while i < len(start_vertices) or j < len(start_vertices_high):
        if i < len(start_vertices):
            all_starts.append(start_vertices[i])
            i += 1
        if j < len(start_vertices_high):
            all_starts.append(start_vertices_high[j])
            j += 1
    
    for start in all_starts:
        if time.time() >= deadline:
            break
        
        path = [start]
        visited = {start}
        dfs(path, start, visited, 0)
        
        # Early termination if we found a Hamiltonian path
        if best_len == comp_size:
            break
    
    return best_path
