import sys
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}
    
    n = P['n']
    edge_order = P['edge_order']
    
    # Map edge to its position (1-based)
    pos = {}
    for i, (u, v) in enumerate(edge_order):
        pos[(u, v)] = i + 1
        pos[(v, u)] = i + 1
    
    def get_edge_pos(u, v):
        return pos.get((u, v), pos.get((v, u), float('inf')))
    
    def cycle_score(path):
        """Compute the max edge position in the cycle."""
        max_pos = 0
        for i in range(len(path)):
            u = path[i]
            v = path[(i + 1) % len(path)]
            epos = get_edge_pos(u, v)
            if epos > max_pos:
                max_pos = epos
        return max_pos
    
    def two_opt_improve(path, max_iter=100):
        """Apply 2-opt to minimize the maximum edge position."""
        path = list(path)
        n = len(path)
        best_score = cycle_score(path)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue  # Would create same cycle
                    # Try reversing path[i+1..j]
                    new_path = path[:i+1] + path[i+1:j+1][::-1] + path[j+1:]
                    new_score = cycle_score(new_path)
                    if new_score < best_score:
                        path = new_path
                        best_score = new_score
                        improved = True
            if not improved:
                break
        return path, best_score
    
    def find_hamilton_cycle(K, start_vertex=0, rng=None):
        """Find a Hamilton cycle in G_K using backtracking with Warnsdorff's rule."""
        if rng is None:
            rng = random.Random(0)
        
        adj = [[] for _ in range(n)]
        for i in range(K):
            u, v = edge_order[i]
            adj[u].append(v)
            adj[v].append(u)
        
        # Check connectivity
        visited = [False] * n
        queue = [start_vertex]
        visited[start_vertex] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        if count < n:
            return None
        
        # Check if start_vertex can close the cycle
        if start_vertex not in adj[start_vertex]:  # This is always false, but we need to check neighbors
            pass
        
        visited = [False] * n
        path = [start_vertex]
        visited[start_vertex] = True
        
        def dfs(u):
            if len(path) == n:
                # Check if we can close the cycle
                return start_vertex in adj[u]
            
            unvisited_neighbors = [v for v in adj[u] if not visited[v]]
            if not unvisited_neighbors:
                return False
            
            # Warnsdorff: sort by number of unvisited neighbors
            def get_unvisited_deg(v):
                count = 0
                for w in adj[v]:
                    if not visited[w]:
                        count += 1
                return count

            degs = [(get_unvisited_deg(v), v) for v in unvisited_neighbors]
            degs.sort()
            
            # Shuffle ties
            i = 0
            while i < len(degs):
                j = i
                while j < len(degs) and degs[j][0] == degs[i][0]:
                    j += 1
                group = degs[i:j]
                rng.shuffle(group)
                degs[i:j] = group
                i = j
                
            for _, v in degs:
                if not visited[v]:
                    visited[v] = True
                    path.append(v)
                    if dfs(v):
                        return True
                    path.pop()
                    visited[v] = False
                    
            return False
            
        if dfs(start_vertex):
            return path
        return None
    
    best_path = None
    best_score = float('inf')
    
    # Strategy: Try different K values and starting vertices
    # Start from a reasonable lower bound
    min_K = 24  # Minimum for a Hamilton cycle
    
    # Try increasing K, with multiple restarts for each K
    for K in range(min_K, len(edge_order) + 1):
        if time.time() > deadline:
            break
        
        # Try a few starting vertices and random seeds
        found_any = False
        for start in range(0, n, 4):  # Try every 4th vertex as start
            if time.time() > deadline:
                break
            for seed in range(3):
                if time.time() > deadline:
                    break
                rng = random.Random(seed * 1000 + start)
                path = find_hamilton_cycle(K, start, rng)
                if path is not None:
                    found_any = True
                    # Apply 2-opt to improve
                    improved_path, score = two_opt_improve(path, max_iter=50)
                    if score < best_score:
                        best_score = score
                        best_path = improved_path
                    break  # Found a cycle for this K and start, try next start
            if found_any:
                break  # Found cycles at this K, move to next K
        
        # If we found a good enough score, we can stop
        if best_score <= K + 5:  # Heuristic: if we're close to optimal for this K
            break
    
    # If we didn't find anything, use a fallback
    if best_path is None:
        # Try a simple approach: use the first K edges that form a connected graph
        for K in range(min_K, len(edge_order) + 1):
            if time.time() > deadline:
                break
            adj = [[] for _ in range(n)]
            for i in range(K):
                u, v = edge_order[i]
                adj[u].append(v)
                adj[v].append(u)
            
            visited = [False] * n
            queue = [0]
            visited[0] = True
            count = 1
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            if count == n:
                # Try to find a cycle with multiple restarts
                for start in range(n):
                    for seed in range(10):
                        if time.time() > deadline:
                            break
                        rng = random.Random(seed * 10000 + start)
                        path = find_hamilton_cycle(K, start, rng)
                        if path is not None:
                            improved_path, score = two_opt_improve(path, max_iter=100)
                            if score < best_score:
                                best_score = score
                                best_path = improved_path
                            break
                    if best_path is not None:
                        break
                if best_path is not None:
                    break
    
    if best_path is None:
        # Ultimate fallback: identity permutation
        best_path = list(range(n))
    
    return best_path

print(construct())
