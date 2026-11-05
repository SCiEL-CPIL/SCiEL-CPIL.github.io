import itertools
import random

def construct():
    # Construct Andrásfai graph And(9)
    n = 26
    k = 9
    # Vertices 0..25
    # Edge if (j-i) mod 26 is in {1, 4, 7, 10, 13, 16, 19, 22, 25}
    # i.e., (j-i) mod 3 == 1
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    
    # Helper to check triangle-free
    def is_triangle_free(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            return False
        return True

    # Helper to calculate independence number using bitsets for speed
    def alpha(n, edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Convert adjacency to bitmasks for fast operations
        adj_mask = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    adj_mask[u] |= (1 << v)
        
        best = 0
        best_set = []
        
        # Use recursive DFS with bitmasks
        # candidates is a bitmask of available vertices
        def dfs(candidates, current_count, current_set):
            nonlocal best, best_set
            if not candidates:
                if current_count > best:
                    best = current_count
                    best_set = current_set.copy()
                return
            
            # Pruning: if current_count + popcount(candidates) <= best, return
            if current_count + bin(candidates).count('1') <= best:
                return
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with max degree in candidates
            # For simplicity, pick the first one
            v = (candidates & -candidates).bit_length() - 1
            rest = candidates & ~(1 << v)
            
            # Option 1: Include v
            # New candidates are those in rest not adjacent to v
            new_candidates = rest & ~adj_mask[v]
            dfs(new_candidates, current_count + 1, current_set + [v])
            
            # Option 2: Exclude v
            dfs(rest, current_count, current_set)

        dfs((1 << n) - 1, 0, [])
        return best

    # Current graph state
    current_n = n
    current_edges = edges
    
    # Greedy expansion
    for _ in range(20): # Limit iterations
        # Find a max independent set
        # For efficiency, we can find one using the same DFS logic but returning the set
        # Or just use a greedy heuristic to find a large independent set, then verify alpha of new graph
        # Since n is small, let's find a max independent set exactly.
        
        adj = [[False]*current_n for _ in range(current_n)]
        for u, v in current_edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Find max independent set
        max_set = []
        # Using a simple recursive search for max independent set
        def find_max_is(candidates, current_set):
            nonlocal max_set
            if not candidates:
                if len(current_set) > len(max_set):
                    max_set = current_set.copy()
                return
            
            if len(current_set) + len(candidates) <= len(max_set):
                return
                
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v
            new_cands = [u for u in rest if not adj[v][u]]
            find_max_is(new_cands, current_set + [v])
            
            # Exclude v
            find_max_is(rest, current_set)

        find_max_is(list(range(current_n)), [])
        
        I = max_set
        
        # Create new vertex
        new_n = current_n + 1
        new_vertex = current_n
        
        # New edges: connect new_vertex to all in I
        new_edges = current_edges.copy()
        for v in I:
            new_edges.append([new_vertex, v])
            
        # Check if new graph is valid
        # 1. Triangle-free? Yes, because I is independent.
        # 2. Alpha < 10?
        new_alpha = alpha(new_n, new_edges)
        
        if new_alpha < 10:
            current_n = new_n
            current_edges = new_edges
        else:
            break
            
    return {'n': current_n, 'edges': current_edges}
