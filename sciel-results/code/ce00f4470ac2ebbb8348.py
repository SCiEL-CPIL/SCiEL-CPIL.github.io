import itertools
import random

def construct():
    # Construct Andrásfai graph And(9)
    n = 26
    k = 9
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

    # Helper to calculate independence number using bitmasking
    def alpha(n, edges):
        # adj_mask[i] is a bitmask of neighbors of i
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        best = 0
        # candidates is a bitmask of vertices available
        # current_set_size is the size of the current independent set
        
        def dfs(candidates, current_size):
            nonlocal best
            if not candidates:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning
            if current_size + bin(candidates).count('1') <= best:
                return
            
            # Pick a vertex with minimum degree in candidates?
            # Or just pick the first one.
            # To speed up, pick vertex with max degree in candidates to prune better?
            # For independent set, we want to exclude neighbors.
            # Let's pick a vertex v.
            
            # Find first set bit
            v = (candidates & -candidates).bit_length() - 1
            rest = candidates ^ (1 << v)
            
            # Option 1: Include v
            # New candidates: rest & ~adj_mask[v]
            new_candidates = rest & ~adj_mask[v]
            dfs(new_candidates, current_size + 1)
            
            # Option 2: Exclude v
            dfs(rest, current_size)

        dfs((1 << n) - 1, 0)
        return best

    # Current graph state
    current_n = n
    current_edges = edges
    
    # Greedy expansion
    for _ in range(20):
        # Find max independent set
        # We need the set, not just the number.
        # Modify alpha to return the set or use a similar DFS to find one.
        
        adj_mask = [0] * current_n
        for u, v in current_edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        max_set = []
        
        def find_max_is(candidates, current_set):
            nonlocal max_set
            if not candidates:
                if len(current_set) > len(max_set):
                    max_set = current_set.copy()
                return
            
            if len(current_set) + bin(candidates).count('1') <= len(max_set):
                return
                
            v = (candidates & -candidates).bit_length() - 1
            rest = candidates ^ (1 << v)
            
            # Include v
            new_cands = rest & ~adj_mask[v]
            find_max_is(new_cands, current_set + [v])
            
            # Exclude v
            find_max_is(rest, current_set)

        find_max_is((1 << current_n) - 1, [])
        
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
