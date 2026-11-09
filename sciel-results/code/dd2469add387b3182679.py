import sys
import random
from collections import defaultdict

def is_triangle_free(adj, n):
    """Check if graph is triangle-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                # Check if any common neighbor
                # common = neighbors[i] & neighbors[j]
                # For speed, iterate smaller neighbor list
                ni = [k for k in range(n) if adj[i][k]]
                nj = [k for k in range(n) if adj[j][k]]
                # Optimize: just check intersection
                # Since n is small (<=41), simple loop is fine
                for k in ni:
                    if k > j and adj[j][k]:
                        return False
                    # Wait, if k < i, we already checked?
                    # Triangle i, j, k.
                    # We just need to know if ANY k exists.
                    # adj[i][k] and adj[j][k] and adj[i][j]
                # Simpler:
                # for k in range(n):
                #    if adj[i][k] and adj[j][k]: return False
                pass
    
    # Let's rewrite cleanly
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def has_independent_set_of_size_k(adj, n, k):
    """
    Checks if there is an independent set of size k.
    Uses branch and bound.
    """
    if k <= 0:
        return True
    if n < k:
        return False
    
    # Use a list of active vertices
    active = list(range(n))
    
    def backtrack(active, k_remaining):
        if k_remaining == 0:
            return True
        if len(active) < k_remaining:
            return False
        
        # Heuristic: pick vertex with highest degree in the induced subgraph
        # Degree in induced subgraph: number of neighbors within active
        best_v = -1
        best_deg = -1
        for v in active:
            deg = 0
            for u in active:
                if u != v and adj[v][u]:
                    deg += 1
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            # No vertices left?
            return k_remaining == 0

        v = best_v
        
        # Branch 1: Include v
        # New active: active \ {v} \ N(v)
        new_active_include = []
        for u in active:
            if u != v and not adj[v][u]:
                new_active_include.append(u)
        
        if backtrack(new_active_include, k_remaining - 1):
            return True
        
        # Branch 2: Exclude v
        # New active: active \ {v}
        new_active_exclude = [u for u in active if u != v]
        
        if backtrack(new_active_exclude, k_remaining):
            return True
            
        return False

    return backtrack(active, k)

def construct():
    # Try to find a valid graph for n=40 down to 30
    # Use random circulant graphs
    
    random.seed(42) # For reproducibility in dev, but maybe remove for production? 
    # Task says "Python standard library...". Random is std.
    
    for n in range(40, 29, -1):
        # Try a specific known good structure first?
        # The seed's structure failed for n>26.
        
        # Try random circulant graphs
        # S subset of {1, ..., n//2}
        half_n = n // 2
        possible_generators = list(range(1, half_n + 1))
        
        # We want degree ~ n/2, so |S| ~ n/4
        target_size = max(1, n // 4)
        
        # Try many random sets
        for _ in range(1000):
            S = random.sample(possible_generators, target_size)
            
            # Build adjacency
            adj = [[False] * n for _ in range(n)]
            for s in S:
                for i in range(n):
                    j = (i + s) % n
                    if i < j:
                        adj[i][j] = True
                        adj[j][i] = True
                    elif i > j:
                        # Already handled by symmetry? 
                        # (i+s)%n = j. If i<j, we set. 
                        # If i>j, then (j+s)%n = i? Not necessarily.
                        # Circulant graphs are symmetric if s is in S, then n-s is effectively in S?
                        # No, edge (i, j) exists if (j-i)%n in S OR (i-j)%n in S.
                        # My loop: 
                        # for i in range(n):
                        #   j = (i+s)%n
                        #   if i<j: set
                        #   else: set (i,j)
                        pass
            
            # Re-implement adj building correctly
            adj = [[False] * n for _ in range(n)]
            for s in S:
                for i in range(n):
                    j = (i + s) % n
                    if i != j:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Check triangle-free
            if not is_triangle_free(adj, n):
                continue
                
            # Check independent set
            if has_independent_set_of_size_k(adj, n, 10):
                continue
            else:
                # Valid
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    # Fallback to a smaller n if 30-40 fails
    # Try n=26 with seed logic?
    # Or just return a small valid graph
    # A single edge is triangle free and alpha = n-1? No.
    # Empty graph: alpha = n. Invalid for n>=10.
    # Complete graph: has triangles.
    # We need a valid graph.
    # C5 is triangle free, alpha=2.
    # For n=26, we can use the seed's graph if it's valid?
    # We established seed's graph for n=26 might be invalid.
    # Let's just return a known small valid graph if all else fails.
    # e.g. n=9, complete bipartite? No, alpha=5.
    # n=9, C9? alpha=4. Valid.
    n = 9
    adj = [[False] * n for _ in range(n)]
    edges = []
    for i in range(n):
        j = (i + 1) % n
        if i < j:
            adj[i][j] = True
            adj[j][i] = True
            edges.append([i, j])
        elif i > j:
            # (i+1)%n = 0 when i=n-1. 0 < n-1.
            # So (n-1, 0) is handled when i=0, j=1? No.
            # i=n-1, j=0. i>j.
            # We set adj[i][j] and adj[j][i].
            pass
            
    # Re-build edges for C9
    edges = []
    for i in range(n):
        j = (i + 1) % n
        if i < j:
            edges.append([i, j])
        else:
            edges.append([j, i])
            
    return {'n': n, 'edges': edges}
