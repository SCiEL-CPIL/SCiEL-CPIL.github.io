"""Spectral-aware C4 minimization for 31 vertices, 91 edges."""
import numpy as np
from collections import defaultdict

def construct():
    n = 31
    m = 91
    
    # Start with a random graph or a structured one
    # Use a seeded random for reproducibility but try multiple seeds if time permits
    # Actually, let's start with a structured approach: 
    # We want to minimize C4. 
    # A good heuristic is to make the graph as "tree-like" as possible locally, 
    # i.e., minimize the number of paths of length 2 between any pair of vertices.
    # The number of C4s is sum over all pairs {u,v} of (common_neighbors(u,v) choose 2).
    # So we want to keep common_neighbors(u,v) as small as possible.
    
    # Strategy: 
    # 1. Initialize with a random graph or a greedy construction.
    # 2. Use local search: swap edges to reduce the total C4 count.
    #    Specifically, try to remove an edge (u,v) and add an edge (x,y) 
    #    such that the change in C4 count is negative.
    
    # Let's implement a simple local search.
    # Representation: adjacency matrix or set of edges.
    # For efficiency, we'll maintain:
    # - adj: list of sets of neighbors
    # - common_counts: a 2D array where common_counts[u][v] is the number of common neighbors.
    #   Actually, we can compute the change in C4 count directly.
    
    # The number of C4s containing edge (u,v) is sum_{w in N(u) \cap N(v)} (deg(w) - 1) / 2? 
    # No. A C4 is u-v-w-x-u. 
    # If we remove edge (u,v), we lose all C4s that use this edge.
    # The number of C4s using edge (u,v) is exactly the number of common neighbors of u and v.
    # Wait, no. If w is a common neighbor, then u-w-v is a path. To form a C4, we need another vertex x 
    # that is connected to both u and v? No, that would be a triangle.
    # A C4 is u-v-w-x-u. 
    # The edges are (u,v), (v,w), (w,x), (x,u).
    # So for a fixed edge (u,v), a C4 containing it corresponds to a vertex w adjacent to v 
    # and a vertex x adjacent to u, such that w and x are adjacent, and w != u, x != v, w != x.
    # This is getting complicated to update locally.
    
    # Alternative: Count C4s by pairs of vertices.
    # C4 count = sum_{u < v} (common_neighbors(u, v) choose 2).
    # This is much easier to update.
    # When we remove edge (a,b):
    #   For every pair (u,v) where u is a neighbor of a (u!=b) and v is a neighbor of b (v!=a),
    #   the common neighbor count for pair (u,v) decreases by 1 (since b is no longer a common neighbor? No, b is not a common neighbor of u and v unless u~b and v~b).
    #   Actually, removing edge (a,b) affects common_neighbors(u,v) only if a is a common neighbor of u,v or b is a common neighbor of u,v.
    #   If a is a common neighbor of u and v, it means u~a and v~a. Since we are removing (a,b), this doesn't change u~a or v~a. 
    #   Wait, common_neighbors(u,v) counts vertices w such that u~w and v~w.
    #   Removing edge (a,b) means a is no longer adjacent to b.
    #   So for the pair (a,b), common_neighbors(a,b) changes? No, common neighbors are vertices adjacent to BOTH a and b. 
    #   Removing the edge between a and b does not change the set of common neighbors of a and b.
    #   However, it changes the common neighbor count for pairs (a, x) where x is a neighbor of b?
    #   Let x be a neighbor of b. Before removal, a was a common neighbor of b and x? No, a~b and a~x?
    #   If a~x, then a is a common neighbor of b and x. After removing (a,b), a is no longer adjacent to b, so a is no longer a common neighbor of b and x.
    #   So, removing (a,b):
    #   For each x in N(b) \ {a}:
    #       common_neighbors(b, x) decreases by 1 (a is no longer a common neighbor).
    #   For each y in N(a) \ {b}:
    #       common_neighbors(a, y) decreases by 1 (b is no longer a common neighbor).
    
    #   Similarly, adding edge (c,d):
    #   For each x in N(d) \ {c}:
    #       common_neighbors(c, x) increases by 1.
    #   For each y in N(c) \ {d}:
    #       common_neighbors(d, y) increases by 1.
    
    #   The change in C4 count is:
    #   Delta = sum over affected pairs (u,v) of [ C(k_new, 2) - C(k_old, 2) ]
    #   where k is the common neighbor count.
    #   C(k,2) - C(k-1,2) = k-1.
    #   C(k,2) - C(k+1,2) = -k.
    
    #   So, removing (a,b):
    #   For x in N(b)\{a}: delta += -(common_neighbors(b,x) - 1)  [since count drops by 1]
    #   For y in N(a)\{b}: delta += -(common_neighbors(a,y) - 1)
    #   Wait, if count drops from k to k-1, the change in C(k,2) is C(k-1,2) - C(k,2) = - (k-1).
    #   So we subtract (k-1).
    
    #   Adding (c,d):
    #   For x in N(d)\{c}: delta += common_neighbors(c,x)  [count goes from k to k+1, change is k]
    #   For y in N(c)\{d}: delta += common_neighbors(d,y)
    
    #   We want Delta < 0.
    
    # Let's implement this.
    
    import random
    import time
    
    start_time = time.time()
    timeout = 110  # leave 10s margin
    
    # Initialize
    # Start with a random graph
    best_edges = None
    best_c4 = float('inf')
    
    for seed in range(10):
        random.seed(seed)
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edges.add((min(u,v), max(u,v)))
        
        # Build data structures
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in adj[u]:
                for w in adj[u]:
                    if w > v:
                        common[v][w] += 1
                        common[w][v] += 1
                    # Actually, this double counts or is inefficient.
        
        # Better way to compute common:
        # For each vertex u, for each pair of neighbors (v,w) of u, increment common[v][w].
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            neigh = list(adj[u])
            for i in range(len(neigh)):
                for j in range(i+1, len(neigh)):
                    v = neigh[i]
                    w = neigh[j]
                    common[v][w] += 1
                    common[w][v] += 1
        
        c4_count = sum(common[u][v] * (common[u][v]-1) // 2 for u in range(n) for v in range(u+1, n))
        
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = set(edges)
        
        # Local Search
        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            # Try all possible edge swaps
            # This is O(m^2 * deg^2) which might be too slow.
            # We need a smarter way.
            # Let's just try to find one improving swap at a time.
            
            # Randomly sample a few swaps to speed up
            # Or iterate through edges
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            for (a, b) in edge_list:
                if (time.time() - start_time) > timeout:
                    break
                # Try removing (a,b)
                # Calculate delta for removal
                delta_rem = 0
                for x in adj[b]:
                    if x != a:
                        delta_rem -= (common[b][x] - 1)
                for y in adj[a]:
                    if y != b:
                        delta_rem -= (common[a][y] - 1)
                
                # Now try to add a new edge (c,d) not in edges, c!=d
                # We want delta_add such that delta_rem + delta_add < 0
                # delta_add = sum_{x in N(d)\{c}} common[c][x] + sum_{y in N(c)\{d}} common[d][y]
                
                # This is expensive to search all (c,d).
                # Heuristic: try to add an edge between two vertices with low common neighbor counts?
                # Or just pick a random non-edge and check.
                
                # Let's pick a random non-edge
                found = False
                for _ in range(20): # try 20 random non-edges
                    c = random.randint(0, n-1)
                    d = random.randint(0, n-1)
                    if c == d:
                        continue
                    c, d = min(c,d), max(c,d)
                    if (c,d) in edges:
                        continue
                    if c == a and d == b:
                        continue
                    
                    delta_add = 0
                    for x in adj[d]:
                        if x != c:
                            delta_add += common[c][x]
                    for y in adj[c]:
                        if y != d:
                            delta_add += common[d][y]
                    
                    total_delta = delta_rem + delta_add
                    if total_delta < 0:
                        # Perform swap
                        edges.remove((a,b))
                        edges.add((c,d))
                        
                        # Update adj
                        adj[a].remove(b)
                        adj[b].remove(a)
                        adj[c].add(d)
                        adj[d].add(c)
                        
                        # Update common
                        # Remove (a,b) effects
                        for x in adj[b]: # adj[b] no longer has a
                            if x != a:
                                common[b][x] -= 1
                                common[x][b] -= 1
                        for y in adj[a]:
                            if y != b:
                                common[a][y] -= 1
                                common[y][a] -= 1
                        # Note: common[a][b] and common[b][a] are not affected by removing edge (a,b) in the sense of common neighbors, 
                        # but they are affected by the change in adjacency? 
                        # No, common[u][v] is number of w such that u~w and v~w.
                        # Removing edge (a,b) means a is not adjacent to b.
                        # Does this change common[a][b]? No, common[a][b] counts w adjacent to a AND b.
                        # The edge (a,b) itself is not a vertex w.
                        # So common[a][b] remains the same? 
                        # Wait. If w=a, is a adjacent to a? No. 
                        # So common[a][b] is unchanged by removing edge (a,b).
                        
                        # But we updated common[b][x] for x in adj[b]. 
                        # Since a was in adj[b] before, and we removed it, 
                        # the loop `for x in adj[b]` now iterates over the NEW adj[b].
                        # We need to be careful. 
                        # Before removal, a was in adj[b]. 
                        # The change in common[b][x] for x in OLD adj[b]\{a} is -1.
                        # The change in common[a][y] for y in OLD adj[a]\{b} is -1.
                        # My code above uses `adj[b]` AFTER removal. 
                        # So `x` iterates over neighbors of b excluding a. This is correct.
                        
                        # Add (c,d) effects
                        for x in adj[d]: # adj[d] now has c
                            if x != c:
                                common[c][x] += 1
                                common[x][c] += 1
                        for y in adj[c]:
                            if y != d:
                                common[d][y] += 1
                                common[y][d] += 1
                        
                        c4_count += total_delta
                        improved = True
                        found = True
                        break
                if found:
                    continue
            # End for (a,b)
        
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = set(edges)
        
        if (time.time() - start_time) > timeout:
            break
    
    # Convert best_edges to list of lists
    result = [[u, v] for u, v in sorted(best_edges)]
    return result

# Run a quick test to ensure it doesn't crash
# if __name__ == "__main__":
#     edges = construct()
#     print(len(edges))
