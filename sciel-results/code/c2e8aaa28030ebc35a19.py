"""
Triangle-free graph without independent 10-set.
Strategy: Use known constructions and try to find larger graphs via
complementary of triangle-free graphs (since complement of independent set
in G is a clique in complement, but we need triangle-free G).

Actually, let's think differently:
- We need G to be triangle-free (no K3)
- We need alpha(G) < 10 (no independent set of size 10)

The Andrasfai graph And(9) has 26 vertices, is triangle-free, and alpha = 9.
That gives n=26.

Known results: R(3,10) is between 40 and 42. So there exist triangle-free
graphs on 39 vertices with no independent set of size 10 (giving R(3,10) > 39).

Let me try to construct a graph on more vertices. 

Approach: Use a random triangle-free graph and check if alpha < 10.
Or use known constructions like Mycielski graphs or other systematic methods.

Actually, the largest known triangle-free graph with alpha <= 9 is on 39 vertices.
Let me try to construct one using a greedy/random approach with verification.

Strategy:
1. Start with a random graph
2. Remove edges to make it triangle-free
3. Check if alpha < 10
4. If not, try again or modify

But random might not work well. Let me use a more structured approach.

Alternative: Use the complement. A graph G is triangle-free iff its complement
has no independent set of size 3. And alpha(G) < 10 iff the complement has no
clique of size 10.

So we need a graph H (the complement of G) such that:
- H has no independent set of size 3 (i.e., alpha(H) <= 2)
- H has no clique of size 10 (i.e., omega(H) <= 9)

This is equivalent to finding a graph with chromatic number constraints...

Actually, let me just try to directly construct a triangle-free graph with small alpha.

Known: The Clebsch graph is triangle-free with 16 vertices and alpha = 5.
The Hoffman-Singleton graph is triangle-free with 50 vertices, but alpha is large.

Let me try a different approach: use a random triangle-free graph and verify.

For a random graph G(n, p), the expected number of triangles is C(n,3)*p^3.
To be triangle-free, we need p small enough. But then alpha will be large.

Better: Use a structured construction. 

Let me try using a bipartite graph (which is triangle-free) and see if we can keep alpha small.
A complete bipartite graph K_{a,b} has alpha = max(a,b). So if a=b=9, alpha=9, n=18.
But we can do better with non-complete bipartite.

Actually, let me try to use the known fact that R(3,10) > 39, meaning there exists
a triangle-free graph on 39 vertices with alpha <= 9. Let me try to construct one.

I'll use a randomized approach:
1. Generate a random graph
2. Iteratively remove edges to eliminate triangles
3. Check alpha
4. Repeat with different seeds

But this might be slow. Let me use a smarter approach.

Actually, let me just use a well-known construction. The graph from the lower bound
for R(3,10) is typically constructed using algebraic methods or specific designs.

Let me try a simple approach: use a random regular graph and check.

For time efficiency, let me try multiple random triangle-free graphs and check alpha.
"""

import random
import time
from itertools import combinations

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj is triangle-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _max_independent_set_size(adj, n):
    """Find the size of the maximum independent set using branch and bound."""
    # Use a simple backtracking with pruning
    best = [0]
    
    def backtrack(vertex, current_set, current_size):
        if current_size + (n - vertex) <= best[0]:
            return
        if vertex == n:
            best[0] = max(best[0], current_size)
            return
        
        # Option 1: Don't include vertex
        backtrack(vertex + 1, current_set, current_size)
        
        # Option 2: Include vertex if it's independent with current_set
        can_add = True
        for v in current_set:
            if adj[vertex][v]:
                can_add = False
                break
        
        if can_add:
            current_set.append(vertex)
            backtrack(vertex + 1, current_set, current_size + 1)
            current_set.pop()
    
    backtrack(0, [], 0)
    return best[0]

def _check_valid(n, edges):
    """Check if the graph is triangle-free and has no independent set of size 10."""
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Check triangle-free
    if not _is_triangle_free(adj, n):
        return False
    
    # Check no independent set of size 10
    # Use a faster check: just verify alpha < 10
    # For efficiency, use a greedy approach first, then exact if needed
    alpha = _max_independent_set_size(adj, n)
    return alpha < 10

def _generate_random_triangle_free(n, seed):
    """Generate a random triangle-free graph on n vertices."""
    rng = random.Random(seed)
    adj = [[False]*n for _ in range(n)]
    
    # Start with a random graph
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.3:
                adj[i][j] = True
                adj[j][i] = True
    
    # Remove edges to make it triangle-free
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            # Remove one of the edges to break the triangle
                            adj[i][k] = False
                            adj[k][i] = False
                            changed = True
                            break
            if changed:
                break
    
    # Collect edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # First, return the known Andrasfai graph as a baseline
    n_base = 26
    edges_base = [[i, j] for i in range(n_base) for j in range(i + 1, n_base) if (j - i) % n_base % 3 == 1]
    best_result = {'n': n_base, 'edges': edges_base}
    
    # Try to find a larger graph
    # Try n from 27 up to 40
    for n in range(27, 41):
        if time.time() > deadline:
            break
        
        # Try multiple random seeds
        found = False
        for seed in range(100):
            if time.time() > deadline:
                break
            
            edges = _generate_random_triangle_free(n, seed)
            
            # Quick check: is it triangle-free?
            adj = [[False]*n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = True
                adj[v][u] = True
            
            if not _is_triangle_free(adj, n):
                continue
            
            # Check alpha < 10
            alpha = _max_independent_set_size(adj, n)
            if alpha < 10:
                best_result = {'n': n, 'edges': edges}
                found = True
                break
        
        if found:
            # Found a valid graph of size n, try to go bigger
            continue
        else:
            # Could not find a valid graph of size n, stop
            break
    
    return best_result
