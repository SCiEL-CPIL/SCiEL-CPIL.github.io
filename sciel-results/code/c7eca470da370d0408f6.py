"""
Construct a C5-free graph with no independent set of size 9, maximizing n.

Strategy:
  1. Start with 8 disjoint K4's (n=32, alpha=8, C5-free) as a valid baseline.
  2. Try to add more vertices by using a search for a C5-free graph with alpha < 9.
  3. Use a greedy/local-search approach within a time budget to find a larger valid graph.
  4. Return the best valid graph found within the time limit.

Key insight: The complement of a C5-free graph with alpha < 9 is a C5-free graph (since C5 is self-complementary) with clique number < 9.
Actually, C5 is self-complementary, so C5-free is preserved under complementation.
alpha(G) < 9  <=>  omega(complement(G)) < 9.

So we need a C5-free graph with clique number at most 8.

Alternative approach: Use known constructions.
The best known lower bound for R(C5, K9) is 32 (from 8 K4's).
Upper bound is 35.

Let's try to find a graph with n=33, 34, or 35.

Approach: Use a local search / greedy construction.
Start with a random graph or a structured graph, then:
  - Add edges to eliminate C5s (if any)
  - Remove edges to reduce alpha (if alpha >= 9)
  - Try to add vertices

Actually, a better approach for this specific problem:
Use the fact that the complement of a C5-free graph is also C5-free.
We want: C5-free AND alpha < 9.

Let's use a different strategy:
- Start with the 8 K4's graph (valid, n=32).
- Try to merge blocks or add cross-edges to allow more vertices.
- Or use a different base construction.

Known result: The Mycielski construction can increase n while preserving C5-freeness and controlling alpha, but it's complex.

Let's try a direct search approach:
- Use a time-bounded local search.
- Represent the graph as an adjacency matrix.
- Start from the 8 K4's graph.
- Try to add a new vertex and connect it appropriately.
- Check C5-freeness and alpha < 9.

For checking C5: A graph is C5-free if and only if for every pair of non-adjacent vertices u,v, there is no path of length 4 between them that forms a C5. More precisely, we can check by iterating over all 5-tuples or use a more efficient method.

Efficient C5 check: For each pair (u,v) that are not adjacent, check if there exist vertices a,b,c such that u-a-b-c-v is a path of length 4 with all distinct vertices and u not adjacent to b, c (to ensure it's a simple cycle). Actually, a C5 is a cycle of length 5. 

A simpler approach for small n (<=35): enumerate all 5-cycles. There are O(n^5) possible 5-tuples, but we can do better. For each edge (u,v), check if there's a path of length 3 from v to u going through 3 other vertices. 

Actually, for n<=35, we can use: for each 5-tuple (v0,v1,v2,v3,v4) with distinct vertices, check if v0-v1-v2-v3-v4-v0 is a cycle. This is O(n^5) = 35^5 ≈ 52M, which is too slow in Python.

Better: For each pair of vertices (u,v) that are not adjacent, the number of common neighbors... no, that's for triangles.

Let me use a different approach: 
A graph contains a C5 if and only if there exist vertices v0,v1,v2,v3,v4 such that:
  v0~v1, v1~v2, v2~v3, v3~v4, v4~v0, and all other pairs are non-adjacent (for induced C5) or just the cycle edges exist (for subgraph C5).

The task says "no 5-cycle (as a subgraph)", so we need to check for C5 as a subgraph, not necessarily induced. So we need: v0~v1, v1~v2, v2~v3, v3~v4, v4~v0 with all vertices distinct.

For n=35, we can check: for each vertex v0, for each neighbor v1 of v0, for each neighbor v2 of v1 (v2 != v0), for each neighbor v3 of v2 (v3 != v0, v1), for each neighbor v4 of v3 (v4 != v0, v1, v2), check if v4 is adjacent to v0 and v4 != v1, v2. This is O(sum of products of degrees), which could be large but manageable for n=35 with moderate density.

Let me implement this check efficiently.

For alpha (independence number): For n<=35, we can use a branch-and-bound or simply check all subsets of size 9. C(35,9) ≈ 70M, which is too slow. Better to use a maximum independent set algorithm.

Actually, for the purpose of this task, we can use a simple greedy + local search for alpha, or use the fact that alpha < 9 means the maximum independent set has size at most 8.

Let me use a different strategy: use a precomputed or known construction.

After research, the best known construction for R(C5, K9) > 32 is indeed 8 K4's. It's an open problem whether R(C5, K9) > 33, 34, or 35.

Given the time constraint, let me try a local search to see if we can find n=33, 34, or 35.

Plan:
1. Start with 8 K4's (n=32).
2. Try to add vertices one by one.
3. For each new vertex, try all possible connections to existing vertices.
4. Check C5-freeness and alpha < 9.
5. Use local search to fix violations.

Let me implement this with a time budget.
"""

import time
import random
from itertools import combinations

DEADLINE = 140  # seconds, safety margin before 150s kill

def _has_c5(adj, n):
    """Check if graph with adjacency matrix adj on n vertices contains a C5 subgraph."""
    # For each 5-tuple (v0, v1, v2, v3, v4) with distinct vertices,
    # check if v0-v1-v2-v3-v4-v0 is a cycle.
    # Efficient: iterate over edges and paths.
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1 + 1, n):
                if v2 == v0:
                    continue
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2 + 1, n):
                    if v3 in (v0, v1):
                        continue
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3 + 1, n):
                        if v4 in (v0, v1, v2):
                            continue
                        if not adj[v3][v4]:
                            continue
                        if adj[v4][v0] and v4 != v1 and v4 != v2:
                            # Check all vertices distinct
                            if len({v0, v1, v2, v3, v4}) == 5:
                                return True
    return False

def _max_independent_set_size(adj, n, timeout=None):
    """Compute the size of the maximum independent set using branch and bound."""
    # Use a simple branch and bound
    best = 0
    visited = [False] * n
    
    def dfs(start, current, size):
        nonlocal best
        if size > best:
            best = size
        if start >= n:
            return
        # Upper bound: size + (n - start)
        if size + (n - start) <= best:
            return
        for i in range(start, n):
            # Check if vertex i can be added
            can_add = True
            for j in range(len(current)):
                if adj[current[j]][i]:
                    can_add = False
                    break
            if can_add:
                current.append(i)
                dfs(i + 1, current, size + 1)
                current.pop()
    
    dfs(0, [], 0)
    return best

def _alpha_lt_9(adj, n):
    """Check if the independence number is less than 9."""
    # Quick check: if any vertex has degree < n-9, that's not sufficient.
    # Use a simpler check: try to find an independent set of size 9.
    # If we find one, return False.
    # Use a greedy + backtracking approach.
    
    # For n <= 35, we can try all combinations of 9 vertices.
    # C(35, 9) ~ 70M, too slow.
    # Use a smarter approach: branch and bound to see if there's an independent set of size 9.
    
    def has_independent_set_size_k(k, adj, n):
        # Branch and bound to find if there's an independent set of size k
        if k <= 0:
            return True
        if n < k:
            return False
        
        # Pick a vertex with minimum degree among remaining
        candidates = list(range(n))
        if not candidates:
            return False
        
        # Sort by degree (ascending) for better pruning
        candidates.sort(key=lambda v: sum(adj[v]))
        
        for v in candidates:
            # Include v: then we can't include any neighbor of v
            new_candidates = [u for u in candidates if u != v and not adj[v][u]]
            if len(new_candidates) >= k - 1:
                if has_independent_set_size_k(k - 1, adj, len(new_candidates)) or \
                   _has_independent_set_size_k(new_candidates, k - 1, adj):
                    return True
            # Exclude v
            remaining = [u for u in candidates if u != v]
            if len(remaining) >= k:
                if _has_independent_set_size_k(remaining, k, adj):
                    return True
        return False
    
    def _has_independent_set_size_k(candidates, k, adj):
        if k <= 0:
            return True
        if len(candidates) < k:
            return False
        if k == 1:
            return True
        
        # Pick vertex with min degree in candidates
        min_deg = float('inf')
        min_v = candidates[0]
        for v in candidates:
            deg = sum(adj[v][u] for u in candidates if u != v)
            if deg < min_deg:
                min_deg = deg
                min_v = v
        
        # Include min_v
        new_cands = [u for u in candidates if u != min_v and not adj[min_v][u]]
        if len(new_cands) >= k - 1:
            if _has_independent_set_size_k(new_cands, k - 1, adj):
                return True
        
        # Exclude min_v
        remaining = [u for u in candidates if u != min_v]
        if len(remaining) >= k:
            if _has_independent_set_size_k(remaining, k, adj):
                return True
        
        return False
    
    return not _has_independent_set_size_k(list(range(n)), 9, adj)

def _is_valid(adj, n):
    """Check if the graph is C5-free and has alpha < 9."""
    if _has_c5(adj, n):
        return False
    if not _alpha_lt_9(adj, n):
        return False
    return True

def construct():
    start_time = time.time()
    
    # Start with 8 K4's
    n = 32
    adj = [[False] * n for _ in range(n)]
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                adj[base + i][base + j] = True
                adj[base + j][base + i] = True
    
    best_n = n
    best_adj = [row[:] for row in adj]
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                best_edges.append([i, j])
    
    # Try to add vertices
    current_n = n
    current_adj = [row[:] for row in adj]
    
    for attempt in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try to add a new vertex
        new_n = current_n + 1
        if new_n > 35:
            break
        
        new_adj = [row[:] + [False] for row in current_adj]
        new_adj.append([False] * new_n)
        
        # Try different connection patterns for the new vertex
        # The new vertex is index current_n
        found = False
        for mask in range(1 << min(current_n, 10)):  # Try first 10 vertices for connection
            new_adj2 = [row[:] for row in new_adj]
            for i in range(min(current_n, 10)):
                if mask & (1 << i):
                    new_adj2[current_n][i] = True
                    new_adj2[i][current_n] = True
            
            if _is_valid(new_adj2, new_n):
                current_n = new_n
                current_adj = new_adj2
                best_n = new_n
                best_adj = [row[:] for row in current_adj]
                best_edges = []
                for i in range(current_n):
                    for j in range(i + 1, current_n):
                        if current_adj[i][j]:
                            best_edges.append([i, j])
                found = True
                break
        
        if not found:
            # Try random connections
            for _ in range(50):
                if time.time() - start_time > DEADLINE:
                    break
                new_adj2 = [row[:] for row in new_adj]
                for i in range(current_n):
                    if random.random() < 0.3:
                        new_adj2[current_n][i] = True
                        new_adj2[i][current_n] = True
                if _is_valid(new_adj2, new_n):
                    current_n = new_n
                    current_adj = new_adj2
                    best_n = new_n
                    best_adj = [row[:] for row in current_adj]
                    best_edges = []
                    for i in range(current_n):
                        for j in range(i + 1, current_n):
                            if current_adj[i][j]:
                                best_edges.append([i, j])
                    found = True
                    break
        
        if not found:
            # Can't add more vertices, break
            break
    
    return {'n': best_n, 'edges': best_edges}
