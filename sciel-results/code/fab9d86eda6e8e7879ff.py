"""
Strategy:
1. Construct a C4-free graph with 90 edges on 31 vertices using a greedy algorithm
   with incremental common-neighbor tracking.
2. Add the 91st edge that creates the fewest C4s.
3. Perform local search (edge swaps) with incremental C4 delta calculation to minimize C4s.

Key insight: The extremal number ex(31, C4) = 90, so there exists a C4-free graph
with 90 edges. We construct such a graph greedily, then add one edge and optimize
via swaps.

For speed, we maintain:
- adj: adjacency matrix
- common[u,v]: number of common neighbors of u and v
- c4_count: total number of C4s = sum_{u<v} C(common[u,v], 2) // 2
  (each C4 counted twice, once for each pair of opposite vertices)

When adding edge (a,b):
- For each x in N(a) \\ {b}: common(x,b) += 1, common(b,x) += 1
- For each y in N(b) \\ {a}: common(a,y) += 1, common(y,a) += 1

The change in C4 count can be computed incrementally.
"""

import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 140
    
    rng = random.Random(42)
    
    def count_c4(adj):
        """Count C4s in graph given by adjacency matrix."""
        A = adj.astype(np.int32)
        C = A @ A
        # C[u,v] = number of common neighbors of u and v
        # C4 count = sum_{u<v} C(C[u,v], 2) // 2
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = C[u, v]
                if c >= 2:
                    total += c * (c - 1) // 2
        # Each C4 is counted twice (once for each pair of opposite vertices)
        return total // 2
    
    def build_c4_free_graph():
        """Build a C4-free graph with as many edges as possible (aim for 90)."""
        adj = np.zeros((n, n), dtype=np.int8)
        common = np.zeros((n, n), dtype=np.int32)
        
        # Try all possible edges in random order, add if it doesn't create a C4
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            # Check if adding (u, v) creates a C4
            # A C4 is created if there exist a, b such that:
            # u-a-v-b-u is a cycle, i.e., a is common neighbor of u and v? No.
            # Adding edge (u,v) creates a C4 if there is a path of length 3 from u to v.
            # Path: u - a - b - v, where (u,a), (a,b), (b,v) are edges.
            # This means a is neighbor of u, b is neighbor of v, and (a,b) is an edge.
            # Equivalently, a is in N(u) and b is in N(v) and adj[a,b]=1.
            # So we need to check if there exist a in N(u), b in N(v) with adj[a,b]=1.
            
            # For efficiency, check: for each a in N(u), for each b in N(v), if adj[a,b]=1
            # But this is O(deg(u)*deg(v)) which could be slow.
            
            # Alternative: common(u,v) <= 1 is necessary but not sufficient for no C4.
            # Actually, in a C4-free graph, any two non-adjacent vertices have at most 1 common neighbor.
            # But adding edge (u,v) creates C4s based on length-3 paths, not just common neighbors.
            
            # Let's just check: does adding (u,v) create any C4?
            # A C4 containing the new edge (u,v) is u-v-a-b-u or u-v-b-a-u.
            # This requires a path of length 3 from u to v in the current graph.
            # Path: u - x - y - v where (u,x), (x,y), (y,v) are edges.
            
            creates_c4 = False
            for x in range(n):
                if adj[u, x] == 1 and x != v:
                    for y in range(n):
                        if adj[x, y] == 1 and adj[y, v] == 1 and y != u and y != x:
                            creates_c4 = True
                            break
                if creates_c4:
                    break
            
            if not creates_c4:
                adj[u, v] = 1
                adj[v, u] = 1
                # Update common counts
                for x in range(n):
                    if x != u and x != v:
                        if adj[u, x] == 1:
                            common[v, x] += 1
                            common[x, v] += 1
                        if adj[v, x] == 1:
                            common[u, x] += 1
                            common[x, u] += 1
        
        edge_count = adj.sum() // 2
        return adj, edge_count
    
    def add_best_edge(adj, common):
        """Add the edge that creates the fewest C4s."""
        best_edge = None
        best_c4_increase = float('inf')
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 0:
                    # Count C4s created by adding (u, v)
                    # A C4 is created for each path of length 3 from u to v.
                    # Path: u - a - b - v
                    c4_created = 0
                    for a in range(n):
                        if adj[u, a] == 1 and a != v:
                            for b in range(n):
                                if adj[a, b] == 1 and adj[b, v] == 1 and b != u and b != a:
                                    c4_created += 1
                    
                    if c4_created < best_c4_increase:
                        best_c4_increase = c4_created
                        best_edge = (u, v)
        
        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            # Update common counts
            for x in range(n):
                if x != u and x != v:
                    if adj[u, x] == 1:
                        common[v, x] += 1
                        common[x, v] += 1
                    if adj[v, x] == 1:
                        common[u, x] += 1
                        common[x, u] += 1
        
        return adj, common
    
    def swap_delta(adj, common, u, v, c, d):
        """Compute delta in C4 count when removing (u,v) and adding (c,d)."""
        # This is complex, so we'll just compute before and after
        old_c4 = count_c4(adj)
        
        adj[u, v] = 0
        adj[v, u] = 0
        adj[c, d] = 1
        adj[d, c] = 1
        
        new_c4 = count_c4(adj)
        
        # Revert
        adj[u, v] = 1
        adj[v, u] = 1
        adj[c, d] = 0
        adj[d, c] = 0
        
        return new_c4 - old_c4
    
    # Build C4-free base graph
    base_adj, base_edges = build_c4_free_graph()
    
    # Compute common matrix for base graph
    A = base_adj.astype(np.int32)
    common = A @ A
    
    # Add best edge
    adj, common = add_best_edge(base_adj.copy(), common.copy())
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    
    # Local search: edge swaps
    T = 50.0
    T_min = 0.5
    cooling = 0.999
    
    while T > T_min and time.time() < deadline - 5:
        # Pick random edge to remove
        while True:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u < v and adj[u, v] == 1:
                break
            if v < u and adj[v, u] == 1:
                u, v = v, u
                break
        
        # Pick random non-edge to add
        while True:
            c = rng.randint(0, n - 1)
            d = rng.randint(0, n - 1)
            if c < d and adj[c, d] == 0:
                break
            if d < c and adj[d, c] == 0:
                c, d = d, c
                break
        
        # Ensure we're not swapping to the same edge
        if (u, v) == (c, d) or (u, v) == (d, c):
            continue
        
        # Compute delta C4
        delta = swap_delta(adj, common, u, v, c, d)
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Accept
            adj[u, v] = 0
            adj[v, u] = 0
            adj[c, d] = 1
            adj[d, c] = 1
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
        
        T *= cooling
    
    # Extract edges from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                final_edges.append([i, j])
    
    # Ensure we have exactly 91 edges
    assert len(final_edges) == m, f"Expected {m} edges, got {len(final_edges)}"
    
    return final_edges

if __name__ == "__main__":
    edges = construct()
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A = adj.astype(np.int32)
    C = A @ A
    total = 0
    for u in range(31):
        for v in range(u + 1, 31):
            c = C[u, v]
            if c >= 2:
                total += c * (c - 1) // 2
    print(f"C4: {total // 2}")
