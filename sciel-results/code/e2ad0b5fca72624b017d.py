"""Construct a 4-regular graph of girth >= g using a probabilistic random lift construction."""
import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    # For the specific case k=4, g=10 (the cage problem n(4,10))
    # We use a random lift construction:
    # Base graph: a 4-regular graph with small girth, then lift it
    # Actually, a better approach: use the known explicit constructions or
    # use random regular graphs with voltage assignment
    
    # The best known construction for n(4,10) gives 320 vertices.
    # The Lazebnik-Ustimenko-Woldar construction gives a specific graph.
    # Let me implement a direct construction approach.
    
    # For k=4, g=10: use the known 320-vertex cage or try to construct
    # a smaller one via random search.
    
    # Approach: Try to construct a 4-regular graph with girth >= 10
    # using a random regular graph approach with girth verification.
    
    # The Moore bound for n(4,10) is 1 + 4*sum_{i=0}^{4} 3^i = 1 + 4*(1+3+9+27+81) = 1 + 4*121 = 485
    # Wait, that's not right. The Moore bound for even girth 2r is:
    # n >= 1 + d * sum_{i=0}^{r-1} (d-1)^i for odd girth
    # n >= 2 * sum_{i=0}^{r-1} (d-1)^i for even girth
    # For g=10, r=5: n >= 2*(1+3+9+27+81) = 2*121 = 243
    
    # So the lower bound is 243, and the best known is 320.
    
    # Let me try a random approach: generate random 4-regular graphs and check girth.
    # But that's very hard for girth 10.
    
    # Better: use the voltage graph / lift approach.
    # Take a small 4-regular base graph and assign voltages from Z_M.
    
    # Actually, let me just implement the known best construction directly.
    # The 320-vertex (4,10)-cage can be constructed using the method of
    # Exoo et al. But implementing that from scratch is complex.
    
    # Alternative: use the Lazebnik-Ustimenko-Woldar construction which is explicit.
    # For q=4, the LUW construction gives a 4-regular graph.
    # The girth of CD(d,q) is at least d+5 for odd d.
    # So for girth 10, we need d+5 >= 10, so d >= 5.
    # With d=5, q=4: the graph has 2*q^d = 2*4^5 = 2048 vertices. That's too many.
    
    # Let me try a different approach: random regular graph with girth constraint.
    # Or use the known 320-vertex construction.
    
    # Actually, let me implement a simple random search for small n.
    # Start with n = 243 (lower bound) and try to find a valid graph.
    # This is likely infeasible by pure random search.
    
    # Let me just use the LUW construction with minimal parameters.
    # For q=4, d=5: 2*4^5 = 2048 vertices, girth >= 10.
    # For q=4, d=3: 2*4^3 = 128 vertices, girth >= 8. Not enough.
    # For q=4, d=4: 2*4^4 = 512 vertices, girth >= 9. Not enough.
    
    # So LUW with q=4, d=5 gives 2048 vertices. That's valid but not optimal.
    
    # Let me try to do better with a random lift approach.
    # Base: 4-cycle (C4), which is 2-regular. Not good.
    # Base: K4, which is 3-regular. Not 4-regular.
    # Base: K_{3,3}, which is 3-regular. Not 4-regular.
    # Base: Petersen graph, which is 3-regular.
    
    # For 4-regular base: 
    # - 4-regular graph on 8 vertices: the cube graph Q3 (but that's 3-regular)
    # - 4-regular on 6 vertices: K6 minus a perfect matching? That's 4-regular on 6 vertices.
    
    # Let me use K6 minus perfect matching as base (6 vertices, 4-regular).
    # Then lift with Z_M to get 6M vertices.
    
    # Actually, let me just go with the LUW construction as a safe fallback,
    # and try to improve with random search if time permits.
    
    # LUW construction for q=4, d=5:
    q = 4
    d = 5  # gives girth >= d+5 = 10
    
    # GF(4) operations
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def gf_add(x, y): return x ^ y
    def gf_sub(x, y): return x ^ y
    def gf_mul(x, y): return table[x][y]
    
    # Rule for LUW graph
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n_luw = len(order)
    best_n = n_luw
    best_edges = edges
    
    # Now try to improve with random lift if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 60.0 and k == 4 and g == 10:
        # Try random lift with K6-minus-matching base
        # K6 minus perfect matching: vertices 0-5, edges:
        base_edges = []
        for i in range(6):
            for j in range(i+1, 6):
                if (i + j) % 2 == 0:  # This doesn't give a perfect matching removal
                    base_edges.append((i, j))
        # Actually, K6 has 15 edges. Remove a perfect matching (3 edges) to get 12 edges, 4-regular.
        # Perfect matching: (0,3), (1,4), (2,5)
        pm = {(0,3), (1,4), (2,5)}
        base_edges = []
        for i in range(6):
            for j in range(i+1, 6):
                if (i,j) not in pm and (j,i) not in pm:
                    base_edges.append((i, j))
        # Check: each vertex should have degree 4
        # Vertex 0: connected to 1,2,4,5 (not 3) -> degree 4. Good.
        
        # Try different M values
        for M in [40, 42, 44, 46, 48, 50]:
            if time.time() > deadline - 30.0:
                break
            n_total = 6 * M
            if n_total >= best_n:
                continue
            
            # Random voltage assignment
            best_for_M = None
            for attempt in range(20):
                if time.time() > deadline - 30.0:
                    break
                voltages = [random.randint(0, M-1) for _ in range(len(base_edges))]
                
                # Build the lifted graph
                # Vertex (v, w) where v in 0..5, w in 0..M-1
                # Edge between (v, w) and (u, w + voltage) mod M for each base edge (v,u)
                adj = [[] for _ in range(n_total)]
                for (u, v), vol in zip(base_edges, voltages):
                    for w in range(M):
                        a = u * M + w
                        b = v * M + ((w + vol) % M)
                        adj[a].append(b)
                        adj[b].append(a)
                
                # Check girth using BFS from each vertex (limited depth)
                # For girth >= 10, we need no cycles of length < 10
                # Check for cycles of length 3, 4, 5, 6, 7, 8, 9
                has_short_cycle = False
                for start_v in range(n_total):
                    if has_short_cycle:
                        break
                    # BFS to depth 4 (to find cycles up to length 9)
                    # Actually, for non-backtracking walks, we need to be more careful
                    # Simple BFS checking for cycles:
                    visited = [[-1] * 2 for _ in range(n_total)]  # [prev, dist]
                    queue = [(start_v, -1, 0)]
                    while queue and not has_short_cycle:
                        v, prev, dist = queue.pop(0)
                        if visited[v][0] != -1:
                            if visited[v][0] != prev:
                                # Found a cycle
                                cycle_len = dist + visited[v][1] + 1
                                if cycle_len < g:
                                    has_short_cycle = True
                                    break
                            continue
                        visited[v][0] = prev
                        visited[v][1] = dist
                        if dist < 4:  # Only go to depth 4
                            for u in adj[v]:
                                if u != prev:
                                    queue.append((u, v, dist + 1))
                
                if not has_short_cycle:
                    # Check it's 4-regular
                    if all(len(adj[i]) == 4 for i in range(n_total)):
                        # Build edge list
                        edge_set = set()
                        for i in range(n_total):
                            for j in adj[i]:
                                if i < j:
                                    edge_set.add((i, j))
                        edge_list = list(edge_set)
                        if len(edge_list) == 4 * n_total // 2:
                            if best_for_M is None or n_total < best_for_M[0]:
                                best_for_M = (n_total, edge_list)
            
            if best_for_M is not None:
                if best_for_M[0] < best_n:
                    best_n = best_for_M[0]
                    best_edges = best_for_M[1]
    
    return best_n, best_edges
