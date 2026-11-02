import time
import random
import sys
from collections import deque
from itertools import product

def construct(k, g):
    deadline = time.time() + 115

    # ---- LUW construction (fallback baseline) ----
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
    n_luw, edges_luw = len(order), edges

    # ---- Non-backtracking walk certification ----
    def certify(adj, n, g):
        # adj: list of lists of neighbours (undirected)
        # Check: no cycle shorter than g
        # For each vertex, do non-backtracking walk; if we return to start with < g steps -> cycle < g
        # More efficient: for each directed edge (u,v), do BFS in non-backtracking space
        # State: (current, prev), distance
        # We want to detect if there's a cycle of length < g
        # Equivalent: for each vertex v, check if there's a walk of length L < g that
        # starts at v and returns to v, with no immediate backtracking.
        # Simpler: build the non-backtracking matrix and check powers? Too slow for large n.
        # Use: for each directed edge (u,v), BFS non-backtracking; if we hit an edge that
        # forms a cycle of total length < g, fail.
        
        # For efficiency, use: for each vertex s, BFS in the non-backtracking graph from
        # state (s, nb) for each nb of s. If we ever reach a state (x, s) where the path
        # from (s, nb) to (x, s) has length < g-2, we found a cycle of length < g.
        
        # Actually, simpler check: build adjacency as list of lists.
        # For each vertex s, for each start neighbour nb of s:
        #   do BFS where state = (current_node, prev_node), start = (nb, s), dist = 1
        #   target: reach (s, x) for any x != nb (which means we completed a cycle back to s)
        #   if distance < g -> cycle of length < g
        #   But we also need to make sure we don't count the trivial "walk back".
        #   The cycle length = distance + 1 (edge from s to nb was already counted as 1)
        #   Wait, let me re-think.
        #   
        # Non-backtracking walk: s -> nb -> ... -> s
        # The first step is s->nb (length 1). Then non-backtracking continuation.
        # If we reach s from some node x (x != nb to avoid immediate backtrack to nb... 
        # actually non-backtracking means we can't go x->s if we came from s... wait,
        # the state (x, prev) means we're at x having come from prev. Non-backtracking
        # means next step can't go to prev.
        # 
        # Start: state (nb, s), dist=1. We want to reach any state (s, prev) with prev != nb,
        # because that means the last edge is prev->s, completing a cycle of length dist+1.
        # Actually if we're in state (s, prev) with prev != nb, the walk was s->nb->...->prev->s
        # which has length dist (current) + 1? No, let me be careful.
        #
        # State (v, p) with dist d means: we have a non-backtracking walk of length d
        # ending at v, having arrived from p. The walk is p -> v was the last step.
        # Start: s -> nb, so state is (nb, s), dist = 1.
        # If we reach state (s, p) with p != nb, dist = D, that means we have a walk
        # nb -> ... -> p -> s of length D-1 (from the start s->nb being step 1, the
        # remaining D-1 steps get us from nb to s via p). Total cycle length = 1 + (D-1) = D.
        # So if D < g, we found a cycle of length D < g.
        #
        # Also need p != nb to avoid the 2-cycle s->nb->s (but that's length 2, and g>=6,
        # so it would be caught; but non-backtracking already prevents it since from
        # (nb, s) we can't go to s).
        
        for s in range(n):
            for nb in adj[s]:
                # BFS from state (nb, s)
                visited = {(nb, s)}
                queue = deque()
                queue.append((nb, s, 1))
                found_short = False
                while queue:
                    if time.time() > deadline:
                        return False
                    v, p, dist = queue.popleft()
                    if v == s and p != nb:
                        if dist < g:
                            found_short = True
                            break
                        continue
                    for u in adj[v]:
                        if u == p:
                            continue  # non-backtracking
                        if (u, v) not in visited:
                            visited.add((u, v))
                            queue.append((u, v, dist + 1))
                if found_short:
                    return False
        return True

    # ---- Attempt to improve via smart edge switching / local search on LUW graph ----
    # The LUW graph for k=3, g=16 has 1280 vertices (based on d=11 or similar).
    # Let's see what n_luw actually is, and try to reduce.
    
    # Build adjacency from edges_luw
    adj_luw = [set() for _ in range(n_luw)]
    for (u, v) in edges_luw:
        adj_luw[u].add(v)
        adj_luw[v].add(u)
    
    # Convert to list for speed
    adj_luw = [list(s) for s in adj_luw]
    
    best_n = n_luw
    best_edges = edges_luw
    best_adj = adj_luw

    # Verify LUW graph has girth >= g
    luw_valid = certify(best_adj, best_n, g)
    
    # ---- Strategy: try to find a smaller graph using a different construction ----
    # For k=3, g=16: Moore bound is 1 + 3*(1 + 2 + 4 + 8 + 16) = 1 + 3*31 = 94? No.
    # Moore bound for girth g (even): n >= 2 * k * ((k-1)^{g/2 - 1} - 1)/(k-2)
    # For k=3, g=16: n >= 2*3*((2^7 - 1)/(1)) = 6*127 = 762
    # Best known is 936, best lower bound 512.
    
    # The LUW construction gives a specific size. Let's try to construct smaller ones.
    
    # Approach: Use the fact that for k=3, we can try different parameterizations.
    # Also try: random k-regular graph with girth >= g via smart construction.
    
    # Strategy: Build graph level by level (BFS tree) with random choices, ensuring
    # no short cycles are formed.
    
    def try_level_bfs_graph(k, g, max_attempts=50, seed_offset=0):
        rng = random.Random(42 + seed_offset)
        target_n = None
        # Estimate: the graph is roughly a tree of depth g/2 - 1 from each vertex
        # Level 0: 1, Level 1: k, Level 2: k*(k-1), ..., Level g/2-1: k*(k-1)^{g/2-1}
        # Total vertices ≈ 2 * sum_{i=0}^{g/2-1} k*(k-1)^i - 1 (for even girth)
        # For k=3, g=16: g/2-1 = 7
        # Sum = 3*(1+2+4+8+16+32+64+128) = 3*255 = 765, doubled minus... 
        # Actually the "sphere" around a vertex has 1 + 3 + 6 + 12 + ... = 1 + 3*255 = 766 vertices
        # Two such spheres share... this is getting complex.
        
        # Just try various target sizes and see if we can construct.
        # The LUW gives n_luw. Try to do better.
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # We'll build a graph by assigning each vertex a "position" in a tree-like structure
            # and then connecting leaves at the appropriate depth.
            
            # For simplicity, use a different approach: 
            # Start with k disjoint cycles of length >= g, then connect them.
            # No, that won't give girth >= g easily.
            
            # Better: use the "cage" construction via linear algebra over GF(2) or similar.
            # Or: try the (3, 16)-cage known constructions.
            
            # Practical approach: try to reduce the LUW graph by finding and removing
            # redundant structure, or try a different (d, q) parameterization.
            pass
        
        return best_n, best_edges, best_adj

    # ---- Try De Bruijn-like construction ----
    # For k=3, g=16: use de Bruijn graph B(3, m) with m such that girth >= g.
    # B(3,m) has 3^m vertices, is 3-regular directed, girth related to m.
    # The undirected version (symmetrizing) has 3^m vertices.
    # Girth of B(3,m) is about 2m or so. We need 2m >= g => m >= 8.
    # B(3,8) has 3^8 = 6561 vertices - way too big.
    # B(3,5) has 243 vertices, girth about 10 - not enough.
    # This approach gives too many vertices.
    
    # ---- Try: modified LUW with different parameters ----
    # The LUW construction depends on (q, d, rule). Try different d values.
    
    for d_try in range(max(3, g - 8), g - 2):
        if time.time() > deadline:
            break
        try:
            rule2 = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d_try + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule2[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
            def neighbours_luw2(vertex, d_val=d_try, rule_val=rule2):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d_val - 1)
                    for c in range(2, d_val + 1):
                        a, b = rule_val[c]
                        y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)
            
            start2 = (0, (None,) + (0,) * d_try)
            index2, order2, edges2 = {start2: 0}, [start2], []
            for vertex in order2:
                for other in neighbours_luw2(vertex):
                    if other not in index2:
                        index2[other] = len(order2)
                        order2.append(other)
                    if vertex[0] == 0:
                        edges2.append((index2[vertex], index2[other]))
            
            n2 = len(order2)
            if n2 < best_n:
                adj2 = [set() for _ in range(n2)]
                for (u, v) in edges2:
                    adj2[u].add(v)
                    adj2[v].add(u)
                adj2 = [list(s) for s in adj2]
                # Verify
                if certify(adj2, n2, g):
                    best_n = n2
                    best_edges = edges2
                    best_adj = adj2
        except Exception:
            continue
    
    # ---- Return best found ----
    # Ensure edges are a list of tuples
    best_edges = [(min(u,v), max(u,v)) for (u,v) in best_edges]
    # Deduplicate
    best_edges = list(set(best_edges))
    best_edges.sort()
    
    return best_n, best_edges
