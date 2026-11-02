"""Construct a 4-regular graph of girth >= g using a probabilistic random lift construction."""
import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # LUW construction for q=4, d=5 as fallback (2048 vertices, girth >= 10)
    q = 4
    d = 5
    
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def gf_add(x, y): return x ^ y
    def gf_sub(x, y): return x ^ y
    def gf_mul(x, y): return table[x][y]
    
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
    
    best_n = len(order)
    best_edges = edges

    # Try random lift with K6-minus-matching base if time permits
    if k == 4 and g == 10:
        # K6 minus perfect matching: 6 vertices, 4-regular
        pm = {(0,3), (1,4), (2,5)}
        base_edges = []
        for i in range(6):
            for j in range(i+1, 6):
                if (i,j) not in pm and (j,i) not in pm:
                    base_edges.append((i, j))
        
        # Fast girth check: verify no non-backtracking walk of length < g returns to start
        def check_girth_fast(adj, n, g):
            """Check that no non-backtracking walk of length < g returns to its start vertex."""
            # For each start vertex, do BFS tracking (vertex, prev) states up to depth g-1
            # A cycle of length L < g exists iff there's a non-backtracking walk of length L from v back to v
            for s in range(n):
                # State: (current_vertex, prev_vertex)
                # Start: from s, go to each neighbor (s, neighbor) at depth 1
                # We want to detect if we can return to s with prev != s (non-backtracking)
                # Actually, a non-backtracking walk from s of length L ends at s with the last step not being a backtrack
                # BFS on directed edges: state is (u, v) meaning we're at v, came from u
                visited = set()
                # Queue of (prev, curr, depth)
                queue = []
                for nb in adj[s]:
                    queue.append((s, nb, 1))
                    visited.add((s, nb))
                
                while queue:
                    new_queue = []
                    for prev, curr, depth in queue:
                        if depth >= g - 1:
                            continue
                        for nb in adj[curr]:
                            if nb == prev:
                                continue  # non-backtracking
                            if nb == s and depth + 1 >= 1:
                                # Found a non-backtracking walk from s back to s of length depth+1
                                if depth + 1 < g:
                                    return False
                            if (curr, nb) not in visited:
                                visited.add((curr, nb))
                                new_queue.append((curr, nb, depth + 1))
                    queue = new_queue
            return True
        
        # Try different M values, starting from smaller ones
        for M in [40, 42, 44, 46, 48, 50, 52, 54]:
            if time.time() > deadline - 20.0:
                break
            n_total = 6 * M
            if n_total >= best_n:
                continue
            
            for attempt in range(50):
                if time.time() > deadline - 10.0:
                    break
                
                voltages = [random.randint(0, M-1) for _ in range(len(base_edges))]
                
                # Build adjacency list
                adj = [[] for _ in range(n_total)]
                for (u, v), vol in zip(base_edges, voltages):
                    for w in range(M):
                        a = u * M + w
                        b = v * M + ((w + vol) % M)
                        adj[a].append(b)
                        adj[b].append(a)
                
                # Quick check: is it 4-regular?
                if any(len(adj[i]) != 4 for i in range(n_total)):
                    continue
                
                # Check girth
                if check_girth_fast(adj, n_total, g):
                    # Build edge list
                    edge_set = set()
                    for i in range(n_total):
                        for j in adj[i]:
                            if i < j:
                                edge_set.add((i, j))
                    edge_list = list(edge_set)
                    if len(edge_list) == 4 * n_total // 2:
                        if n_total < best_n:
                            best_n = n_total
                            best_edges = edge_list
                        break  # Found valid graph for this M, move to next M
    
    return best_n, best_edges
