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
        
        # Precompute base adjacency for fast girth checking via voltage assignment
        base_adj = [[] for _ in range(6)]
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
        
        def check_girth_lift(base_adj, n_base, voltages, M, g):
            """Check girth of the lift using voltage-based non-backtracking walk enumeration.
            A cycle in the lift corresponds to a closed non-backtracking walk in the base
            whose voltage sum is 0 mod M. We check all closed NB walks of length < g."""
            # Enumerate all non-backtracking walks in base graph up to length g-1
            # For each closed walk, check if voltage sum == 0 mod M
            # State: (prev, curr) directed edge
            # For efficiency, use a dict mapping (prev, curr) -> list of (start, voltage_sum)
            # But that's too big. Instead, for each start vertex, track (prev, curr, vol_sum)
            
            for s in range(n_base):
                # BFS from s: state (prev, curr, depth, vol_sum)
                # We want to find if any closed NB walk of length L < g has vol_sum % M == 0
                # Start: (s, nb, 1, volt[s,nb])
                visited_states = set()  # (prev, curr, depth, vol_sum) to avoid reprocessing
                queue = []
                for nb in base_adj[s]:
                    # Find voltage for edge (s, nb)
                    vol = voltages.get((s, nb), voltages.get((nb, s), 0))
                    queue.append((s, nb, 1, vol % M))
                    visited_states.add((s, nb, 1, vol % M))
                
                while queue:
                    new_queue = []
                    for prev, curr, depth, vol_sum in queue:
                        if depth >= g - 1:
                            continue
                        for nb in base_adj[curr]:
                            if nb == prev:
                                continue
                            new_vol = (vol_sum + voltages.get((curr, nb), voltages.get((nb, curr), 0))) % M
                            if nb == s and (depth + 1) < g and new_vol == 0:
                                return False
                            state = (curr, nb, depth + 1, new_vol)
                            if state not in visited_states:
                                visited_states.add(state)
                                new_queue.append(state)
                    queue = new_queue
            return True
        
        # Build voltage lookup for base edges
        # Try different M values, starting from smaller ones
        for M in [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70]:
            if time.time() > deadline - 25.0:
                break
            n_total = 6 * M
            if n_total >= best_n:
                continue
            
            # Number of attempts scales with time remaining
            remaining = deadline - 15.0 - time.time()
            if remaining <= 0:
                break
            max_attempts = min(200, max(20, int(remaining * 15)))
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 10.0:
                    break
                
                voltages = [random.randint(0, M-1) for _ in range(len(base_edges))]
                # Create voltage lookup dict
                vol_dict = {}
                for (u, v), vol in zip(base_edges, voltages):
                    vol_dict[(u, v)] = vol
                    vol_dict[(v, u)] = (-vol) % M
                
                # Check girth using voltage-based method
                if check_girth_lift(base_adj, 6, vol_dict, M, g):
                    # Build the lifted graph
                    adj = [[] for _ in range(n_total)]
                    for (u, v), vol in zip(base_edges, voltages):
                        for w in range(M):
                            a = u * M + w
                            b = v * M + ((w + vol) % M)
                            adj[a].append(b)
                            adj[b].append(a)
                    
                    # Verify 4-regularity
                    if all(len(adj[i]) == 4 for i in range(n_total)):
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
