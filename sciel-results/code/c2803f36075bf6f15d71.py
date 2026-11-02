import time
import random
from collections import deque
import sys

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
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
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _voltage_graph_construct(k, g):
    if k != 5:
        return None
        
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n_base = 10
    # Left: 0..4, Right: 5..9
    base_edges = []
    edge_index = {}
    idx = 0
    for u in range(5):
        for v in range(5, 10):
            base_edges.append((u, v))
            edge_index[(u, v)] = idx
            edge_index[(v, u)] = idx
            idx += 1
            
    # Precompute cycles of length 4, 6, 8 in K_{5,5}
    # Represent as tuple of edge indices
    cycles_4 = []
    cycles_6 = []
    cycles_8 = []
    
    # 4-cycles: u1-v1-u2-v2-u1
    for u1 in range(5):
        for u2 in range(u1+1, 5):
            for v1 in range(5, 10):
                for v2 in range(v1+1, 10):
                    e1 = edge_index[(u1, v1)]
                    e2 = edge_index[(u2, v1)]
                    e3 = edge_index[(u2, v2)]
                    e4 = edge_index[(u1, v2)]
                    cycles_4.append((e1, e2, e3, e4))
                    
    # 6-cycles: u1-v1-u2-v2-u3-v3-u1
    # Choose 3 distinct u's and 3 distinct v's
    for u1 in range(5):
        for u2 in range(u1+1, 5):
            for u3 in range(u2+1, 5):
                us = [u1, u2, u3]
                for v1 in range(5, 10):
                    for v2 in range(v1+1, 10):
                        for v3 in range(v2+1, 10):
                            vs = [v1, v2, v3]
                            # 6 permutations of vs, 2 permutations of us[1:]?
                            # Actually, just generate all valid cycles
                            # Fix u1.
                            # Sequence: u1, va, ub, vb, uc, vc, u1
                            # va, vb, vc is a permutation of vs
                            # ub, uc is a permutation of us[1:]
                            for pv in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                                for pu in [(0,1), (1,0)]:
                                    u_seq = [u1, us[pu[0]], us[pu[1]]]
                                    v_seq = [vs[pv[0]], vs[pv[1]], vs[pv[2]]]
                                    
                                    e1 = edge_index[(u_seq[0], v_seq[0])]
                                    e2 = edge_index[(u_seq[1], v_seq[0])]
                                    e3 = edge_index[(u_seq[1], v_seq[1])]
                                    e4 = edge_index[(u_seq[2], v_seq[1])]
                                    e5 = edge_index[(u_seq[2], v_seq[2])]
                                    e6 = edge_index[(u_seq[0], v_seq[2])]
                                    
                                    # Avoid duplicates?
                                    # Each cycle is counted twice (direction).
                                    # We just add it. Duplicates are fine for checking sum==0.
                                    cycles_6.append((e1, e2, e3, e4, e5, e6))

    # 8-cycles: u1-v1-u2-v2-u3-v3-u4-v4-u1
    # Choose 4 distinct u's and 4 distinct v's
    for u1 in range(5):
        for u2 in range(u1+1, 5):
            for u3 in range(u2+1, 5):
                for u4 in range(u3+1, 5):
                    us = [u1, u2, u3, u4]
                    for v1 in range(5, 10):
                        for v2 in range(v1+1, 10):
                            for v3 in range(v2+1, 10):
                                for v4 in range(v3+1, 10):
                                    vs = [v1, v2, v3, v4]
                                    # Permutations of vs and us[1:]
                                    # This is getting large. 5! * 4! = 120 * 24 = 2880 per set of 4,4.
                                    # Number of sets: 5.
                                    # Total 8-cycles ~ 14400.
                                    # Manageable.
                                    import itertools
                                    for pv in itertools.permutations(range(4)):
                                        for pu in itertools.permutations(range(1, 4)):
                                            u_seq = [u1, us[pu[0]], us[pu[1]], us[pu[2]]]
                                            v_seq = [vs[pv[0]], vs[pv[1]], vs[pv[2]], vs[pv[3]]]
                                            
                                            e1 = edge_index[(u_seq[0], v_seq[0])]
                                            e2 = edge_index[(u_seq[1], v_seq[0])]
                                            e3 = edge_index[(u_seq[1], v_seq[1])]
                                            e4 = edge_index[(u_seq[2], v_seq[1])]
                                            e5 = edge_index[(u_seq[2], v_seq[2])]
                                            e6 = edge_index[(u_seq[3], v_seq[2])]
                                            e7 = edge_index[(u_seq[3], v_seq[3])]
                                            e8 = edge_index[(u_seq[0], v_seq[3])]
                                            
                                            cycles_8.append((e1, e2, e3, e4, e5, e6, e7, e8))

    all_cycles = cycles_4 + cycles_6 + cycles_8
    
    best_n = float('inf')
    best_edges = None
    
    # Search m from 50 to 150
    for m in range(50, 151):
        if time.time() > deadline:
            break
            
        n_lift = 10 * m
        if n_lift >= best_n:
            continue # No point in larger m if we already have a smaller graph?
                     # Wait, we want smallest n. So we should return the first valid m?
                     # Yes, m increases, so n increases.
                     # So we can return the first valid m.
                     
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            voltages = [random.randint(1, m-1) for _ in range(25)]
            
            valid = True
            for cyc in all_cycles:
                s = 0
                for e in cyc:
                    s += voltages[e]
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                # Build graph
                adjacency = [set() for _ in range(n_lift)]
                for u in range(5):
                    for v in range(5, 10):
                        w = voltages[edge_index[(u, v)]]
                        for g in range(m):
                            u_idx = u * m + g
                            v_idx = v * m + (g + w) % m
                            adjacency[u_idx].add(v_idx)
                            adjacency[v_idx].add(u_idx)
                
                # Verify girth just in case
                # BFS from 0
                dist = [-1] * n_lift
                dist[0] = 0
                q = deque([0])
                min_girth = float('inf')
                while q:
                    u = q.popleft()
                    for v in adjacency[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                        else:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                if min_girth >= 9:
                    edges = []
                    for u in range(n_lift):
                        for v in adjacency[u]:
                            if u < v:
                                edges.append((u, v))
                    return (n_lift, edges)
                
    return None

def construct(k, g):
    if k == 5 and g == 9:
        res = _voltage_graph_construct(k, g)
        if res is not None:
            return res
    return _luw_construct(k, g)
