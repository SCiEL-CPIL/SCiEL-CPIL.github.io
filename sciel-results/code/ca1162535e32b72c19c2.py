import time as _time
import math
import random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """
    Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y:(x+y)%q
        sub = lambda x,y:(x-y)%q
        mul = lambda x,y:x*y%q
        
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
            
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140
    
    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, g=16
    # Try random voltage lifts of Petersen
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    e_count = len(petersen_edges)
    
    best_n = float('inf')
    best_result = None
    
    # Fallback LUW
    # Try d=7 first (n=4374), then d=8 (n=13122)
    for d in [7, 8, 9]:
        if _time.time() > deadline - 20:
            break
        try:
            n_luw, edges_luw = _luw_construct(3, g, d_override=d)
            # Verify girth of LUW
            adj = [set() for _ in range(n_luw)]
            for u, v in edges_luw:
                adj[u].add(v)
                adj[v].add(u)
            adj_list = [list(s) for s in adj]
            if _verify_girth_fast(adj_list, n_luw, g):
                if n_luw < best_n:
                    best_n = n_luw
                    best_result = (n_luw, edges_luw)
                break # Found valid LUW, no need for larger d
        except Exception:
            pass

    # Try Lifts
    # m from 50 to 500
    for m in range(50, 501, 10):
        if _time.time() > deadline - 10:
            break
        n = n_base * m
        if n >= best_n:
            break
        
        # Try a few random voltage assignments
        for trial in range(20):
            if _time.time() > deadline - 5:
                break
            shifts = [random.randint(0, m-1) for _ in range(e_count)]
            
            # Build lift
            adj = [set() for _ in range(n)]
            edges = []
            valid = True
            for idx, (u, v) in enumerate(petersen_edges):
                vol = shifts[idx]
                for i in range(m):
                    a = u * m + i
                    b = v * m + (i + vol) % m
                    if b in adj[a]:
                        # Duplicate edge or loop?
                        # In a simple graph lift, this shouldn't happen if base is simple
                        # and we treat edges as directed? 
                        # Petersen is simple.
                        # If (a,b) already in adj, it's a multi-edge.
                        # The task requires simple graph.
                        # So this lift construction might produce multi-edges if voltages collide?
                        # No, for a fixed base edge (u,v), the lifted edges are (u+i, v+i+vol).
                        # These are distinct for different i.
                        # However, if the base graph has multiple edges between u and v,
                        # or if we have a loop, we might have issues.
                        # Petersen is simple.
                        # But wait, if we have (u,v) and (v,u) in base? No, undirected.
                        # The code iterates over `petersen_edges`.
                        # If `petersen_edges` contains both (u,v) and (v,u), we'd add duplicates.
                        # My list `petersen_edges` contains each undirected edge once.
                        # So `a` and `b` are distinct.
                        # `adj[a].add(b)` and `adj[b].add(a)`.
                        # If we encounter the same pair again, it's a duplicate.
                        # This can happen if two different base edges map to the same pair of lifts?
                        # No, base edges are distinct.
                        # So `valid` should remain True unless we have a loop (a==b).
                        # a == b implies u*m + i == v*m + (i+vol)%m.
                        # u == v (loop in base) or u != v but indices wrap?
                        # u, v in 0..9. m >= 50.
                        # u*m + i and v*m + j are in different blocks if u != v.
                        # So a != b.
                        pass
                    
                    adj[a].add(b)
                    adj[b].add(a)
                    edges.append((a, b))
            
            # Check regularity
            for i in range(n):
                if len(adj[i]) != 3:
                    valid = False
                    break
            
            if valid:
                adj_list = [list(s) for s in adj]
                if _verify_girth_fast(adj_list, n, g):
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break # Found valid lift for this m, try next m (smaller n)

    if best_result is not None:
        return best_result
    
    # Final fallback: LUW with d=10 just in case
    n_luw, edges_luw = _luw_construct(3, g, d_override=10)
    return n_luw, edges_luw
