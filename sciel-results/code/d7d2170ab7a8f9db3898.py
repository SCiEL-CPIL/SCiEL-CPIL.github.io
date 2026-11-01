import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety
    
    if k != 4 or g != 10:
        # Fallback for other parameters
        return _fallback_construct(k, g, start_time, deadline)
    
    # Specialized construction for (4,10)-cage
    # Try voltage graph on K4,4 with Z_40
    result = _try_voltage_k44(40, start_time, deadline)
    if result is not None:
        n, edges = result
        if _verify_graph(n, edges, 4, 10, start_time, deadline):
            return n, edges
            
    # Fallback: try other m values for K4,4
    for m in [42, 38, 44, 36, 46, 34, 48, 32]:
        if time.time() > deadline - 20:
            break
        result = _try_voltage_k44(m, start_time, deadline)
        if result is not None:
            n, edges = result
            if _verify_graph(n, edges, 4, 10, start_time, deadline):
                return n, edges

    # Final fallback: LUW construction
    return _luw_construct(4, 10, start_time, deadline)

def _try_voltage_k44(m, start_time, deadline):
    """Try to find voltages for K4,4 base with voltage group Z_m."""
    n_base = 8
    # Vertices 0-3 are A, 4-7 are B
    # Edges: (i, j) for i in 0..3, j in 4..7
    # Map base indices: 0,1,2,3 (A), 4,5,6,7 (B)
    # Let's use 0..3 for A and 0..3 for B in logic, then map to 0..7
    
    for attempt in range(500):
        if time.time() > deadline:
            return None
        
        # vols[a][b] for a in 0..3, b in 0..3
        # w_ab is voltage from A_a to B_b
        # w_ba is voltage from B_b to A_a
        w_ab = [[0]*4 for _ in range(4)]
        w_ba = [[0]*4 for _ in range(4)]
        
        valid = True
        for a in range(4):
            for b in range(4):
                w_ab[a][b] = random.randint(1, m-1)
                w_ba[b][a] = random.randint(1, m-1)
                if (w_ab[a][b] + w_ba[b][a]) % m == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Check girth
        short_cycle_found = False
        
        # Check cycles of length 4, 6, 8
        # Length 4: a1-b1-a2-b2-a1
        # a1 != a2, b1 != b2
        for a1 in range(4):
            for a2 in range(4):
                if a1 == a2: continue
                for b1 in range(4):
                    for b2 in range(4):
                        if b1 == b2: continue
                        # Cycle: a1 -> b1 -> a2 -> b2 -> a1
                        vol = (w_ab[a1][b1] + w_ba[b1][a2] + w_ab[a2][b2] + w_ba[b2][a1]) % m
                        if vol == 0:
                            short_cycle_found = True
                            break
                    if short_cycle_found: break
                if short_cycle_found: break
            if short_cycle_found: break
            if short_cycle_found: break
        
        if short_cycle_found:
            continue
            
        # Length 6: a1-b1-a2-b2-a3-b3-a1
        # a1,a2,a3 distinct, b1,b2,b3 distinct
        for a1 in range(4):
            for a2 in range(4):
                if a1 == a2: continue
                for a3 in range(4):
                    if a3 == a1 or a3 == a2: continue
                    for b1 in range(4):
                        for b2 in range(4):
                            if b2 == b1: continue
                            for b3 in range(4):
                                if b3 == b1 or b3 == b2: continue
                                # Cycle: a1->b1->a2->b2->a3->b3->a1
                                vol = (w_ab[a1][b1] + w_ba[b1][a2] + w_ab[a2][b2] + 
                                       w_ba[b2][a3] + w_ab[a3][b3] + w_ba[b3][a1]) % m
                                if vol == 0:
                                    short_cycle_found = True
                                    break
                                if short_cycle_found: break
                            if short_cycle_found: break
                        if short_cycle_found: break
                    if short_cycle_found: break
                if short_cycle_found: break
            if short_cycle_found: break
            if short_cycle_found: break
            
        if short_cycle_found:
            continue
            
        # Length 8: a1-b1-a2-b2-a3-b3-a4-b4-a1
        # a1..a4 distinct, b1..b4 distinct
        # This is a lot of permutations. 4! * 4! = 576.
        # We can optimize by iterating over permutations of A and B
        import itertools
        for perm_a in itertools.permutations(range(4)):
            if short_cycle_found: break
            for perm_b in itertools.permutations(range(4)):
                # Cycle: a1->b1->a2->b2->a3->b3->a4->b4->a1
                a1, a2, a3, a4 = perm_a
                b1, b2, b3, b4 = perm_b
                vol = (w_ab[a1][b1] + w_ba[b1][a2] + w_ab[a2][b2] + 
                       w_ba[b2][a3] + w_ab[a3][b3] + w_ba[b3][a4] + 
                       w_ab[a4][b4] + w_ba[b4][a1]) % m
                if vol == 0:
                    short_cycle_found = True
                    break
                    
        if short_cycle_found:
            continue
            
        # If we reach here, no short cycles found up to length 8.
        # Since graph is bipartite, no odd cycles.
        # Girth is at least 10.
        
        # Build graph
        edges = []
        # Base vertices: 0,1,2,3 (A), 4,5,6,7 (B)
        # Lift vertex (base_idx, a) -> base_idx * m + a
        for a in range(4):
            for b in range(4):
                base_u = a # A
                base_v = 4 + b # B
                w_uv = w_ab[a][b]
                w_vu = w_ba[b][a]
                
                for x in range(m):
                    u = base_u * m + x
                    v = base_v * m + ((x + w_uv) % m)
                    if u < v:
                        edges.append((u, v))
                    else:
                        edges.append((v, u))
                    
                    # The reverse edge is handled by the other direction?
                    # No, we are iterating over all undirected edges in base.
                    # The edge (u, v) is unique.
                    # We don't need to add (v, u) separately as it's the same edge.
        
        # Deduplicate
        edge_set = set()
        unique_edges = []
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in edge_set:
                edge_set.add(key)
                unique_edges.append(key)
                
        return (8 * m, unique_edges)
        
    return None

def _verify_graph(n, edges, k, g, start_time, deadline):
    """Verify graph is simple, k-regular, and has girth >= g."""
    if n == 0 or time.time() > deadline:
        return False
    
    adj = [[] for _ in range(n)]
    seen = set()
    
    for u, v in edges:
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Girth check using BFS from each vertex
    max_depth = g // 2
    for u in range(n):
        if time.time() > deadline:
            return False
        
        dist = [-1] * n
        parent = [-1] * n
        dist[u] = 0
        q = deque([u])
        
        while q:
            curr = q.popleft()
            if dist[curr] >= max_depth:
                continue
            
            for v in adj[curr]:
                if dist[v] == -1:
                    dist[v] = dist[curr] + 1
                    parent[v] = curr
                    q.append(v)
                else:
                    if v == parent[curr]:
                        continue
                    cycle_len = dist[curr] + 1 + dist[v]
                    if cycle_len < g:
                        return False
    
    return True

def _luw_construct(k, g, start_time, deadline):
    """LUW construction as fallback."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    
    def get_rule(d_val):
        r = dict(rule)
        for c in range(5, d_val + 1):
            rc, c0 = (c - 5) % 4, c - (c - 5) % 4
            r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
        return r
    
    min_d = max(3, g - 5)
    best_n = float('inf')
    best_edges = None
    
    for d_val in range(min_d, min_d + 10):
        if time.time() > deadline:
            break
        try:
            r = get_rule(d_val)
            
            def neighbours(vertex):
                side, x = vertex
                for first in range(q):
                    y = [None, first] + [0] * (d_val - 1)
                    for c in range(2, d_val + 1):
                        a, b = r[c]
                        y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                    yield 1 - side, tuple(y)
            
            start = (0, (None,) + (0,) * d_val)
            index, order, edges = {start: 0}, [start], []
            for vertex in order:
                for other in neighbours(vertex):
                    if other not in index:
                        index[other] = len(order)
                        order.append(other)
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
            
            n = len(order)
            if n < best_n and n < 50000:
                if _verify_graph(n, edges, k, g, start_time, deadline):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass
    
    if best_edges is not None:
        return best_n, best_edges
    
    return 0, []

def _fallback_construct(k, g, start_time, deadline):
    """Generic fallback for non-(4,10) cases."""
    return _luw_construct(k, g, start_time, deadline)
