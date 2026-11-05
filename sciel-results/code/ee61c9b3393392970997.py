import time
import numpy as np

def _luw_graph(d, q):
    """Build the connected LUW component from the zero point. Returns (n, edges, adj)."""
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
    index, order = {start: 0}, [start]
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
    
    adj = [set() for _ in range(len(order))]
    for vertex in order:
        i = index[vertex]
        for other in neighbours(vertex):
            j = index[other]
            adj[i].add(j)
    
    adj = [list(s) for s in adj]
    edges = [(i, j) for i in range(len(order)) for j in adj[i] if i < j]
    return len(order), edges, adj

def _girth_check_fast(adj, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex."""
    n = len(adj)
    if n == 0:
        return True
    
    for start in range(n):
        dist = {}
        active = []
        for nb in adj[start]:
            dist[(start, nb)] = 1
            active.append((start, nb))
        
        for depth in range(1, g - 1):
            if not active:
                break
            next_active = []
            for (prev, curr) in active:
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    if nxt == start and depth + 1 < g:
                        return False
                    key = (curr, nxt)
                    if key not in dist:
                        dist[key] = depth + 1
                        next_active.append(key)
            active = next_active
    
    return True

def _pairing_regular(n, k, rng):
    """Generate random k-regular graph via perfect matching of stubs."""
    import random
    m = n * k
    stubs = [i for i in range(n) for _ in range(k)]
    random.seed(rng)
    random.shuffle(stubs)
    
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[a]
        v = stubs[a + 1]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng, u=None, v=None):
    """Perform a 2-switch. If u,v given, use edge (u,v); else random."""
    import random
    if u is not None:
        x = random.randrange(n)
        y = adj[x][random.randrange(k)]
    else:
        u = random.randrange(n)
        v = adj[u][random.randrange(k)]
        x = random.randrange(n)
        y = adj[x][random.randrange(k)]
    
    if (u == x and v == y) or (u == y and v == x):
        return False
    if u == x or v == y:
        return False
    if x in adj[u] or y in adj[v]:
        return False
    
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    adj[u].append(x)
    adj[x].append(u)
    adj[v].append(y)
    adj[y].append(v)
    return True

def _find_short_cycles(adj, g):
    """Find all cycles shorter than g. Returns list of (cycle_vertices, length)."""
    from collections import deque
    n = len(adj)
    adj_sorted = [sorted(a) for a in adj]
    cycles = []
    
    for start in range(n):
        parent = {}
        dist = {}
        queue = deque()
        for nb in adj_sorted[start]:
            parent[(start, nb)] = None
            dist[(start, nb)] = 1
            queue.append((start, nb))
        
        while queue:
            prev, curr = queue.popleft()
            d = dist[(prev, curr)]
            if d >= g - 1:
                continue
            for nxt in adj_sorted[curr]:
                if nxt == prev:
                    continue
                if nxt == start:
                    if d + 1 < g:
                        cycle = [start, curr]
                        p = (prev, curr)
                        while p is not None:
                            p_prev, p_curr = p
                            if p_prev != start:
                                cycle.append(p_prev)
                            p = parent.get(p)
                        cycle.reverse()
                        key = tuple(sorted(cycle))
                        cycles.append((key, d + 1))
                    continue
                key = (curr, nxt)
                if key not in dist:
                    parent[key] = (prev, curr)
                    dist[key] = d + 1
                    queue.append(key)
    
    return cycles

def construct(k, g):
    deadline = time.time() + 118.0
    rng = 0x9E3779B97F4A7C15
    
    def time_left():
        return deadline - time.time()
    
    # Moore bound
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
    
    best_n = None
    best_adj = None
    
    # Try LUW with various parameters
    d_vals = [g - 5]
    if d_vals[0] % 2 == 0:
        d_vals[0] += 1
    d_vals = [d_vals[0], d_vals[0] + 2, d_vals[0] + 4]
    
    for d in d_vals:
        if time_left() < 10:
            break
        try:
            n_luw, edges_luw, adj_luw = _luw_graph(d, k)
            if n_luw >= moore and all(len(a) == k for a in adj_luw):
                if _girth_check_fast(adj_luw, g):
                    if best_n is None or n_luw < best_n:
                        best_n = n_luw
                        best_adj = [list(a) for a in adj_luw]
        except Exception:
            pass
    
    # Now try random search with swaps, starting from Moore bound
    n_start = max(2 * k, moore)
    if n_start % 2 != 0:
        n_start += 1
    
    max_n = max(n_start + 60, 500)
    n_values = list(range(n_start, max_n + 1, 2))
    
    for n in n_values:
        if time_left() < 8:
            break
        if best_n is not None and n >= best_n:
            break
        
        max_attempts = 15
        for attempt in range(max_attempts):
            if time_left() < 5:
                break
            
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            
            if _girth_check_fast(adj, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_adj = [list(a) for a in adj]
                break
            
            swaps_done = 0
            max_swaps = 300
            while time_left() > 3 and swaps_done < max_swaps:
                try:
                    cycles = _find_short_cycles(adj, g)
                except Exception:
                    cycles = []
                
                if not cycles:
                    if _girth_check_fast(adj, g):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_adj = [list(a) for a in adj]
                        break
                    break
                
                cycle, clen = cycles[rng % len(cycles)]
                edge_idx = rng % clen
                u = cycle[edge_idx]
                v = cycle[(edge_idx + 1) % clen]
                
                if u in adj[v]:
                    if _swap_regular(adj, n, k, rng, u, v):
                        swaps_done += 1
                        if swaps_done % 20 == 0:
                            if _girth_check_fast(adj, g):
                                if best_n is None or n < best_n:
                                    best_n = n
                                    best_adj = [list(a) for a in adj]
                                break
                else:
                    if _swap_regular(adj, n, k, rng):
                        swaps_done += 1
                        if swaps_done % 20 == 0:
                            if _girth_check_fast(adj, g):
                                if best_n is None or n < best_n:
                                    best_n = n
                                    best_adj = [list(a) for a in adj]
                                break
        
        # Continue to next n
    
    if best_adj is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best_adj[i] if i < j]
    
    # Fallback: LUW
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n_luw, edges_luw, _ = _luw_graph(d, k)
    return n_luw, edges_luw
