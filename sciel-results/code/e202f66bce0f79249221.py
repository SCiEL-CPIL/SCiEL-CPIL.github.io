import time
import numpy as np
from array import array
from collections import deque

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

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
    
    # Build adjacency as sorted lists for speed
    adj_sorted = [sorted(a) for a in adj]
    
    for start in range(n):
        # dist[prev][curr] = min distance of non-backtracking walk from start
        # Use dict for sparse representation
        dist = {}
        active = []
        for nb in adj_sorted[start]:
            dist[(start, nb)] = 1
            active.append((start, nb))
        
        for depth in range(1, g - 1):
            if not active:
                break
            next_active = []
            for (prev, curr) in active:
                for nxt in adj_sorted[curr]:
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

def _girth_check_np(adj, g):
    """NumPy-based girth check - faster for moderate graphs."""
    n = len(adj)
    if n == 0:
        return True
    
    adj_arr = np.array([sorted(a) for a in adj], dtype=np.int32)
    
    for start in range(n):
        # dist[prev, curr] = min depth
        dist = np.full((n, n), -1, dtype=np.int32)
        active = []
        for nb in adj_arr[start]:
            dist[start, nb] = 1
            active.append((start, int(nb)))
        
        for depth in range(1, g - 1):
            if not active:
                break
            next_active = []
            for (prev, curr) in active:
                for nxt in adj_arr[curr]:
                    nxt = int(nxt)
                    if nxt == prev:
                        continue
                    if nxt == start and depth + 1 < g:
                        return False
                    if dist[curr, nxt] == -1:
                        dist[curr, nxt] = depth + 1
                        next_active.append((curr, nxt))
            active = next_active
    
    return True

def _find_short_cycles(adj, g):
    """Find all cycles shorter than g. Returns list of (cycle_vertices, length)."""
    n = len(adj)
    adj_sorted = [sorted(a) for a in adj]
    cycles = []
    
    for start in range(n):
        # BFS from start, tracking parent
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
                        # Reconstruct cycle
                        cycle = [start, curr]
                        p = (prev, curr)
                        while p is not None:
                            p_prev, p_curr = p
                            if p_prev != start:
                                cycle.append(p_prev)
                            p = parent.get(p)
                        cycle.reverse()
                        # Deduplicate
                        key = tuple(sorted(cycle))
                        cycles.append((key, d + 1))
                    continue
                key = (curr, nxt)
                if key not in dist:
                    parent[key] = (prev, curr)
                    dist[key] = d + 1
                    queue.append(key)
    
    return cycles

def _swap_regular(adj, n, k, rng, u=None, v=None):
    """Perform a 2-switch. If u,v given, use edge (u,v); else random."""
    if u is not None:
        x = _randrange(rng, n)
        y = adj[x][_randrange(rng, k)]
    else:
        u = _randrange(rng, n)
        v = adj[u][_randrange(rng, k)]
        x = _randrange(rng, n)
        y = adj[x][_randrange(rng, k)]
    
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

def _pairing_regular(n, k, rng):
    """Generate random k-regular graph via perfect matching of stubs."""
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj

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
    
    # For k=5, g=9: moore = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # But that's the Moore bound (lower bound). We need n >= moore.
    # Actually for odd girth g=9, half=4, moore = 1 + 5*(1+4+16+64) = 426
    # But we know n(k,9) for k=5 is at least 428.
    
    # Try LUW construction first as a starting point
    # LUW gives girth >= d+5, so for g=9 we need d >= 4, but d must be odd
    # d = g-5 = 4, but must be odd, so d=5
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
    # Try sizes from n_start upward
    n_values = list(range(n_start, max_n + 1, 2))
    
    for n in n_values:
        if time_left() < 8:
            break
        if best_n is not None and n >= best_n:
            break
        
        # Try multiple random initializations
        max_attempts = 15
        for attempt in range(max_attempts):
            if time_left() < 5:
                break
            
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            
            # Quick girth check
            if _girth_check_fast(adj, g):
                if best_n is None or n < best_n:
                    best_n = n
                    best_adj = [list(a) for a in adj]
                break
            
            # Local search: targeted swaps to eliminate short cycles
            swaps_done = 0
            max_swaps = 300
            while time_left() > 3 and swaps_done < max_swaps:
                # Find short cycles to target
                try:
                    cycles = _find_short_cycles(adj, g)
                except Exception:
                    cycles = []
                
                if not cycles:
                    # No short cycles found - verify
                    if _girth_check_fast(adj, g):
                        if best_n is None or n < best_n:
                            best_n = n
                            best_adj = [list(a) for a in adj]
                        break
                    # Girth check says fail but no cycles found - trust girth check
                    break
                
                # Pick a random short cycle and try to break it
                cycle, clen = cycles[_randrange(rng, len(cycles))]
                # Pick a random edge in the cycle to swap
                edge_idx = _randrange(rng, clen)
                u = cycle[edge_idx]
                v = cycle[(edge_idx + 1) % clen]
                
                if u in adj[v]:
                    if _swap_regular(adj, n, k, rng, u, v):
                        swaps_done += 1
                        # Check girth periodically
                        if swaps_done % 20 == 0:
                            if _girth_check_fast(adj, g):
                                if best_n is None or n < best_n:
                                    best_n = n
                                    best_adj = [list(a) for a in adj]
                                break
                else:
                    # Edge not found (shouldn't happen), do random swap
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
