"""Construct a 3-regular graph of girth at least g via random generation + edge-swap local search."""
import time
import random
import numpy as np

DEADLINE = 145  # seconds; leave a safety margin before the 150 s kill

def _moore_bound(k, g):
    """Lower bound on n for a k-regular graph of girth g (Moore bound)."""
    if k == 2:
        return g if g % 2 == 0 else g + 1
    if g % 2 == 0:
        # even girth: tree-like up to depth g/2
        # n >= k * sum_{i=0}^{g/2-1} (k-1)^i  ... actually the Moore bound for even girth:
        # n >= k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^(g/2 - 1))
        half = g // 2
        s = 1
        for i in range(1, half):
            s *= (k - 1)
            # accumulate
        # Recompute properly
        s = 0
        term = 1
        for i in range(half):
            s += term
            term *= (k - 1)
        return k * s
    else:
        # odd girth
        half = (g - 1) // 2
        s = 1
        for i in range(1, half + 1):
            s *= (k - 1)
        # n >= 2 * sum_{i=0}^{half} (k-1)^i - ... 
        # Standard Moore bound for odd girth g: n >= 2 * (1 + (k-1) + ... + (k-1)^{(g-1)/2}) - ... 
        # Actually: n >= k * (1 + (k-1) + ... + (k-1)^{(g-3)/2}) + (k-1)^{(g-1)/2} * (k-1)
        # Let me use the standard formula:
        # For odd girth g = 2r+1: n >= 2 * sum_{i=0}^{r} (k-1)^i - ... no
        # Standard: n >= k * sum_{i=0}^{r-1} (k-1)^i + (k-1)^r * (k-1)  where r = (g-1)/2
        r = (g - 1) // 2
        s = 0
        term = 1
        for i in range(r):
            s += term
            term *= (k - 1)
        return k * s + term * (k - 1)


def _generate_random_3_regular(n, rng):
    """Generate a random 3-regular graph on n vertices using configuration model.
    Returns adjacency list or None if failed."""
    if n % 2 != 0:
        n += 1
    if n < 4:
        return None
    # Each vertex has 3 stubs
    stubs = list(range(n)) * 3
    rng.shuffle(stubs)
    # Pair up stubs
    edges = set()
    adj = [set() for _ in range(n)]
    for i in range(0, 2 * 3 * n // 2, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            continue  # self-loop, skip (will need to retry)
        if v in adj[u]:
            continue  # parallel edge, skip
        adj[u].add(v)
        adj[v].add(u)
        edges.add((min(u, v), max(u, v)))
    # Check all vertices have degree 3
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return adj


def _find_shortest_cycle_bfs(adj, n, girth):
    """Find a cycle of length < girth using BFS from each vertex.
    Returns (cycle_length, cycle_vertices) or (inf, None) if girth >= girth."""
    # Use BFS to find shortest cycle. For each vertex, BFS and track parent.
    # A cycle is found when we hit a visited vertex that isn't the parent.
    INF = float('inf')
    best_len = INF
    best_cycle = None
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if dist[u] >= (girth - 1) // 2 + 1:
                break
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] < dist[u]:
                    # Found a cycle: u - v, and v was visited earlier
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best_len and cycle_len < girth:
                        best_len = cycle_len
                        # Reconstruct cycle
                        cycle = []
                        # Path from v to start
                        path_v = []
                        cur = v
                        while cur != -1:
                            path_v.append(cur)
                            cur = parent[cur]
                        path_v.reverse()
                        # Path from u to start
                        path_u = []
                        cur = u
                        while cur != -1:
                            path_u.append(cur)
                            cur = parent[cur]
                        path_u.reverse()
                        # Find LCA
                        # Simpler: just return the length for now
                        best_cycle = list(range(cycle_len))  # placeholder
                        break
            if best_len < girth:
                break
        if best_len < girth:
            break
    return best_len, best_cycle


def _count_short_cycles(adj, n, girth):
    """Count cycles of length < girth using non-backtracking walks."""
    count = 0
    for start in range(n):
        # Non-backtracking walks of length up to girth-1
        # State: (current_vertex, previous_vertex)
        states = [(start, -1)]
        for step in range(1, girth):
            new_states = []
            for (u, prev) in states:
                for v in adj[u]:
                    if v != prev:
                        new_states.append((v, u))
                        if v == start and step >= 3:
                            count += 1
            states = new_states
    return count // 2  # each cycle counted twice (two directions)


def _edge_swap(adj, n, rng):
    """Perform a random edge swap: remove (u,v) and (w,x), add (u,w) and (v,x) or (u,x) and (v,w).
    Returns True if a valid swap was made."""
    # Collect all edges
    edge_list = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edge_list.append((u, v))
    if len(edge_list) < 2:
        return False
    # Pick two edges that don't share vertices
    for _ in range(50):
        e1 = rng.choice(edge_list)
        e2 = rng.choice(edge_list)
        if e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]:
            continue
        u, v = e1
        w, x = e2
        # Try swap (u,w) and (v,x)
        if w not in adj[u] and x not in adj[v]:
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
            return True
        # Try swap (u,x) and (v,w)
        if x not in adj[u] and w not in adj[v]:
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(w)
            adj[w].add(v)
            return True
    return False


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    start_time = time.time()
    
    if k != 3:
        # Fallback to the algebraic construction for other k values
        return _algebraic_construct(k, g)
    
    # For k=3, use random generation + local search
    # Moore bound for 3-regular, girth 16
    mb = _moore_bound(3, g)
    
    # Try n values from Moore bound upward
    # n must be even for 3-regular
    best_n = None
    best_adj = None
    best_edges = None
    
    # Start from a reasonable n
    # The known best is 936, so try around there
    # But we want to minimize, so try smaller first
    
    # Try a range of n values
    n_candidates = []
    for n in range(mb, 1000, 2):  # even n only
        if time.time() - start_time > DEADLINE:
            break
        n_candidates.append(n)
    
    # Also add some specific candidates
    for n in [512, 514, 516, 600, 700, 800, 900, 936, 938, 940]:
        if n >= mb and n % 2 == 0 and n not in n_candidates:
            n_candidates.append(n)
    
    n_candidates = sorted(set(n_candidates))
    
    rng = random.Random(42)
    
    for n in n_candidates:
        if time.time() - start_time > DEADLINE:
            break
        
        # Try to generate a random 3-regular graph
        adj = None
        for attempt in range(200):
            if time.time() - start_time > DEADLINE:
                break
            adj = _generate_random_3_regular(n, rng)
            if adj is not None:
                break
        
        if adj is None:
            continue
        
        # Check girth
        cycle_len, _ = _find_shortest_cycle_bfs(adj, n, g)
        if cycle_len >= g:
            # Found a valid graph!
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        
        # Otherwise, try to improve via edge swaps
        # Simulated annealing style: accept swaps that don't create shorter cycles
        current_cycle_len = cycle_len
        for step in range(5000):
            if time.time() - start_time > DEADLINE:
                break
            
            old_adj = [set(s) for s in adj]
            if not _edge_swap(adj, n, rng):
                continue
            
            # Check if the swap improved things
            new_cycle_len, _ = _find_shortest_cycle_bfs(adj, n, g)
            
            if new_cycle_len > current_cycle_len or (new_cycle_len == current_cycle_len and rng.random() < 0.1):
                current_cycle_len = new_cycle_len
                if current_cycle_len >= g:
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
            else:
                # Revert
                adj = old_adj
    
    # If we didn't find a valid graph, fall back to algebraic construction
    return _algebraic_construct(k, g)


def _algebraic_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar style construction."""
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
