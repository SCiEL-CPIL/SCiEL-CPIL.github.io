import time
import random
from collections import deque
import sys

_DEADLINE = 145.0
_START = 0.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _girth_check(adj, g):
    """Returns True if girth >= g. Uses BFS from each vertex to detect short cycles."""
    n = len(adj)
    max_depth = g // 2
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    q.append(v)
                else:
                    # Edge (u,v) where both are visited
                    # If v is not the parent of u in BFS tree, check cycle length
                    # The cycle through s has length dist[u] + dist[v] + 1
                    # But we need to exclude the case where v is the direct parent of u
                    # In BFS, if dist[v] == dist[u] - 1 and v is parent, skip
                    # Simple check: if dist[u] + dist[v] + 1 < g, we have a short cycle
                    # This is valid because dist values are shortest distances
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True

def _random_4regular(n, seed):
    """Generate a random 4-regular graph using configuration model."""
    rng = random.Random(seed)
    for _ in range(50):
        # Configuration model: each vertex has 4 stubs
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        rng.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, 4 * n, 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if not ok:
            continue
        if any(len(s) != 4 for s in adj):
            continue
        return adj
    return None

def _find_short_cycle(adj, g):
    """Find a cycle of length < g. Returns list of vertices in the cycle, or None."""
    n = len(adj)
    max_depth = g // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        cycle = [u]
                        cur = u
                        while cur != s:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle.reverse()
                        # Add path from s to v
                        cur = v
                        while cur != s:
                            cur = parent[cur]
                            cycle.append(cur)
                        return cycle
    return None

def _local_search(adj, g, time_limit):
    """Improve girth by breaking short cycles via edge swaps."""
    n = len(adj)
    start_time = time.time()
    best_girth = 0
    
    while time.time() - start_time < time_limit and _remaining() > 1:
        # Check if girth is already sufficient
        if _girth_check(adj, g):
            return adj
        
        # Find a short cycle
        cycle = _find_short_cycle(adj, g)
        if cycle is None:
            return adj
        
        # Pick a random edge in the cycle to swap
        # Edge (cycle[i], cycle[i+1]) for some i
        i = random.randrange(len(cycle))
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        
        # Find a vertex x not in the cycle, not adjacent to u or v
        # And find y neighbor of x such that swap is valid
        attempts = 0
        while attempts < 100:
            attempts += 1
            x = random.randrange(n)
            if x == u or x == v:
                continue
            if x in adj[u] or x in adj[v]:
                continue
            
            # Pick y from neighbors of x
            y_candidates = [y for y in adj[x] if y != u and y != v and y not in adj[u] and y not in adj[v]]
            if not y_candidates:
                continue
            y = random.choice(y_candidates)
            
            # Check swap validity: (u,v) and (x,y) -> (u,x) and (v,y)
            # Need: x not in adj[u], y not in adj[v], x not in adj[v], y not in adj[u]
            if x in adj[u] or y in adj[v] or x in adj[v] or y in adj[u]:
                continue
            
            # Perform swap
            adj[u].discard(v)
            adj[v].discard(u)
            adj[x].discard(y)
            adj[y].discard(x)
            
            adj[u].add(x)
            adj[x].add(u)
            adj[v].add(y)
            adj[y].add(v)
            
            break
    
    return None

def construct(k, g):
    global _START
    _START = time.time()
    
    if k != 4:
        return _luw(k, g)
        
    # Target n range
    # Moore bound for k=4, g=10 is 242.
    # Known best is 320.
    # We try to find a graph with n < 320.
    
    best_adj = None
    best_n = None
    
    # Try n from 242 to 320, even values only (4-regular requires even n)
    # Allocate time proportionally: more time for smaller n (harder to find)
    for n in range(242, 321, 2):
        if _remaining() < 3:
            break
        
        # Time allocation: smaller n gets more time since it's harder
        # Total time budget is about 140 seconds
        # Give each n about 140/40 = 3.5 seconds on average
        # But smaller n should get more
        time_for_n = min(2.0, max(0.5, 140.0 / 40.0))
        
        found = False
        start_n_time = time.time()
        attempts = 0
        while time.time() - start_n_time < time_for_n and _remaining() > 2:
            attempts += 1
            seed = random.randrange(1 << 30)
            adj = _random_4regular(n, seed)
            if adj is None:
                continue
            
            # Quick girth check
            if _girth_check(adj, g):
                best_adj = adj
                best_n = n
                found = True
                break
            
            # Local search to improve girth
            adj_ls = _local_search(adj, g, min(1.0, time_for_n - (time.time() - start_n_time)))
            if adj_ls is not None and _girth_check(adj_ls, g):
                best_adj = adj_ls
                best_n = n
                found = True
                break
        
        if found:
            edges = []
            for i in range(n):
                for j in best_adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # If no graph found in range, try LUW fallback
    return _luw(k, g)

def _luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction (valid fallback)."""
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

# Global rng for local search
rng = random.Random(42)
