"""Cage graph construction for n(3,16) using improved algebraic methods."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _fast_cayley_girth(d, generators):
    """Fast girth check for Cayley graph on (Z_2)^d using BFS with early termination.
    Returns the girth (minimum cycle length) or infinity if no cycle found within depth g/2."""
    n = 1 << d
    half_g = 8  # For girth 16, we only need to check up to distance 7 from identity
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    min_cycle = float('inf')
    
    while q:
        u = q.popleft()
        du = dist[u]
        if du >= half_g:
            continue
        for gen in generators:
            v = u ^ gen
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
            elif v != 0 and dist[v] + du + 1 < min_cycle:
                # Found a cycle: path 0->...->u->v->...->0
                # Cycle length = dist[u] + dist[v] + 1, but need to be careful
                # Actually for Cayley graphs, if we reach v from u where v is already visited,
                # the cycle goes from 0 to v (dist[v] steps), then v to u (1 step via gen),
                # then u back to 0 (dist[u] steps) - but this isn't quite right.
                # Better: the cycle is 0 -> ... -> v -> u -> ... -> 0
                # Length = dist[v] + 1 + dist[u] only if the paths don't overlap improperly
                # For our purposes, if dist[u] + dist[v] + 1 < min_cycle, update
                min_cycle = dist[u] + dist[v] + 1
            elif v == 0 and du + 1 < min_cycle:
                min_cycle = du + 1
    
    return min_cycle if min_cycle < float('inf') else 999


def _smart_generator_search(d, k, g, deadline, num_trials=500):
    """Search for k generators in (Z_2)^d with girth >= g using smart selection.
    Uses linear independence and spread-out vectors."""
    n = 1 << d
    best_gens = None
    best_girth = 0
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        # Generate k random vectors that are linearly independent
        gens = []
        used_mask = 0
        for _ in range(k):
            for _attempt in range(100):
                x = random.randint(1, n - 1)
                # Check linear independence
                if x & ~used_mask:
                    gens.append(x)
                    used_mask |= x
                    break
            else:
                break
        
        if len(gens) < k:
            continue
        
        girth = _fast_cayley_girth(d, gens)
        if girth >= g:
            return gens
        if girth > best_girth:
            best_girth = girth
            best_gens = gens
    
    return best_gens


def _cayley_to_edges(d, generators):
    """Convert Cayley graph on (Z_2)^d with given generators to edge list."""
    n = 1 << d
    edges = []
    for i in range(n):
        for gen in generators:
            j = i ^ gen
            if i < j:
                edges.append((i, j))
    return edges


def construct(k, g):
    """Construct a k-regular graph of girth at least g with minimal vertices."""
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    if k != 3 or g != 16:
        return _lzw_fallback(k, g)
    
    random.seed(42)
    
    # Try d=9 (512 vertices) first - this is the Moore bound
    # Then d=10 (1024), d=11 (2048), etc.
    for d in [9, 10, 11, 12]:
        if time.time() > deadline:
            break
        n = 1 << d
        gens = _smart_generator_search(d, k, g, deadline, num_trials=200)
        if gens is not None:
            edges = _cayley_to_edges(d, gens)
            if _verify_girth(n, edges, g) and _check_regular(n, edges, k):
                return n, edges
    
    # Fallback: LZW construction
    return _lzw_fallback(k, g)


def _verify_girth(n, edges, g):
    """Verify no cycle shorter than g using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = {}
        q = deque()
        for v in adj[start]:
            dist[(start, v)] = 1
            q.append((start, v))
        while q:
            prev, curr = q.popleft()
            d = dist[(prev, curr)]
            if d >= g:
                continue
            if curr == start and prev != -1:
                return False
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                state = (curr, nxt)
                if state not in dist:
                    dist[state] = d + 1
                    q.append(state)
    return True


def _check_regular(n, edges, k):
    """Check graph is simple and k-regular."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    return True


def _lzw_fallback(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
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

# EVOLVE-BLOCK-END
