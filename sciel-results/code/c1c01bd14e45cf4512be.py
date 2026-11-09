"""Rewritten program for the suite task; EVOLVE-BLOCK contains the core logic."""
import time as _time
import random as _random
import sys

# EVOLVE-BLOCK-START

_DEADLINE = _time.time() + 108.0

def _now():
    return _time.time()

def _time_left():
    return _DEADLINE - _now()

def _girth_ok(adj, n, g):
    """Check that no cycle shorter than g exists using non-backtracking walks."""
    if g <= 3:
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u:
                        return False
        return True
    
    for s in range(n):
        frontier = [(-1, v) for v in adj[s]]
        depth = 1
        seen = set(frontier)
        while frontier and depth < g - 1:
            nxt = []
            for prev, cur in frontier:
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if w == s:
                        return False
                    st = (cur, w)
                    if st not in seen:
                        seen.add(st)
                        nxt.append(st)
            frontier = nxt
            depth += 1
    return True

def _build_adj(n, edges, k):
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    for v in range(n):
        if len(adj[v]) != k:
            return None
    return [sorted(a) for a in adj]

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction for k=4."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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

def _rand_reg(n, k, rng):
    """Generate a random k-regular graph using configuration model."""
    if k % 2 == 1:
        return None
    half = []
    for v in range(n):
        for _ in range(k):
            half.append(v)
    rng.shuffle(half)
    edges = set()
    ok = True
    for i in range(0, n * k, 2):
        a, b = half[i], half[i + 1]
        if a == b:
            ok = False
            break
        if (a, b) in edges or (b, a) in edges:
            ok = False
            break
        edges.add((a, b))
    if not ok:
        return None
    return sorted(edges)

def _short_cycles(adj, n, g):
    """Find short cycles (length < g) using non-backtracking walks."""
    cycles = []
    for s in range(n):
        frontier = [(-1, v, [s, v]) for v in adj[s]]
        depth = 1
        seen = set(frontier[:2])
        while frontier and depth < g - 1:
            nxt = []
            for prev, cur, path in frontier:
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if w == s:
                        cycles.append(path)
                        continue
                    st = (cur, w)
                    if st not in seen:
                        seen.add(st)
                        nxt.append((cur, w, path + [w]))
            frontier = nxt
            depth += 1
            if len(cycles) > 100:
                return cycles
    return cycles

def _repair(adj, n, k, g, rng, max_iters=5000):
    """Try to eliminate short cycles via 2-switches."""
    adj = [set(a) for a in adj]
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        cycles = _short_cycles(adj, n, g)
        if not cycles:
            return adj
        cyc = min(cycles, key=len)
        L = len(cyc)
        i = rng.randrange(L)
        a, b = cyc[i], cyc[(i + 1) % L]
        
        cand_a = list(adj[a])
        cand_b = list(adj[b])
        done = False
        for _ in range(100):
            c = rng.choice(cand_a)
            d = rng.choice(cand_b)
            if c == d:
                continue
            if d in adj[a]:
                continue
            if c in adj[b]:
                continue
            adj[a].discard(b); adj[b].discard(a)
            adj[a].discard(c); adj[c].discard(a)
            adj[b].discard(d); adj[d].discard(b)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(d); adj[d].add(b)
            done = True
            break
        if not done:
            adj[a].add(b); adj[b].add(a)
    return adj

def _random_search(k, g, target_n, rng, time_budget):
    """Random search for smaller graphs."""
    candidates = []
    for n in range(target_n - 1, max(480, target_n - 50), -1):
        if n % 2 == 0:
            candidates.append(n)
    if not candidates:
        candidates = [target_n - 2]
    rng.shuffle(candidates)
    for n in candidates:
        if _time_left() < 10.0:
            break
        for attempt in range(20):
            if _time_left() < 10.0:
                break
            base = _rand_reg(n, k, rng)
            if base is None:
                continue
            adj = [set() for _ in range(n)]
            for i, j in base:
                adj[i].add(j)
                adj[j].add(i)
            adj = _repair(adj, n, k, g, rng, max_iters=2000)
            if all(len(a) == k for a in adj) and _girth_ok(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    return None

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    global _DEADLINE
    _DEADLINE = _time.time() + 108.0

    try:
        n_seed, edges_seed = _luw(k, g)
    except Exception:
        n_seed, edges_seed = None, None
    
    seed_valid = False
    if n_seed is not None:
        adj = _build_adj(n_seed, edges_seed, k)
        seed_valid = (adj is not None) and _girth_ok(adj, n_seed, g)
        
    best_n, best_edges = n_seed, edges_seed
    if not seed_valid:
        pass

    rng = _random.Random(12345)
    found = None
    if seed_valid and n_seed is not None:
        try:
            found = _random_search(k, g, n_seed, rng, time_budget=_time_left() - 5.0)
        except Exception:
            found = None
            
    if found is not None:
        n_f, e_f = found
        if n_f < (best_n if best_n is not None else float('inf')):
            best_n, best_edges = n_f, e_f

    if best_edges is None:
        return n_seed, edges_seed
    return best_n, best_edges

# EVOLVE-BLOCK-END
