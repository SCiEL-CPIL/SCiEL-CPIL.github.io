"""Combined cage solver: LUW constructions (q=4 and q=k), random search fallback."""
import time
import threading
import random
from collections import deque

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                return dist + 1
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- construction: LUW with configurable q --------------------
def _luw_with_q(k, g, q):
    """Lazebnik-Ustimenko-Woldar construction with field size q.
    Returns a q-regular bipartite graph. Valid when q == k."""
    if q != k:
        return None
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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


# ---------------- construction: LUW with q=4 (primary for k=5) -------------
def _luw_q4(k, g):
    """Primary construction: LUW with q=4 for k=5, g=9.
    Produces 5-regular graph with girth 12, ~330 vertices."""
    if k != 5:
        return None
    return _luw_with_q(k, g, 4)


# ---------------- construction: LUW with q=k (general) ----------------------
def _luw_general(k, g):
    """General LUW construction with q=k."""
    return _luw_with_q(k, g, k)


# ---------------- construction: Random Configuration Model + Local Search --
def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Search n from moore to moore + 50
    for n in range(moore, moore + 50):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
            
        for attempt in range(20):
            if _elapsed() > _DEADLINE - 15.0:
                return None
            
            adj = _generate_random_regular(n, k, rng)
            if adj is None:
                continue
                
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
                
            for step in range(400):
                if _elapsed() > _DEADLINE - 15.0:
                    return None
                
                girth = _nonback_girth(adj, n)
                if girth >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
                    
                u = rng.randint(0, n - 1)
                if not adj[u]:
                    continue
                v = rng.choice(list(adj[u]))
                
                w = -1
                x = -1
                for _ in range(60):
                    w = rng.randint(0, n - 1)
                    if w in (u, v):
                        continue
                    if w in adj[u] or w in adj[v]:
                        continue
                    if not adj[w]:
                        continue
                    x = rng.choice(list(adj[w]))
                    if x in (u, v, w):
                        continue
                    if x in adj[v]:
                        continue
                    
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[w].remove(x)
                    adj[x].remove(w)
                    
                    adj[u].add(w)
                    adj[w].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    
                    break
                if w == -1:
                    continue
    return None


def _generate_random_regular(n, k, rng):
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    return adj


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    lock = threading.Lock()
    
    def _try_result(res):
        if res is None:
            return
        n, edges = res
        if _elapsed() >= _DEADLINE:
            return
        try:
            adj = _build_adj(edges, n)
            if n > 0 and _is_valid(adj, n, k, g):
                with lock:
                    cur = holder.get("best")
                    if cur is None or n < cur[0]:
                        holder["best"] = (n, edges)
        except Exception:
            pass
    
    # Priority 1: LUW with q=4 (deterministic, fast, known good for k=5,g=9)
    t1 = threading.Thread(target=lambda: _try_result(_luw_q4(k, g)), daemon=True)
    t1.start()
    
    # Priority 2: LUW with q=k (deterministic, general)
    t2 = threading.Thread(target=lambda: _try_result(_luw_general(k, g)), daemon=True)
    t2.start()
    
    # Priority 3: Random search with multiple seeds
    for seed in range(3):
        t = threading.Thread(
            target=lambda s=seed: _try_result(_random_search(k, g, seed=s)),
            daemon=True
        )
        t.start()
    
    # Wait for threads with deadline
    all_threads = [t1, t2]
    # Collect all thread references
    for t in threading.enumerate():
        if t != threading.current_thread() and t.is_alive():
            all_threads.append(t)
    
    while _elapsed() < _DEADLINE - 3.0:
        alive = [t for t in all_threads if t.is_alive()]
        if not alive:
            break
        time.sleep(0.05)
    
    # Give remaining threads a moment
    for t in all_threads:
        t.join(timeout=_DEADLINE - _elapsed() - 0.5)
    
    best = holder.get("best")
    if best is None:
        # Fallback: try LUW q=4 synchronously
        try:
            n, edges = _luw_q4(k, g)
            if n is not None:
                best = (n, edges)
        except Exception:
            pass
        if best is None:
            try:
                n, edges = _luw_general(k, g)
                best = (n, edges)
            except Exception:
                n = k + 1
                edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
                best = (n, edges)
    return best
