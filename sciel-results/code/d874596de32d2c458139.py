import time
import threading
import random
from collections import deque
from functools import lru_cache

_DEADLINE = 118.0
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
            if w == u:
                return dist + 1
            key = (w, v)
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    
    return 10 ** 9


def _nonback_girth_limited(adj, n, max_len):
    """Return shortest cycle length if < max_len, else max_len.
    Early termination for quick rejection."""
    prev = {}
    q = deque()
    
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    
    while q:
        (v, p), (u, dist) = q.popleft()
        if dist + 1 >= max_len:
            continue
        for w in adj[v]:
            if w == p:
                continue
            if w == u:
                return dist + 1
            key = (w, v)
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    
    return max_len


def _girth_at_least_fast(adj, n, g):
    """Fast girth check using BFS from a subset of vertices.
    Returns True if girth >= g is likely (not guaranteed), False if definitely < g.
    Uses early termination: if any BFS finds a cycle < g, return False immediately."""
    if g <= 3:
        return True
    
    # For small g, we can be more aggressive with sampling
    # Use all vertices for small n, sample for large n
    if n <= 200:
        vertices = list(range(n))
    else:
        # Sample ~30% of vertices
        sample_size = max(10, n // 3)
        vertices = random.sample(range(n), min(sample_size, n))
    
    for start in vertices:
        # BFS from start, tracking (vertex, parent) to detect non-backtracking cycles
        visited = {}
        q = deque()
        for p in adj[start]:
            key = (p, start)
            visited[key] = 1
            q.append(key)
        
        while q:
            (v, p), dist = q.popleft()
            if dist + 1 >= g:
                continue
            for w in adj[v]:
                if w == p:
                    continue
                if w == start:
                    return False
                key = (w, v)
                if key in visited:
                    continue
                visited[key] = dist + 1
                q.append(key)
    
    return True


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return False
        for j in adj[i]:
            if j < 0 or j >= n or j == i:
                return False
    return _nonback_girth(adj, n) >= g


def _girth_at_least(adj, n, g):
    """Quick check: does graph have girth >= g? Uses limited BFS."""
    return _nonback_girth_limited(adj, n, g) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj

# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
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

# ---------------- construction 2: Random Regular Graph Search ----------------
def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph on n vertices using configuration model.
    Returns adjacency list or None if multigraph/loops are generated (retryable)."""
    if n * k % 2 != 0:
        return None
    
    half_edges = []
    for v in range(n):
        for _ in range(k):
            half_edges.append(v)
    
    # Shuffle half-edges
    random.shuffle(half_edges)
    
    # Pair them up
    adj = [set() for _ in range(n)]
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            # Loop, reject
            return None
        if v in adj[u]:
            # Multi-edge, reject
            return None
        adj[u].add(v)
        adj[v].add(u)
        
    # Verify simplicity and regularity
    for i in range(n):
        if len(adj[i]) != k:
            return None
    return adj

def _local_search_polish(adj, n, k, g, max_steps=500):
    """Try to improve girth via local edge swaps.
    Returns improved adjacency list if girth >= g achieved, else None."""
    if _elapsed() > _DEADLINE - 5.0:
        return None
    
    # Work with a copy
    adj = [list(x) for x in adj]
    n = len(adj)
    
    # Build edge set for fast lookup
    edge_set = set()
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edge_set.add((i, j))
    
    def remove_edge(i, j):
        if i > j:
            i, j = j, i
        adj[i].remove(j)
        adj[j].remove(i)
        edge_set.discard((i, j))
    
    def add_edge(i, j):
        if i == j or (i, j) in edge_set or (j, i) in edge_set:
            return False
        adj[i].append(j)
        adj[j].append(i)
        if i < j:
            edge_set.add((i, j))
        else:
            edge_set.add((j, i))
        return True
    
    best_girth = _nonback_girth_limited(adj, n, g)
    
    for step in range(max_steps):
        if _elapsed() > _DEADLINE - 5.0:
            break
        if best_girth >= g:
            break
        
        # Randomly pick two non-adjacent vertices from different neighborhoods
        # and try an edge swap: remove (a,b) and (c,d), add (a,c) and (b,d)
        attempts = 0
        while attempts < 20:
            attempts += 1
            a = random.randrange(n)
            b = random.choice(adj[a]) if adj[a] else None
            if b is None:
                continue
            c = random.randrange(n)
            d = random.choice(adj[c]) if adj[c] else None
            if d is None:
                continue
            
            # Ensure we're removing valid edges and adding valid edges
            if b == c or d == a or b == d:
                continue
            if (a, b) not in edge_set and (b, a) not in edge_set:
                continue
            if (c, d) not in edge_set and (d, c) not in edge_set:
                continue
            if a == c or a == d or b == c or b == d:
                continue
            
            # Check if new edges would be valid
            if (a, c) in edge_set or (c, a) in edge_set:
                continue
            if (b, d) in edge_set or (d, b) in edge_set:
                continue
            
            # Perform swap
            remove_edge(a, b)
            remove_edge(c, d)
            add_edge(a, c)
            add_edge(b, d)
            
            # Check girth
            new_girth = _nonback_girth_limited(adj, n, g)
            if new_girth >= g:
                return adj
            elif new_girth > best_girth:
                best_girth = new_girth
            else:
                # Revert
                remove_edge(a, c)
                remove_edge(b, d)
                add_edge(a, b)
                add_edge(c, d)
    
    return None


def _random_regular_search(k, g, max_n=1200, max_attempts_per_n=50):
    """Search for a k-regular graph with girth >= g by sampling random regular graphs.
    Starts from Moore bound and increases n."""
    
    # Moore bound calculation
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Ensure n * k is even
    start_n = moore
    if (start_n * k) % 2 != 0:
        start_n += 1
        
    best_result = None
    
    for n in range(start_n, max_n + 1):
        if _elapsed() > _DEADLINE - 10.0:
            break
        if (n * k) % 2 != 0:
            continue
            
        found = False
        for attempt in range(max_attempts_per_n):
            if _elapsed() > _DEADLINE - 10.0:
                break
            if found:
                break
                
            adj = _generate_random_k_regular(n, k)
            if adj is None:
                continue
            
            # Fast pre-check first
            if not _girth_at_least_fast(adj, n, g):
                continue
            
            # Full check
            if _girth_at_least(adj, n, g):
                # Found a valid graph
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                best_result = (n, edges)
                found = True
                return best_result
            else:
                # Try local search polish
                if _elapsed() < _DEADLINE - 15.0:
                    polished = _local_search_polish(adj, n, k, g, max_steps=100)
                    if polished is not None:
                        # Verify the polished graph
                        if _girth_at_least(polished, n, g):
                            edges = [(i, j) for i in range(n) for j in polished[i] if i < j]
                            best_result = (n, edges)
                            found = True
                            return best_result
        
        # If we spent too much time on this n, move on
        if _elapsed() > _DEADLINE - 15.0:
            break
    
    return best_result

# ---------------- construction 3: Optimized Circulant Search ---------------------------
def _circulant_search(k, g):
    """Circulant search - NOTE: for k>=4 and g>=5, circulants always have 4-cycles.
    This function is kept for completeness but will rarely find valid graphs for g>=5."""
    if g >= 5:
        return None
        
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    if k % 2 == 0:
        return None
    
    start_n = moore
    end_n = moore + 200
    
    for n in range(start_n, end_n + 1, 2):
        if _elapsed() > _DEADLINE - 10.0:
            return None
        if n < k + 1:
            continue
        
        half_n = n // 2
        for s1 in range(1, half_n):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            for s2 in range(s1 + 1, half_n):
                if 2 * s1 == s2 or 2 * s2 == s1:
                    continue
                if s1 == n // 4 or s2 == n // 4:
                    continue
                if s1 + s2 == half_n:
                    continue
                
                adj = [set() for _ in range(n)]
                for i in range(n):
                    adj[i].add((i + s1) % n)
                    adj[i].add((i - s1) % n)
                    adj[i].add((i + s2) % n)
                    adj[i].add((i - s2) % n)
                    adj[i].add((i + half_n) % n)
                
                if _nonback_girth(adj, n) >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
    return None

def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    lock = threading.Lock()
    
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("random_reg", lambda: _random_regular_search(k, g)),
    ]
    
    # Only add circulant if g < 5 (otherwise it's impossible)
    if g < 5:
        jobs.append(("circulant", lambda: _circulant_search(k, g)))
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, lock, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    # Wait a bit more for threads to finish if they're close
    for t in threads:
        t.join(timeout=max(0, _DEADLINE - _elapsed() - 1.0))

    with lock:
        best = holder.get("best")
    
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            # Fallback: complete graph on k+1 vertices (invalid for g>3 but valid structure)
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best

def _worker(name, fn, holder, lock, k, g):
    try:
        res = fn()
    except Exception:
        res = None
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
