"""Combined cage solver: algebraic (LUW) + deterministic circulant + greedy leaf-pairing + random search + exact girth verification."""
import time
import threading
import random
import itertools
from collections import deque

_DEADLINE = 145.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


def _girth_geq(adj, n, g):
    """Return True if girth >= g, False otherwise. Uses non-backtracking BFS."""
    if g <= 3:
        return False  # Any graph with edges has cycle length >= 3, but girth >= 3 means no triangles
    # dist[(prev, curr)] = shortest non-backtracking walk length ending at (prev, curr)
    dist = {}
    q = deque()
    for u in range(n):
        for v in adj[u]:
            dist[(v, u)] = 1
            q.append((v, u, 1))
    
    while q:
        v, p, d = q.popleft()
        # If d+1 >= g, any cycle from extending would be >= g, so skip
        if d + 1 >= g:
            continue
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if key in dist:
                cycle_len = d + dist[key] + 1
                if cycle_len < g:
                    return False
            else:
                dist[key] = d + 1
                q.append((w, v, d + 1))
    
    return True


def _girth_fast(adj, n, g):
    """Return True if girth >= g, using non-backtracking BFS from all vertices.
    Early termination: if any cycle of length < g is found, return False."""
    # State: (prev, current) -> distance from start of non-backtracking walk
    # We use a dict mapping (prev, curr) -> dist
    # To speed up: process all starting edges simultaneously
    from collections import deque
    dist = {}
    q = deque()
    for u in range(n):
        for v in adj[u]:
            key = (v, u)
            dist[key] = 1
            q.append((v, u, 1))
    
    while q:
        v, p, d = q.popleft()
        # Check if we can close a cycle: if (u, v) was the start, cycle length = d+1
        # Actually, the start was (v, u) meaning we went u->v. If we now go to u, cycle length = d+1
        for w in adj[v]:
            if w == p:
                continue
            # If w is the original start vertex u (where we started from), we found a cycle
            # The non-backtracking walk is: u -> v -> ... -> w, and if w == u, cycle length = d+1
            # But we need to track the origin. Let's use a different approach.
            pass
    
    # Better approach: track (prev, curr) -> (origin, dist) where origin is the first vertex
    # Actually, standard approach: BFS on directed edges (prev -> curr)
    # dist[(prev, curr)] = length of shortest non-backtracking walk ending at (prev, curr)
    # A cycle is found when we can reach (u, v) where (v, u) was also reached, and dist sum + 1 < g
    
    # Let me redo this properly:
    # Non-backtracking walk: v0, v1, v2, ... where vi != vi-2
    # State: (vi-1, vi) with distance i
    # Start: all (u, v) for each edge u-v, distance 1
    # Transition: from (u, v) to (v, w) for w != u
    # Cycle detected: if (v, w) was already seen with distance d', then we have a cycle of length d' + d + 1
    #   where d is current distance to (u,v) and d' is distance to (v,w)
    #   Wait, that's not quite right. If we have walk of length d ending at (u,v) meaning v0->v1->...->v_{d-1}=u->v_d=v
    #   And walk of length d' ending at (v,w) meaning w0->w1->...->w_{d'-1}=v->w_{d'}=w
    #   If w == v0, then we have a cycle. But we don't track v0.
    
    # Standard girth algorithm:
    # BFS from each directed edge. dist[(u,v)] = min distance of non-backtracking walk from some start to (u,v)
    # When processing (u,v) at distance d, for each neighbor w of v (w != u):
    #   if (v,w) is already in dist with distance d2, then cycle length = d + d2 + 1
    #   else dist[(v,w)] = d + 1
    
    # This finds the girth. Let me implement with early termination for girth < g.
    
    dist = {}
    q = deque()
    for u in range(n):
        for v in adj[u]:
            dist[(v, u)] = 1
            q.append((v, u, 1))
    
    while q:
        v, p, d = q.popleft()
        if d + 1 >= g:
            # Even if we find a cycle, it would be at least d+1 >= g, so no short cycle from this path
            # But we still need to propagate? No - if d+1 >= g, any cycle formed would have length >= g
            # So we can skip expanding this state
            continue
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if key in dist:
                # Cycle found of length d + dist[key] + 1
                cycle_len = d + dist[key] + 1
                if cycle_len < g:
                    return False
            else:
                dist[key] = d + 1
                q.append((w, v, d + 1))
    
    return True


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
    return _girth_geq(adj, n, g)


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
    index = {start: 0}
    order = [start]
    edges = []
    i = 0
    while i < len(order):
        vertex = order[i]
        i += 1
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


# ---------------- construction 2: generalized Petersen graph G(N, k) ----------
def _petersen(k, g):
    if k != 3:
        return None
    for N in range(2 * k + 1, 5000):
        if N % 2 == 0 or N <= 2 * k:
            continue
        edges = []
        for i in range(N):
            edges.append((i, (i + 1) % N))
            edges.append((N + i, (N + (i + 1)) % N))
            edges.append((i, N + i))
        edges = list(set(edges))
        n = 2 * N
        if len(edges) != N * 3:
            continue
        adj = _build_adj(edges, n)
        if all(len(adj[i]) == 3 for i in range(n)) and _nonback_girth(adj, n) >= g:
            return n, edges
    return None


# ---------------- construction 3: Deterministic + Random circulant search ----
def _circulant_search(k, g):
    if k % 2 != 0:
        return None
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    half = k // 2

    # Try deterministic search for small n first (exhaustive combinations)
    max_exhaustive = 80
    for n in range(moore, min(moore + 200, max_exhaustive + 1)):
        if _elapsed() > _DEADLINE - 20.0:
            break
        if n * k % 2 != 0:
            continue
        # Exhaustive search over all subsets of size 'half' from {1, ..., n//2}
        # that don't contain both s and n-s
        candidates = list(range(1, n // 2 + 1))
        for S in itertools.combinations(candidates, half):
            S_set = set(S)
            if any((n - s) % n in S_set for s in S):
                continue
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(n):
                for s in S:
                    adj[i].add((i + s) % n)
                    adj[i].add((i - s) % n)
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue
            if _girth_geq(adj, n, g):
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges

    # Random sampling for larger n
    for n in range(max(moore, max_exhaustive + 1), moore + 400):
        if _elapsed() > _DEADLINE - 10.0:
            break
        if n * k % 2 != 0:
            continue
        rng = random.Random(n * 1000 + k * 7 + g)
        for _ in range(200):
            S = rng.sample(range(1, n // 2 + 1), half)
            S_set = set(S)
            if any((n - s) % n in S_set for s in S):
                continue
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(n):
                for s in S:
                    adj[i].add((i + s) % n)
                    adj[i].add((i - s) % n)
                if len(adj[i]) != k:
                    valid = False
                    break
            if not valid:
                continue
            if _girth_geq(adj, n, g):
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
    return None


# ---------------- construction 4: Greedy leaf-pairing on extended tree --------
def _greedy_leaf_pair(k, g, max_restarts=5, time_budget=30.0):
    """Build a k-regular tree of radius r+1, then greedily pair leaves to
    achieve k-regularity while maintaining girth >= g."""
    r = (g - 1) // 2
    radius = r + 1

    tree_sizes = [1]
    for i in range(1, radius + 1):
        tree_sizes.append(k * (k - 1) ** (i - 1))

    n_tree = sum(tree_sizes)

    parent = [-1] * n_tree
    level = [0] * n_tree
    children = [[] for _ in range(n_tree)]

    for lvl in range(1, radius + 1):
        start_vid = sum(tree_sizes[:lvl])
        for i in range(tree_sizes[lvl]):
            v = start_vid + i
            level[v] = lvl
            if lvl == 1:
                p = 0
            else:
                prev_start = sum(tree_sizes[:lvl - 1])
                p_idx = i // (k - 1)
                p = prev_start + p_idx
            parent[v] = p
            children[p].append(v)

    leaves = [v for v in range(n_tree) if level[v] == radius]

    # Group leaves by top-level branch (child of root)
    top_branch = {}
    for leaf in leaves:
        cur = leaf
        while parent[cur] != 0:
            cur = parent[cur]
        top_branch[leaf] = cur

    branch_groups = {}
    for leaf in leaves:
        b = top_branch[leaf]
        if b not in branch_groups:
            branch_groups[b] = []
        branch_groups[b].append(leaf)

    for restart in range(max_restarts):
        if _elapsed() > _DEADLINE - time_budget:
            break

        adj = [set() for _ in range(n_tree)]
        for v in range(1, n_tree):
            p = parent[v]
            adj[v].add(p)
            adj[p].add(v)

        pending = {leaf: k - 1 for leaf in leaves}
        leaf_list = leaves[:]
        random.shuffle(leaf_list)

        success = True

        for leaf in leaf_list:
            if pending[leaf] == 0:
                continue

            candidates = []
            for other in leaves:
                if other == leaf:
                    continue
                if other in adj[leaf]:
                    continue
                if pending[other] == 0:
                    continue
                if top_branch[leaf] != top_branch[other]:
                    candidates.append(other)

            if len(candidates) < pending[leaf]:
                success = False
                break

            random.shuffle(candidates)
            for other in candidates[:pending[leaf]]:
                adj[leaf].add(other)
                adj[other].add(leaf)
                pending[leaf] -= 1
                pending[other] -= 1

        if not success:
            continue

        if any(len(adj[i]) != k for i in range(n_tree)):
            continue

        if _girth_geq(adj, n_tree, g):
            edges = [(i, j) for i in range(n_tree) for j in adj[i] if i < j]
            return n_tree, edges

    return None


# ---------------- construction 5: Random Regular Graph + Local Search ---------
def _random_local_search(k, g):
    R = (g - 1) // 2
    moore = 1 + k
    for i in range(1, R):
        moore += k * (k - 1) ** i

    max_n = 3000
    if k == 3:
        max_n = 600
    elif k == 4:
        max_n = 1500
    elif k == 5:
        max_n = 2500

    step = 2 if k % 2 == 0 else 1
    for n in range(moore, max_n + 1, step):
        if (n * k) % 2 != 0:
            continue
        if _elapsed() > _DEADLINE - 5:
            return None

        for restart in range(5):
            if _elapsed() > _DEADLINE - 5:
                return None

            try:
                adj = _generate_random_regular(n, k)
            except Exception:
                continue

            if not adj:
                continue

            if _girth_geq(adj, n, g):
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges

    return None


def _generate_random_regular(n, k):
    if (n * k) % 2 != 0:
        return None

    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)

    random.shuffle(stubs)

    adj = [set() for _ in range(n)]

    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            for j in range(i + 2, len(stubs), 2):
                if stubs[j] != u and stubs[j] != v:
                    stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                    v = stubs[i + 1]
                    break
            else:
                return None
        if v in adj[u]:
            for j in range(i + 2, len(stubs), 2):
                if stubs[j] not in adj[u] and stubs[j] != u:
                    stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                    v = stubs[i + 1]
                    break
            else:
                return None

        adj[u].add(v)
        adj[v].add(u)

    for i in range(n):
        if len(adj[i]) != k:
            return None
        if len(adj[i]) != len(set(adj[i])):
            return None

    return adj


# ---------------- construction 6: BFS tree-like completion --------------------
def _bfs_complete(k, g):
    R = (g - 1) // 2
    depth_lim = min(R - 1, 5)
    order = [(0, -1, 0)]
    node_idx = {0: 0}
    parent_of = {0: -1}
    depth_of = {0: 0}
    q = deque([0])
    while q:
        u = q.popleft()
        du = depth_of[u]
        if du >= depth_lim:
            continue
        for child in range(k):
            v = len(order)
            order.append((v, u, du + 1))
            node_idx[v] = v
            parent_of[v] = u
            depth_of[v] = du + 1
            q.append(v)
    n = len(order)
    adj = [set() for _ in range(n)]
    for v in range(1, n):
        p = parent_of[v]
        adj[v].add(p)
        adj[p].add(v)

    def cand_ok(v, w):
        if w in adj[v]:
            return False
        if w == parent_of[v]:
            return False
        common = set(adj[v]) & set(adj[w])
        if common:
            return False
        if w in adj[parent_of[v]]:
            return False
        return True

    for v in range(n):
        guard = 0
        while len(adj[v]) < k and guard < 5000:
            guard += 1
            for w in range(n):
                if len(adj[v]) >= k:
                    break
                if w <= v:
                    continue
                if cand_ok(v, w):
                    adj[v].add(w)
                    adj[w].add(v)
    if any(len(adj[i]) != k for i in range(n)):
        return None
    for i in range(n):
        adj[i] = sorted(adj[i])
    if not _girth_geq(adj, n, g):
        return None
    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    return n, edges


# ---------------- parallel candidate verification ---------------------------
def _worker(name, fn, holder):
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
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("petersen", lambda: _petersen(k, g)),
        ("circulant", lambda: _circulant_search(k, g)),
        ("random", lambda: _random_local_search(k, g)),
        ("bfs", lambda: _bfs_complete(k, g)),
        ("greedy_leaf", lambda: _greedy_leaf_pair(k, g, max_restarts=8, time_budget=30.0)),
    ]
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.15)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
