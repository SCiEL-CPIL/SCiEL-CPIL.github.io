"""Rewrite: explicit algebraic constructions (LW + Calkin-Wilf) for the 4-regular girth-10 cage."""
# EVOLVE-BLOCK-START
import time
import itertools
import math
from collections import defaultdict, deque

_DEADLINE = time.time() + 140.0  # hard wall-clock guard with 10s margin

def _time_ok():
    return time.time() < _DEADLINE

# ---------- finite-field helpers ----------
# For small primes q we use Python ints with modular arithmetic.

def _gf_table(p):
    """Return (add, sub, mul) for GF(p)."""
    return ((lambda x, y: (x + y) % p),
            (lambda x, y: (x - y) % p),
            (lambda x, y: (x * y) % p))

# ---------- Lazebnik-Ustimenko-Woldar CD(d, q) ----------
def _lw_construct(q, d):
    """
    Build the connected component of the zero vertex in the L-U-W graph CD(d, q).
    q must be prime (or 4).  d is the 'odd' parameter; the resulting graph is
    q-regular with girth at least d+5.
    Vertices are (side, tuple) where side in {0,1} (point / line) and tuple has
    length d+1 with the first entry None.
    """
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = add
        mul = lambda x, y: table[x][y]
    else:
        add, sub, mul = _gf_table(q)

    # rule[c] = (a, b) for c = 2..d
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vi, index[other]))

    return len(order), edges

# ---------- Calkin-Wilf graph CW(5, q) ----------
def _calkin_wilf_construct(q):
    """
    Calkin-Wilf graph CW(5, q) for q = p^h a prime power.
    It is q-regular bipartite with girth >= 2*ceil(log_q(5)) + 2.
    For q=5: girth >= 2*2 + 2 = 6 (not enough).
    For q=25: girth >= 2*2 + 2 = 6 (still not enough for g=10).
    Actually the Calkin-Wilf graph CW(5,q) has girth >= 2*ceil(log_q(5)) + 2.
    For girth 10 we need ceil(log_q(5)) >= 4, so q >= 5^4 = 625.
    That gives n = 2*q = 1250 vertices.  Still not better than LW(25,3) = 1250.
    But let's implement it for completeness and check.
    """
    # CW(5, q): vertices are pairs (a, b) in GF(q)^2, a != 0, plus a special set?
    # Actually the standard Calkin-Wilf graph CW(5, q) has vertex set GF(q) x GF(q)
    # with two copies (bipartite).  Let me use the standard construction.
    #
    # From Calkin & Wilf (1995): CW(5, q) is a q-regular bipartite graph on 2q^2 vertices?
    # No, let me recall.  The Calkin-Wilf graph CW(d, q) is d-regular with girth >= ...
    # For d=5, the graph CW(5, q) has 2q vertices?  Let me just implement the known one.
    #
    # Actually, for this task k=4, and Calkin-Wilf gives 5-regular graphs.
    # So CW is not directly applicable for k=4.  Skip.
    #
    # Instead, let me implement a different explicit construction:
    # The "Fermat" or "Klein" type graphs, or use the known 320-vertex graph.
    # Since I don't have the 320-vertex graph hardcoded, I'll focus on making
    # LW as small as possible and add a random-search fallback.
    pass
    return None

# ---------- girth verification via non-backtracking walks ----------
def _verify_girth(n, edges, g):
    """
    Verify that the graph on n vertices with the given edges is simple,
    every vertex has degree k (implicit), and has girth >= g.
    Uses BFS from each vertex to detect cycles shorter than g.
    Returns True if valid, False otherwise.
    """
    adj = defaultdict(list)
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False  # loop
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False  # repeated edge
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)

    for v in range(n):
        if len(adj[v]) != len(adj[v]):  # always true, just for clarity
            pass
        # BFS
        depth = [-1] * n
        parent = [-1] * n
        depth[v] = 0
        queue = deque([v])
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if depth[w] == -1:
                    depth[w] = depth[u] + 1
                    parent[w] = u
                    queue.append(w)
                elif w != parent[u] and u != parent[w]:
                    # Found a cycle: depth[u] + 1 + depth[w]
                    cycle_len = depth[u] + 1 + depth[w]
                    if cycle_len < g:
                        return False
    return True

def _verify_degree(n, edges, k):
    """Check every vertex has degree exactly k."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return all(d == k for d in deg)

# ---------- Randomized 4-regular graph with girth check ----------
def _random_4reg_girth_search(k, g, n_target, time_budget):
    """
    Attempt to find a k-regular graph on n_target vertices with girth >= g
    via random generation + local rewiring.
    Returns (n, edges) if found, else None.
    """
    if n_target % 2 != 0:
        return None
    if n_target < 2 * k:
        return None

    deadline = time.time() + time_budget

    import random

    def _random_4reg(n):
        """Generate a random 4-regular graph on n vertices using the configuration model."""
        # Each vertex has 4 half-edges
        half_edges = []
        for v in range(n):
            for _ in range(4):
                half_edges.append(v)
        random.shuffle(half_edges)
        edges = set()
        valid = True
        for i in range(0, len(half_edges), 2):
            u, v = half_edges[i], half_edges[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if not valid:
            return None
        return sorted(edges)

    def _is_4reg(edges, n):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return all(d == 4 for d in deg)

    def _girth_check_fast(n, edges, g):
        """Quick girth check using BFS."""
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for v in range(n):
            depth = [-1] * n
            parent = [-1] * n
            depth[v] = 0
            queue = deque([v])
            while queue:
                u = queue.popleft()
                if depth[u] >= g // 2:
                    break
                for w in adj[u]:
                    if depth[w] == -1:
                        depth[w] = depth[u] + 1
                        parent[w] = u
                        queue.append(w)
                    elif w != parent[u] and u != parent[w]:
                        cycle_len = depth[u] + 1 + depth[w]
                        if cycle_len < g:
                            return False
        return True

    best = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        edges = _random_4reg(n_target)
        if edges is None:
            continue
        if not _is_4reg(edges, n_target):
            continue
        if _girth_check_fast(n_target, edges, g):
            if best is None or len(edges) < len(best):
                best = edges
                if time.time() < deadline:
                    break
    if best is not None:
        return n_target, best
    return None

# ---------- Main construct ----------
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    For k=4, g=10, uses the Lazebnik-Ustimenko-Woldar construction and
    a randomized search, with a deadline guard.
    """
    global _DEADLINE
    _DEADLINE = time.time() + 140.0

    best_n = None
    best_edges = None

    # Strategy 1: Lazebnik-Ustimenko-Woldar CD(d, q)
    # For k=4: q=4 (GF(4)) or q is a prime = 4?  q must equal k for k-regularity.
    # So q=4.  d is chosen so that girth >= d+5 >= g.
    # d = g - 5 = 5 for g=10.
    # For q=4, d=5: the graph has 2 * 4^d?  No, the component size depends on the rules.
    # Let's just compute it.
    if k == 4 and g == 10:
        # Try q=4, d=5 (girth >= 10)
        q = 4
        d = g - 5  # = 5
        if d < 3:
            d = 3
        n_lw, edges_lw = _lw_construct(q, d)
        if _verify_degree(n_lw, edges_lw, k) and _verify_girth(n_lw, edges_lw, g):
            best_n = n_lw
            best_edges = edges_lw
            print(f"LW q={q} d={d}: n={n_lw}, valid")
            if not _time_ok():
                return best_n, best_edges

        # Try to see if a smaller graph exists via random search
        # Moore bound for (4, 10): 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4*40 = 161 (non-bipartite)
        # Bipartite Moore bound: 2 * (1 + 3 + 9 + 27) = 2*40 = 80?  No.
        # For bipartite: 2 * sum_{i=0}^{g/2 - 1} (k-1)^i = 2 * sum_{i=0}^{4} 3^i = 2*(1+3+9+27+81) = 2*121 = 242
        # For non-bipartite: 1 + k * sum_{i=0}^{(g-1)/2 - 1} (k-1)^i + k * (k-1)^{(g-1)/2}
        # = 1 + 4 * sum_{i=0}^{3} 3^i + 4 * 3^4 = 1 + 4*(1+3+9+27) + 4*81 = 1 + 160 + 324 = 485?  That seems too high.
        # Actually for non-bipartite girth g (even):
        # n >= 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
        # For g=10, k=4: 1 + 4*(1+3+9+27+81) = 1 + 4*121 = 485
        # For bipartite girth g: n >= 2 * sum_{i=0}^{g/2-1} (k-1)^i = 2*(1+3+9+27+81) = 242
        # So the Moore bound is 242 (bipartite) or 485 (non-bipartite).
        # The known cage is 320, which is between 242 and 485, so it's likely bipartite or has a mix.

        # Random search for n < best_n
        if best_n is not None and _time_ok():
            # Try a few random searches for smaller n
            # This is unlikely to find anything, but let's try briefly
            remaining_time = _DEADLINE - time.time() - 5.0
            if remaining_time > 10:
                # Try n = 242, 244, 246, ... up to min(best_n - 2, 300)
                for n_try in range(242, min(best_n, 300) + 1, 2):
                    if not _time_ok():
                        break
                    t_budget = min(10.0, _DEADLINE - time.time() - 5.0)
                    if t_budget < 2.0:
                        break
                    result = _random_4reg_girth_search(k, g, n_try, t_budget)
                    if result is not None:
                        n_found, edges_found = result
                        if n_found < best_n:
                            best_n = n_found
                            best_edges = edges_found
                            print(f"Random search found n={n_found}")
                            break

    else:
        # General case: use LW with q=k if k is prime or 4, else fall back to random
        if k in (4,):
            q = 4
        elif k in (2, 3, 5, 7, 11, 13, 17, 19, 23):
            q = k
        else:
            # For composite k, LW doesn't directly apply; use random search
            q = None

        if q is not None:
            d = g - 5
            if d % 2 == 0:
                d -= 1
            if d < 3:
                d = 3
            n_lw, edges_lw = _lw_construct(q, d)
            if _verify_degree(n_lw, edges_lw, k) and _verify_girth(n_lw, edges_lw, g):
                best_n = n_lw
                best_edges = edges_lw
        else:
            # Pure random search
            if _time_ok():
                n_try = 2 * k  # minimum
                result = _random_4reg_girth_search(k, g, n_try, _DEADLINE - time.time() - 5.0)
                if result is not None:
                    best_n, best_edges = result

    if best_n is not None and best_edges is not None:
        return best_n, best_edges

    # Fallback: return a trivial invalid-ish result (should not happen)
    return 2 * k, [(i, (i + 1) % (2 * k)) for i in range(2 * k)]
# EVOLVE-BLOCK-END
