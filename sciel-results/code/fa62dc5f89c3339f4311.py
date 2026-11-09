"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import random as _random

# EVOLVE-BLOCK-START

_DEADLINE = _time.time() + 108.0
_DEADLINE_DEADLINE = _time.time() + 117.0


def _now():
    return _time.time()


def _time_left():
    return _DEADLINE - _now()


def _girth_ok(adj, n, g):
    """Exact girth check: no cycle shorter than g via non-backtracking walks."""
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
    """Lazebnik-Ustimenko-Woldar construction."""
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


# ---------------------------------------------------------------------------
# Constraint-satisfaction backtracking: build a k-regular graph vertex by
# vertex while maintaining a set of "forbidden" undirected edges that would
# create a cycle of length < g.  Forward checking prunes candidates.
# ---------------------------------------------------------------------------
def _forbidden_edges_for(adj, n, g, u, v):
    """
    Given current adjacency (partial), check whether adding edge (u,v) would
    create a cycle of length < g.  Returns True if (u,v) is forbidden.
    We check: is there a path of length 1..g-2 from u to v in the current graph
    that does not use edge (u,v)?  If yes, adding (u,v) creates a cycle of
    length path_len+1 <= g-1.
    Since we build incrementally, adj[u] and adj[v] are the current neighbours.
    """
    # BFS from u to v, depth up to g-2, avoiding direct u-v (not present yet)
    if u == v:
        return True
    # quick: if v in adj[u] already -> duplicate, forbidden
    if v in adj[u]:
        return True
    # BFS non-backtracking from u, depth 1..g-2, see if we reach v
    # state: (prev, cur)
    frontier = [(-1, w) for w in adj[u]]  # depth 1
    depth = 1
    seen = set(frontier)
    while frontier and depth <= g - 2:
        nxt = []
        for prev, cur in frontier:
            if cur == v and depth + 1 <= g - 1:
                # path u->...->cur->v has length depth+1, cycle length depth+1
                # wait: path from u to cur has length depth, then cur->v adds 1
                # total path length = depth+1, cycle length = depth+1
                # we need cycle length < g, i.e., depth+1 < g, i.e., depth <= g-2
                # which is our loop condition
                return True
            for w in adj[cur]:
                if w == prev:
                    continue
                if w == u:
                    continue  # would close a cycle through u, but u is start
                st = (cur, w)
                if st not in seen:
                    seen.add(st)
                    nxt.append(st)
        frontier = nxt
        depth += 1
    return False


def _backtrack_build(n, k, g, rng, time_limit=50.0):
    """
    Build a k-regular graph on n vertices with girth >= g using backtracking
    with forward checking.  Vertices are added in order 0..n-1.  At each step,
    we assign k neighbours to the current vertex from {0..v-1} (already built)
    or from future vertices, ensuring no short cycles are created.
    
    Strategy: process vertices in order. For vertex v, choose k neighbours from
    vertices 0..v-1 that are not forbidden. Track degrees of already-assigned
    vertices. At the end, all degrees must be exactly k.
    
    This is complex; a simpler variant: build a random k-regular graph, then
    use 2-switches guided by a forbidden-edge oracle to eliminate short cycles.
    """
    pass  # replaced by the 2-switch approach below


# ---------------------------------------------------------------------------
# Improved 2-switch local search with explicit forbidden-edge checking.
# ---------------------------------------------------------------------------
def _count_short_cycles_fast(adj, n, g):
    """Count short cycles (length 4..g-1) efficiently using BFS from each vertex."""
    count = 0
    for s in range(n):
        # BFS non-backtracking, depth up to g-1
        # state: (prev, cur)
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
                        count += 1
                        continue
                    st = (cur, w)
                    if st not in seen:
                        seen.add(st)
                        nxt.append(st)
            frontier = nxt
            depth += 1
            if count > 500:
                return count
    return count


def _find_one_short_cycle(adj, n, g):
    """Find one short cycle (length < g) or return None."""
    for s in range(n):
        # BFS non-backtracking with path tracking
        frontier = [(-1, v, [s, v]) for v in adj[s]]
        depth = 1
        seen = set()
        for prev, cur, path in frontier:
            seen.add((cur, prev))
        while frontier and depth < g - 1:
            nxt = []
            for prev, cur, path in frontier:
                for w in adj[cur]:
                    if w == prev:
                        continue
                    if w == s:
                        return path
                    st = (cur, w)
                    if st not in seen:
                        seen.add(st)
                        nxt.append((cur, w, path + [w]))
            frontier = nxt
            depth += 1
    return None


def _two_switch_search(adj, n, k, g, rng, max_iters=5000):
    """
    2-switch local search: repeatedly find a short cycle, pick an edge on it,
    and perform a 2-switch that breaks the cycle.  Use a forbidden-edge check
    to ensure the new edges don't create other short cycles (or at least don't
    increase the total count).
    """
    adj = [set(a) for a in adj]
    best_count = _count_short_cycles_fast(adj, n, g)
    for it in range(max_iters):
        if _time_left() < 3.0:
            break
        if best_count == 0:
            return adj
        cyc = _find_one_short_cycle(adj, n, g)
        if cyc is None:
            return adj
        L = len(cyc)
        # Try several random edges on the cycle
        improved = False
        for _ in range(20):
            i = rng.randrange(L)
            a, b = cyc[i], cyc[(i + 1) % L]
            # 2-switch: remove a-b and c-d, add a-c and b-d
            # c must be a neighbour of a (not b), d a neighbour of b (not a)
            # and c-d must be an existing edge
            cand_c = [x for x in adj[a] if x != b]
            cand_d = [x for x in adj[b] if x != a]
            if not cand_c or not cand_d:
                continue
            c = rng.choice(cand_c)
            d = rng.choice(cand_d)
            if c == d:
                continue
            if d not in adj[c]:
                continue
            # Check: a-c and b-d should not be existing edges
            if c in adj[a] and d in adj[b]:
                # would create duplicates
                continue
            # Perform the switch
            adj[a].discard(b); adj[b].discard(a)
            adj[c].discard(d); adj[d].discard(c)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(d); adj[d].add(b)
            new_count = _count_short_cycles_fast(adj, n, g)
            if new_count < best_count:
                best_count = new_count
                improved = True
                break
            else:
                # revert
                adj[a].discard(c); adj[c].discard(a)
                adj[b].discard(d); adj[d].discard(b)
                adj[a].add(b); adj[b].add(a)
                adj[c].add(d); adj[d].add(c)
        if not improved:
            # Try random 2-switches to escape local minima
            for _ in range(10):
                # pick random edge a-b
                a = rng.randrange(n)
                if not adj[a]:
                    continue
                b = rng.choice(list(adj[a]))
                cand_c = [x for x in adj[a] if x != b]
                cand_d = [x for x in adj[b] if x != a]
                if not cand_c or not cand_d:
                    continue
                c = rng.choice(cand_c)
                d = rng.choice(cand_d)
                if c == d or d not in adj[c]:
                    continue
                if c in adj[a] or d in adj[b]:
                    continue
                adj[a].discard(b); adj[b].discard(a)
                adj[c].discard(d); adj[d].discard(c)
                adj[a].add(c); adj[c].add(a)
                adj[b].add(d); adj[d].add(b)
                new_count = _count_short_cycles_fast(adj, n, g)
                if new_count <= best_count:
                    best_count = new_count
                    break
                else:
                    adj[a].discard(c); adj[c].discard(a)
                    adj[b].discard(d); adj[d].discard(b)
                    adj[a].add(b); adj[b].add(a)
                    adj[c].add(d); adj[d].add(c)
    return adj


def _rand_reg(n, k, rng):
    """Generate a random k-regular simple graph via configuration model."""
    if k % 2 == 1:
        return None
    if n < k + 1:
        return None
    for attempt in range(200):
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
            key = (min(a, b), max(a, b))
            if key in edges:
                ok = False
                break
            edges.add(key)
        if ok:
            return sorted(edges)
    return None


def _random_search(k, g, target_n, rng):
    """Try to find a k-regular girth-g graph with n < target_n."""
    candidates = []
    for n in range(target_n - 2, max(150, target_n - 80), -2):
        candidates.append(n)
    rng.shuffle(candidates)
    for n in candidates:
        if _time_left() < 10.0:
            break
        for attempt in range(30):
            if _time_left() < 10.0:
                break
            base = _rand_reg(n, k, rng)
            if base is None:
                continue
            adj = [set() for _ in range(n)]
            for i, j in base:
                adj[i].add(j)
                adj[j].add(i)
            adj = _two_switch_search(adj, n, k, g, rng, max_iters=3000)
            if all(len(a) == k for a in adj) and _girth_ok(adj, n, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
    return None


def construct(k, g):
    global _DEADLINE, _DEADLINE_DEADLINE
    _DEADLINE = _time.time() + 108.0
    _DEADLINE_DEADLINE = _time.time() + 117.0

    # 1) Seed construction
    try:
        n_seed, edges_seed = _luw(k, g)
    except Exception:
        n_seed, edges_seed = None, None
    seed_valid = False
    if n_seed is not None:
        adj = _build_adj(n_seed, edges_seed, k)
        seed_valid = (adj is not None) and _girth_ok(adj, n_seed, g)
    best_n, best_edges = n_seed, edges_seed

    # 2) Randomized 2-switch search for smaller graphs
    rng = _random.Random(42)
    found = None
    if seed_valid and n_seed is not None:
        try:
            found = _random_search(k, g, n_seed, rng)
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
