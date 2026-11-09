import time
import random
import itertools

# EVOLVE-BLOCK-START
def _validate(edges):
    """Check: simple graph, colours in 0..19, every 4-cycle rainbow."""
    n = 20
    adj = {}
    for u, v, c in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            return False
        if not (0 <= c < n):
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in adj:
            return False
        adj[(a, b)] = c
        adj[(b, a)] = c
    # enumerate all 4-cycles (u < v < w < x), check the 3 distinct 4-cycles on {u,v,w,x}
    verts = list(range(n))
    for u, v, w, x in itertools.combinations(verts, 4):
        # cycle u-v-w-x-u
        if all(e in adj for e in [(u, v), (v, w), (w, x), (x, u)]):
            cols = {adj[(u, v)], adj[(v, w)], adj[(w, x)], adj[(x, u)]}
            if len(cols) != 4:
                return False
        # cycle u-v-x-w-u
        if all(e in adj for e in [(u, v), (v, x), (x, w), (w, u)]):
            cols = {adj[(u, v)], adj[(v, x)], adj[(x, w)], adj[(w, u)]}
            if len(cols) != 4:
                return False
        # cycle u-w-v-x-u
        if all(e in adj for e in [(u, w), (w, v), (v, x), (x, u)]):
            cols = {adj[(u, w)], adj[(w, v)], adj[(v, x)], adj[(x, u)]}
            if len(cols) != 4:
                return False
    return True


def _is_valid_addition(adj, u, v, c, n):
    """Check whether adding edge (u,v) colour c keeps all 4-cycles rainbow.
    Only new 4-cycles containing (u,v) need checking: a 4th vertex w such that
    the cycle u-v-w-? -u ... actually a 4-cycle containing edge u-v uses two more
    distinct vertices a,b with edges u-a, a-b, b-v OR u-b, b-a, a-v (i.e. the
    two vertices a,b must be adjacent and each adjacent to one of u,v).
    For each pair (a,b) with a!=b, a,b not in {u,v}, adjacent, and (u adjacent a and b adjacent v)
    or (u adjacent b and a adjacent v), the 4-cycle u-v-b-a-u (or u-v-a-b-u) is formed.
    """
    # neighbours sets
    nu = adj.get(u, set())
    nv = adj.get(v, set())
    # For a 4-cycle u-v-b-a-u: need edge v-b, b-a, a-u.
    # So a in nu, b in nv, a != b, a,b not u,v, and (a,b) in adj.
    for a in nu:
        if a == v:
            continue
        for b in nv:
            if b == u or b == a:
                continue
            # need edge a-b
            if b in adj.get(a, set()):
                # cycle u-v-b-a-u edges: (u,v)=c, (v,b), (b,a), (a,u)
                cols = {c, adj[(v, b)], adj[(b, a)], adj[(a, u)]}
                if len(cols) != 4:
                    return False
    return True


def _greedy(seed, time_limit):
    random.seed(seed)
    n = 20
    adj = {i: set() for i in range(n)}
    col_of = {}
    edges = []

    # candidate edges in random order
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)

    # colour assignment: try to pick a colour that keeps all 4-cycles rainbow.
    # For a given edge (u,v), a colour c is valid if for every 4-cycle it completes,
    # the other three edges already have distinct colours and none equals c.
    def valid_colour(u, v):
        nu = adj.get(u, set())
        nv = adj.get(v, set())
        # collect forbidden colours from each 4-cycle that would be completed
        forbidden = set()
        ok = True
        for a in nu:
            if a == v:
                continue
            for b in nv:
                if b == u or b == a:
                    continue
                if b in adj.get(a, set()):
                    # cycle u-v-b-a-u ; other three edges colours:
                    c1 = col_of[(v, b)]
                    c2 = col_of[(b, a)]
                    c3 = col_of[(a, u)]
                    # the three must be distinct and c not equal to any
                    if len({c1, c2, c3}) != 3:
                        # existing cycle already invalid? shouldn't happen
                        ok = False
                    forbidden.add(c1)
                    forbidden.add(c2)
                    forbidden.add(c3)
        if not ok:
            return None
        for c in range(n):
            if c not in forbidden:
                return c
        return None

    deadline = time.time() + time_limit
    for (u, v) in all_pairs:
        if time.time() > deadline:
            break
        c = valid_colour(u, v)
        if c is None:
            continue
        adj[u].add(v)
        adj[v].add(u)
        col_of[(u, v)] = c
        col_of[(v, u)] = c
        edges.append([u, v, c])
    return edges


def construct():
    start = time.time()
    total_budget = 150.0
    safety = 10.0
    deadline = start + total_budget - safety

    best = None
    best_len = -1

    # Try multiple random seeds with greedy; keep best.
    seed = 12345
    while time.time() < deadline:
        per = max(0.5, min(20.0, deadline - time.time()))
        edges = _greedy(seed, per)
        if len(edges) > best_len:
            # validate to be safe
            if _validate(edges):
                best = edges
                best_len = len(edges)
        seed += 7919

    if best is None:
        # fallback: star (always valid)
        best = [[0, v, 0] for v in range(1, 20)]
    return best
# EVOLVE-BLOCK-END
