"""Path decomposition via maximum edge-pairing (transition) local search."""
import time
from random import Random

_START = time.time()
_DEADLINE = _START + 130.0  # explicit wall-clock guard (limit is 150 s)


def _instance(n, p, seed):
    """Reproduce the verifier's instance_graph exactly."""
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _solve(edges, seed):
    """Return (n_paths, paths) for one graph using transition local search."""
    if not edges:
        return 0, []
    rng = Random(seed)

    # edge index -> endpoints
    ep = [None] * len(edges)
    # vertex -> list of incident edge indices
    adj = {}
    for i, (u, v) in enumerate(edges):
        ep[i] = (u, v)
        adj.setdefault(u, []).append(i)
        adj.setdefault(v, []).append(i)
    m = len(edges)

    # partner[e] = the edge paired with e at their shared vertex, or -1.
    # Invariant: partner[partner[e]] == e (or -1), and each vertex's incident
    # edges are partitioned into pairs/singletons.
    partner = [-1] * m
    for v in adj:
        lst = adj[v]
        for k in range(0, len(lst) - 1, 2):
            a, b = lst[k], lst[k + 1]
            partner[a] = b
            partner[b] = a

    def transitions():
        return sum(1 for e in range(m) if partner[e] != -1 and e < partner[e])

    def cycle2(a, b):
        # True iff pairing a-b creates a 2-cycle (a non-simple cycle of length 2).
        # Only possible if a,b share vertex v and their other endpoints are joined by an edge.
        ua, va = ep[a]
        ub, vb = ep[b]
        if ua == ub:
            x, y = va, vb
        elif ua == vb:
            x, y = va, ub
        elif va == ub:
            x, y = ua, vb
        elif va == vb:
            x, y = ua, vb
        else:
            return False
        if x == y:
            return False
        return (min(x, y), max(x, y)) in edge_set

    edge_set = set(edges)

    # Precompute, per vertex, the list of "forbidden" partner pairs (would make a 2-cycle).
    forbidden = {}
    for v in adj:
        lst = adj[v]
        others = {}
        for e in lst:
            u, w = ep[e]
            others[e] = w if u == v else u
        fset = set()
        for i in range(len(lst)):
            a = lst[i]
            oa = others[a]
            for j in range(i + 1, len(lst)):
                b = lst[j]
                ob = others[b]
                if oa == ob:
                    continue
                if (min(oa, ob), max(oa, ob)) in edge_set:
                    fset.add((a, b) if a < b else (b, a))
        forbidden[v] = fset

    best = transitions()

    # Build candidate move list: for each vertex, all pairs of incident edges.
    moves = []
    for v in adj:
        lst = adj[v]
        L = len(lst)
        for i in range(L):
            for j in range(i + 1, L):
                moves.append((v, lst[i], lst[j]))

    def try_swap(v, a, b):
        # Non-recursive 2-opt: swap the transition partners of a and b at vertex v.
        pa = partner[a]
        pb = partner[b]
        if pa == b or pb == a:
            return 0  # a,b already paired at v -> no change
        # Remove the two old transitions (if any) and add two new ones.
        delta = 0
        if pa != -1:
            delta -= 1
            if (min(pa, b), max(pa, b)) in forbidden[v]:
                return None  # would create a 2-cycle
        if pb != -1:
            delta -= 1
            if (min(pb, a), max(pb, a)) in forbidden[v]:
                return None
        # new pairs a-b (at v) and (if any) pa-pb, pb-pa at their vertices
        if (a, b) in forbidden[v] or (b, a) in forbidden[v]:
            return None
        if pa != -1 and pb != -1:
            # pa's other vertex and pb's other vertex must be same to pair them; they are
            # the vertices where a and b were previously paired (not v).
            if (min(pa, pb), max(pa, pb)) in forbidden.get(v, set()):
                return None
            delta += 1
        return delta

    def apply_swap(v, a, b):
        pa = partner[a]
        pb = partner[b]
        partner[a] = b
        partner[b] = a
        if pa != -1:
            partner[pa] = pb
        if pb != -1:
            partner[pb] = pa

    # Local search with random restarts and simulated-annealing-ish acceptance.
    best_partner = None
    for restart in range(40):
        if time.time() > _DEADLINE - 5.0:
            break
        # fresh random pairing
        partner = [-1] * m
        for v in adj:
            lst = adj[v][:]
            rng.shuffle(lst)
            for k in range(0, len(lst) - 1, 2):
                a, b = lst[k], lst[k + 1]
                partner[a] = b
                partner[b] = a
        cur = transitions()
        if cur > best:
            best = cur
            best_partner = partner[:]
        # hill-climb with limited restarts
        temp = 2.0
        for it in range(4000):
            if time.time() > _DEADLINE - 5.0:
                break
            v, a, b = moves[rng.randrange(len(moves))]
            d = try_swap(v, a, b)
            if d is None:
                continue
            if d > 0 or (d < 0 and rng.random() < (0.02 / (1.0 + it * 0.001))):
                apply_swap(v, a, b)
                cur += d
                if cur > best:
                    best = cur
                    best_partner = partner[:]

    # Reconstruct paths from best pairing.
    if best_partner is None:
        # fallback: every edge its own path
        return m, [list(e) for e in edges]
    partner = best_partner
    # path heads: edges with no predecessor. A path alternates start->...; we walk
    # following partner links, but partner is symmetric (transition). We need direction:
    # build the path graph where consecutive edges are linked; it's a set of paths.
    # Represent each edge as a node; an undirected "adjacency in path" link exists
    # between paired edges. We need to orient. Simplest: treat transitions as edges
    # between edge-nodes; the resulting graph is a collection of paths (degree<=2).
    # Build adjacency among edge-nodes.
    link = [[] for _ in range(m)]
    for e in range(m):
        p = partner[e]
        if p != -1 and e < p:
            link[e].append(p)
            link[p].append(e)
    # Walk each path.
    used = [False] * m
    paths = []
    for s in range(m):
        if used[s]:
            continue
        if not link[s]:
            used[s] = True
            paths.append([list(ep[s])])
            continue
        # find an endpoint of this path component (degree 1)
        # walk from s
        comp = []
        # find endpoint
        e = s
        while link[e] and not used[e]:
            pass
        # choose endpoint: a node in this component with degree 1
        # gather component
        stack = [s]
        seen = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            for nb in link[x]:
                if nb not in seen:
                    stack.append(nb)
        comp = list(seen)
        # endpoint = node with degree 1 (or the only node)
        endpoint = None
        for x in comp:
            if len(link[x]) == 1:
                endpoint = x
                break
        if endpoint is None:
            endpoint = comp[0]
        # walk
        path_edges = []
        prev = -1
        cur = endpoint
        while True:
            used[cur] = True
            path_edges.append(cur)
            nxt = -1
            for nb in link[cur]:
                if nb != prev and not used[nb]:
                    nxt = nb
                    break
            if nxt == -1:
                break
            prev, cur = cur, nxt
        # build vertex list from edge sequence
        verts = []
        for idx, e in enumerate(path_edges):
            u, w = ep[e]
            if idx == 0:
                verts.append(u)
                verts.append(w)
            else:
                pu, pw = ep[path_edges[idx - 1]]
                # last vertex in verts must be one endpoint of e
                last = verts[-1]
                if last == u:
                    verts.append(w)
                else:
                    verts.append(u)
        paths.append(verts)
    return len(paths), paths


def construct():
    results = []
    seeds = (1, 2, 3)
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    for (n, p, s), rseed in zip(specs, seeds):
        edges = _instance(n, p, s)
        np_, paths = _solve(edges, rseed * 1000 + 7)
        results.append(paths)
    return results
