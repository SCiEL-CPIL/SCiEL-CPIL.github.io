import time
import random
from itertools import combinations

N = 18
DEADLINE = 120.0

def _build(parts, intra):
    """parts: list of vertex lists (3 parts). intra: list of (i, a, b) with a<b in part i.
    Returns (edge_set, adj) for base 3-partite + intra edges."""
    adj = [set() for _ in range(N)]
    edges = set()
    # base 3-partite
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a].add(b); adj[b].add(a)
                    edges.add((min(a, b), max(a, b)))
    for (i, a, b) in intra:
        adj[a].add(b); adj[b].add(a)
        edges.add((min(a, b), max(a, b)))
    return edges, adj

def _is_k4_free(adj):
    # K4 exists iff two adjacent vertices share 2+ common neighbors.
    for u in range(N):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
    return True

def _has_triangle(adj, S):
    """S: frozenset of vertices; True if induced subgraph on S has a triangle."""
    Sset = set(S)
    for u in S:
        for v in adj[u]:
            if v > u and v in Sset:
                if (adj[u] & adj[v]) & Sset:
                    return True
    return False

def _exact_largest_triangle_free(adj):
    """Exact branch and bound: largest subset with no triangle."""
    best = [0]

    def rec(S, size):
        if size > best[0]:
            best[0] = size
        if size + len(S) <= best[0]:
            return
        # prune: if S already contains a triangle, we cannot keep all; but we add vertices,
        # so check current S is triangle-free (invariant).
        # branch
        for i, v in enumerate(S):
            rest = S - {v}
            # if current S is triangle-free, S without v is too
            rec(rest, size)
        # also the branch where we keep everything is size itself (already counted)

    # We want max subset of all vertices with no triangle.
    # Standard: branch on include/exclude with triangle constraint.
    best[0] = 0
    allv = frozenset(range(N))

    def rec2(S, cur):
        # cur is a triangle-free set (frozenset); S are candidates to add.
        if len(cur) > best[0]:
            best[0] = len(cur)
        if len(cur) + len(S) <= best[0]:
            return
        for v in S:
            S2 = S - {v}
            # can we add v?
            if not _has_triangle(adj, cur | {v}):
                rec2(S2, cur | {v})
            rec2(S2, cur)

    rec2(allv, frozenset())
    return best[0]

def _candidate(intra, parts):
    edges, adj = _build(parts, intra)
    if not _is_k4_free(adj):
        return None
    return edges, adj

def construct():
    t0 = time.time()
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]

    # Precompute within-part candidate edges (a<b)
    part_edges = []
    for i in range(3):
        for a, b in combinations(parts[i], 2):
            part_edges.append((i, a, b))

    # A good target: within each part, add edges so that each part has a triangle
    # (3 edges forming a triangle) -> then a part alone has a triangle, forcing
    # triangle-free sets to spread across parts but the between-part density limits them.

    # We'll do a randomized search over small intra sets, verify K4-free, and
    # compute the exact largest triangle-free set. Track the best (smallest) score.

    best_edges = None
    best_score = None
    best_adj = None

    rng = random.Random(12345)

    # Heuristic seed: put a triangle in each part.
    def tri_in_part(i):
        p = parts[i]
        return [(i, p[0], p[1]), (i, p[1], p[2]), (i, p[0], p[2])]

    def try_config(intra):
        nonlocal best_edges, best_score, best_adj
        cand = _candidate(intra, parts)
        if cand is None:
            return
        edges, adj = cand
        # quick upper-bound check: compute exact only if promising
        score = _exact_largest_triangle_free(adj)
        if best_score is None or score < best_score:
            best_score = score
            best_edges = edges
            best_adj = adj

    # Start with one triangle per part.
    seed_intra = []
    for i in range(3):
        seed_intra += tri_in_part(i)
    try_config(seed_intra)

    # Randomized local search: start from seed, add/remove within-part edges.
    def current_intra():
        return list(seed_intra)

    cur = current_intra()
    cur_score = best_score

    def eval_intra(intra):
        cand = _candidate(intra, parts)
        if cand is None:
            return None
        edges, adj = cand
        return _exact_largest_triangle_free(adj)

    # Greedy + random perturbations with deadline guard.
    attempt = 0
    while time.time() - t0 < DEADLINE - 25:
        attempt += 1
        # random restart of intra set around seed
        intra = list(seed_intra)
        # randomly toggle a few within-part edges
        for _ in range(rng.randint(0, 6)):
            e = rng.choice(part_edges)
            if e in intra:
                intra.remove(e)
            else:
                intra.append(e)
        sc = eval_intra(intra)
        if sc is not None:
            if best_score is None or sc < best_score:
                best_score = sc
                cand = _candidate(intra, parts)
                best_edges, best_adj = cand
            # local improvement: try adding single edges that keep K4-free and lower score
            improved = True
            while improved and time.time() - t0 < DEADLINE - 25:
                improved = False
                for e in part_edges:
                    if e in intra:
                        continue
                    intra2 = intra + [e]
                    sc2 = eval_intra(intra2)
                    if sc2 is not None and (best_score is None or sc2 < best_score):
                        intra = intra2
                        best_score = sc2
                        cand = _candidate(intra, parts)
                        best_edges, best_adj = cand
                        improved = True
                        break
                # also try removing an edge to possibly reduce score
                for e in list(intra):
                    intra2 = [x for x in intra if x != e]
                    sc2 = eval_intra(intra2)
                    if sc2 is not None and (best_score is None or sc2 < best_score):
                        intra = intra2
                        best_score = sc2
                        cand = _candidate(intra, parts)
                        best_edges, best_adj = cand
                        improved = True
                        break

    # Fallback: ensure we have something valid (the seed config is always valid).
    if best_edges is None:
        cand = _candidate(seed_intra, parts)
        best_edges, best_adj = cand

    return [[a, b] for (a, b) in best_edges]
