"""Clique partition of two fixed dense random graphs.

Strategy:
  1. Build the graph exactly as the verifier does (Random(seed), random()<p).
  2. Enumerate all maximal cliques (Bron–Kerbosch + pivot).
  3. Greedily pick maximal cliques (size>=2) that cover the most still-uncovered
     edges; this covers almost every edge (dense graphs -> most edges in a triangle).
  4. Patch any residual edges with 2-cliques (guarantees a valid partition).
  5. Post-process: greedily merge overlapping cliques (when the union is a clique
     and no edge is shared) to reduce the number of cliques.
"""
import time

_T0 = time.time()
_DEADLINE = 112.0  # seconds; safety margin under the 120s wall clock


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _build(n, p, seed):
    """Return adjacency bitmasks and the edge list, identical to the verifier."""
    from random import Random
    rng = Random(seed)
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                edges.append((u, v))
    return adj, edges


def _maximal_cliques(adj):
    n = len(adj)
    out = []

    def bk(R, P, X):
        if _remaining() <= 0.0:
            return
        if not P and not X:
            if R:
                out.append(R)
            return
        # pivot: vertex in P|X with the most neighbors in P
        ux = P | X
        best = -1
        bestc = -1
        t = ux
        while t:
            lsb = t & (-t)
            v = (lsb.bit_length() - 1)
            t -= lsb
            c = (adj[v] & P).bit_count()
            if c > bestc:
                bestc = c
                best = v
        if best >= 0:
            cand = P & ~(adj[best])
        else:
            cand = P
        t = cand
        while t:
            lsb = t & (-t)
            v = (lsb.bit_length() - 1)
            t -= lsb
            bk(R | lsb, P & adj[v], X & adj[v])
            P ^= lsb
            X |= lsb

    bk(0, (1 << n) - 1, 0)
    return out


def _solve(n, p, seed):
    adj, edges = _build(n, p, seed)
    cliques = _maximal_cliques(adj)

    # keep only cliques with >= 2 vertices
    cliques = [c for c in cliques if c.bit_count() >= 2]
    if not cliques:
        return [[list(e)] for e in edges]

    # edge -> list of cliques containing it
    edge_map = {}
    for c in cliques:
        vs = []
        t = c
        while t:
            lsb = t & (-t)
            vs.append(lsb.bit_length() - 1)
            t -= lsb
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                a, b = vs[i], vs[j]
                key = (a, b) if a < b else (b, a)
                edge_map.setdefault(key, []).append(c)

    covered = set()
    chosen = []
    for c in sorted(cliques, key=lambda c: -c.bit_count()):
        if _remaining() <= 0.0:
            break
        gain = 0
        for (a, b) in ((vs_i, vs_j) for vs_i in [] for vs_j in []):
            pass
        # compute edges of c not yet covered
        vs = []
        t = c
        while t:
            lsb = t & (-t)
            vs.append(lsb.bit_length() - 1)
            t -= lsb
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                a, b = vs[i], vs[j]
                key = (a, b) if a < b else (b, a)
                if key not in covered:
                    gain += 1
        if gain > 0:
            chosen.append(c)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, b = vs[i], vs[j]
                    key = (a, b) if a < b else (b, a)
                    covered.add(key)

    # patch residual edges with 2-cliques
    for e in edges:
        key = e if e[0] < e[1] else (e[1], e[0])
        if key not in covered:
            chosen.append(1 << e[0] | 1 << e[1])
            covered.add(key)

    # ---- post-processing: merge overlapping cliques to reduce count ----
    def to_list(c):
        vs = []
        t = c
        while t:
            lsb = t & (-t)
            vs.append(lsb.bit_length() - 1)
            t -= lsb
        return vs

    cur = [to_list(c) for c in chosen]

    # edge -> index of clique that owns it (must be exactly one)
    edge_owner = {}
    for idx, vs in enumerate(cur):
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                a, b = vs[i], vs[j]
                key = (a, b) if a < b else (b, a)
                edge_owner[key] = idx

    def is_clique(vs):
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                if not (adj[vs[i]] >> vs[j]) & 1:
                    return False
        return True

    def merge_pass():
        changed = True
        while changed and _remaining() > 0.0:
            changed = False
            # rebuild structures
            edge_owner = {}
            for idx, vs in enumerate(cur):
                for i in range(len(vs)):
                    for j in range(i + 1, len(vs)):
                        a, b = vs[i], vs[j]
                        key = (a, b) if a < b else (b, a)
                        edge_owner[key] = idx
            for i in range(len(cur)):
                if _remaining() <= 0.0:
                    break
                if i >= len(cur):
                    break
                vi = cur[i]
                for j in range(i + 1, len(cur)):
                    vj = cur[j]
                    # find a shared edge
                    si = {(a, b) if a < b else (b, a)
                          for a in vi for b in vi if a < b}
                    shared = False
                    for k in range(len(vj)):
                        for l in range(k + 1, len(vj)):
                            a, b = vj[k], vj[l]
                            key = (a, b) if a < b else (b, a)
                            if key in si:
                                shared = True
                                break
                        if shared:
                            break
                    if not shared:
                        continue
                    union = sorted(set(vi) | set(vj))
                    if len(union) < 2:
                        continue
                    if not is_clique(union):
                        continue
                    # merge j into i
                    cur[i] = union
                    cur[j] = []
                    changed = True
        # drop empties
        return [vs for vs in cur if len(vs) >= 2]

    cur = merge_pass()
    return cur


def construct():
    r1 = _solve(32, 0.6, 601)
    r2 = _solve(64, 0.55, 602)
    return [r1, r2]
