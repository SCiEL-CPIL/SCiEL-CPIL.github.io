"""Rewritten program: greedy clique cover for split graphs (valid partition, fewer cliques)."""
import time
import random

_DEADLINE = 145.0

def _make_split(n, k, p, seed):
    rng = random.Random(seed)
    C = list(range(k))
    I = list(range(k, n))
    cn = {}
    for v in I:
        s = set()
        for u in C:
            if rng.random() < p:
                s.add(u)
        cn[v] = s
    return C, I, cn

def _clique_edges(cl):
    out = set()
    m = len(cl)
    for i in range(m):
        a = cl[i]
        for j in range(i + 1, m):
            out.add(frozenset((a, cl[j])))
    return out

def _greedy_cover(C, I, cn):
    cliques = []
    used_c_edges = set()
    for v in I:
        S = list(cn[v])
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if frozenset((S[i], S[j])) in used_c_edges:
                        S.pop(j)
                        changed = True
                        break
                if changed:
                    break
        if len(S) >= 1:
            cl = [v] + S
            cliques.append(cl)
            for e in _clique_edges(cl):
                a, b = tuple(e)
                if a in C and b in C:
                    used_c_edges.add(e)
    all_c_edges = set()
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            all_c_edges.add(frozenset((C[i], C[j])))
    rem = all_c_edges - used_c_edges
    while rem:
        deg = {}
        for e in rem:
            a, b = tuple(e)
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        if not deg:
            break
        seed_v = max(deg, key=lambda x: (deg[x], x))
        cand = set()
        for e in rem:
            a, b = tuple(e)
            if a == seed_v:
                cand.add(b)
            elif b == seed_v:
                cand.add(a)
        chosen = [seed_v]
        cand_list = list(cand)
        random.shuffle(cand_list)
        for u in cand_list:
            ok = True
            for w in chosen:
                if w != u and frozenset((u, w)) not in rem:
                    ok = False
                    break
            if ok:
                chosen.append(u)
        if len(chosen) < 2:
            e = next(iter(rem))
            chosen = list(e)
        cliques.append(list(chosen))
        for e in _clique_edges(list(chosen)):
            rem.discard(e)
    return cliques

def construct():
    t0 = time.time()
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for (n, k, p, seed) in specs:
        if time.time() - t0 > _DEADLINE - 5:
            break
        C, I, cn = _make_split(n, k, p, seed)
        cliques = _greedy_cover(C, I, cn)
        results.append([[int(v) for v in cl] for cl in cliques])
    return results
