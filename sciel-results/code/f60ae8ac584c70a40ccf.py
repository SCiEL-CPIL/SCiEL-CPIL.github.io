"""Cochromatic partition via exact set-cover over maximal cliques and maximal independent sets."""

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _maximal_sets(n, adj, is_clique):
    """Return all maximal cliques (if is_clique=True) or maximal independent sets."""
    results = []
    all_v = set(range(n))

    def bronk(R, P_set, X):
        if not P_set and not X:
            results.append(frozenset(R))
            return
        union = P_set | X
        if not union:
            return
        # Pivot: choose u in union that minimizes candidates
        if is_clique:
            best_u = max(union, key=lambda u: len(P_set & adj[u]))
            candidates = P_set - adj[best_u]
        else:
            # For independent sets, non-neighbors of u in P_set
            non_neighbors = {v for v in P_set if v != best_u and v not in adj[best_u]}
            best_u = max(union, key=lambda u: len({v for v in P_set if v != u and v not in adj[u]}))
            candidates = {v for v in P_set if v != best_u and v not in adj[best_u]}

        for v in list(candidates):
            if is_clique:
                new_P = P_set & adj[v]
                new_X = X & adj[v]
            else:
                new_P = P_set & (all_v - adj[v] - {v})
                new_X = X & (all_v - adj[v] - {v})
            bronk(R | {v}, new_P, new_X)
            P_set.remove(v)
            X.add(v)

    bronk(frozenset(), set(range(n)), frozenset())
    return [list(s) for s in results]


def construct():
    import time
    start_time = time.time()
    deadline = start_time + 55.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    max_cliques = _maximal_sets(n, adj, is_clique=True)
    max_indeps = _maximal_sets(n, adj, is_clique=False)

    candidates = list(set([frozenset(s) for s in max_cliques] + [frozenset(s) for s in max_indeps]))
    candidates.sort(key=len, reverse=True)

    full_mask = (1 << n) - 1
    cand_masks = []
    for c in candidates:
        mask = 0
        for v in c:
            mask |= (1 << v)
        cand_masks.append(mask)

    best = [n + 1]
    best_partition = None

    def solve(uncovered_mask, chosen_masks, depth):
        nonlocal best, best_partition
        if time.time() > deadline:
            return
        if uncovered_mask == 0:
            if depth < best[0]:
                best[0] = depth
                bp = []
                for m in chosen_masks:
                    verts = [v for v in range(n) if m & (1 << v)]
                    bp.append(verts)
                best_partition = bp
            return
        if depth + 1 >= best[0]:
            return

        for cm in cand_masks:
            if cm & ~uncovered_mask:
                continue
            disjoint = True
            for cm2 in chosen_masks:
                if cm & cm2:
                    disjoint = False
                    break
            if not disjoint:
                continue
            solve(uncovered_mask & ~cm, chosen_masks + [cm], depth + 1)

    solve(full_mask, [], 0)

    if best_partition is not None:
        return [[int(v) for v in part] for part in best_partition]

    # Fallback greedy
    parts = []
    es = {tuple(e) for e in P['edges']}
    es |= {(v, u) for u, v in es}
    for v in range(n):
        for part in parts:
            ok = True
            for u in part:
                if (u, v) in es:
                    # has edge, check all others must have edges
                    for w in part:
                        if w != u and (w, v) not in es:
                            ok = False
                            break
                    if not ok:
                        break
                else:
                    # no edge, check all others must have no edges
                    for w in part:
                        if w != u and (w, v) in es:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                part.append(v)
                break
        else:
            parts.append([v])
    return [[int(v) for v in part] for part in parts]
