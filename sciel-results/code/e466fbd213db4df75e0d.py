"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _mkedge(u, v):
    return (u, v) if u < v else (v, u)


def _is_biclique(L, R, host):
    # Every cross pair must be a host edge
    for a in L:
        for b in R:
            if _mkedge(a, b) not in host:
                return False
    return True


def _max_biclique(seed_u, seed_v, host, covered):
    # Build maximal biclique containing seed edge, using only uncovered edges
    L = {seed_u}
    R = {seed_v}
    changed = True
    while changed:
        changed = False
        # Expand L: vertices connected (uncovered) to all of R
        for w in host[seed_u]:
            if w in L or w in R:
                continue
            ok = True
            for r in R:
                if _mkedge(w, r) in covered:
                    ok = False
                    break
            if ok:
                L.add(w)
                changed = True
        # Expand R: vertices connected (uncovered) to all of L
        for w in host[seed_v]:
            if w in L or w in R:
                continue
            ok = True
            for l in L:
                if _mkedge(w, l) in covered:
                    ok = False
                    break
            if ok:
                R.add(w)
                changed = True
    R = R - L
    return L, R


def _greedy(host, all_edges, deadline):
    n = P['n']
    covered = set()
    uncovered = set(all_edges)
    bicliques = []
    while uncovered:
        if time.time() > deadline:
            break
        # Pick seed edge with largest common-neighborhood in uncovered subgraph
        best_e = None
        best_score = -1
        for e in uncovered:
            u, v = e
            # common neighbors w where both (u,w) and (v,w) uncovered
            common = 0
            for w in host[u]:
                if _mkedge(u, w) in uncovered and _mkedge(v, w) in uncovered:
                    common += 1
            score = common * 100 + len(host[u]) + len(host[v])
            if score > best_score:
                best_score = score
                best_e = e
        u, v = best_e
        L, R = _max_biclique(u, v, host, covered)
        if not L or not R:
            L, R = {u}, {v}
        bicliques.append((L, R))
        for a in L:
            for b in R:
                covered.add(_mkedge(a, b))
    # Fallback for any leftover
    for e in uncovered:
        u, v = e
        bicliques.append(({u}, {v}))
    return bicliques


def _merge_pass(bicliques, host, all_edges, deadline):
    # Repeatedly merge two bicliques whose union is a valid biclique
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        m = len(bicliques)
        for i in range(m):
            if time.time() > deadline:
                return bicliques
            for j in range(i + 1, m):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                # Candidate orientations: (L1∪L2, R1∪R2) and (L1∪R2, R1∪L2)
                done = False
                for Lc, Rc in ((L1 | L2, R1 | R2), (L1 | R2, R1 | L2)):
                    if not Lc or not Rc:
                        continue
                    if Lc & Rc:
                        continue
                    # union must be a host biclique
                    if not _is_biclique(Lc, Rc, host):
                        continue
                    # union's edge set must equal union of the two edge sets
                    e1 = set()
                    for a in L1:
                        for b in R1:
                            e1.add(_mkedge(a, b))
                    e2 = set()
                    for a in L2:
                        for b in R2:
                            e2.add(_mkedge(a, b))
                    e_new = set()
                    for a in Lc:
                        for b in Rc:
                            e_new.add(_mkedge(a, b))
                    if e_new == (e1 | e2):
                        new = (Lc, Rc)
                        rest = [bicliques[k] for k in range(m) if k != i and k != j]
                        bicliques = rest + [new]
                        changed = True
                        done = True
                        break
                if done:
                    break
    return bicliques


def construct():
    start = time.time()
    deadline = start + 80.0
    n = P['n']

    host = [set() for _ in range(n)]
    for u, v in P['edges']:
        host[u].add(v)
        host[v].add(u)
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add(_mkedge(u, v))

    # Greedy partition
    bicliques = _greedy(host, all_edges, deadline)

    # Merge refinement
    bicliques = _merge_pass(bicliques, host, all_edges, deadline)

    # Build result with exact types
    result = []
    for L, R in bicliques:
        left = [int(x) for x in sorted(L)]
        right = [int(x) for x in sorted(R)]
        if left and right and not (set(left) & set(right)):
            result.append([left, right])

    return result
# EVOLVE-BLOCK-END
