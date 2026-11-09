"""Structural change: greedy maximal biclique extraction + local search (merge/reassign/split)."""
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
EDGES = P['edges']
ALL_EDGE_SET = set()
for a, b in EDGES:
    ALL_EDGE_SET.add((min(a, b), max(a, b)))
ALL_EDGE_LIST = list(ALL_EDGE_SET)


def _edge_key(u, v):
    return (u, v) if u < v else (v, u)


def _is_biclique(left, right, adj):
    if not left or not right:
        return False
    if set(left) & set(right):
        return False
    for u in left:
        for v in right:
            if v not in adj[u]:
                return False
    return True


def _build_adj():
    adj = [set() for _ in range(N)]
    for a, b in ALL_EDGE_LIST:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _greedy_extract(adj, seed=None, max_bicliques=None):
    rng = random.Random(seed)
    remaining = set(ALL_EDGE_SET)
    bicliques = []
    deg = [0] * N
    for a, b in ALL_EDGE_LIST:
        deg[a] += 1
        deg[b] += 1

    while remaining:
        if time.time() > _DEADLINE:
            break
        # pick an edge; prefer high-degree endpoints
        cand = list(remaining)
        rng.shuffle(cand)
        # sort by sum of degrees desc, take top few, pick random
        cand.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
        pick_from = cand[:max(1, len(cand) // 2)]
        u0, v0 = rng.choice(pick_from)

        # grow biclique containing (u0, v0)
        left = {u0}
        right = {v0}
        # candidate pools
        L_pool = set(adj[u0]) | {u0}
        R_pool = set(adj[v0]) | {v0}
        # We need every u in left adjacent to every v in right.
        # Start: left={u0}, right={v0}.
        # Expand: repeatedly try to add a vertex to one side that is adjacent to all on the other side.
        changed = True
        while changed and (max_bicliques is None or len(bicliques) < max_bicliques):
            changed = False
            # try add to left: v in adj of all current right, and v not in right, and (v,u) in remaining for some u? 
            # Actually we only require cross edges be in the graph; they may or may not be in remaining (they must be in remaining to be "covered").
            # For a valid partition, every cross pair must be a graph edge AND we will assign it to this biclique.
            # We can only use edges that are in remaining (not yet covered).
            # So when adding v to left, we need (v, w) in remaining for all w in right.
            best_gain = 0
            best_v = None
            best_side = None
            # try left additions
            for v in L_pool - right:
                ok = True
                gain = 0
                for w in right:
                    if _edge_key(v, w) not in remaining:
                        ok = False
                        break
                    gain += 1
                if ok and gain > best_gain:
                    best_gain = gain
                    best_v = v
                    best_side = 'L'
            # try right additions
            for v in R_pool - left:
                ok = True
                gain = 0
                for w in left:
                    if _edge_key(v, w) not in remaining:
                        ok = False
                        break
                    gain += 1
                if ok and gain > best_gain:
                    best_gain = gain
                    best_v = v
                    best_side = 'R'
            if best_v is not None and best_gain > 0:
                if best_side == 'L':
                    left.add(best_v)
                else:
                    right.add(best_v)
                changed = True
                # update pools
                L_pool = set()
                for w in right:
                    L_pool |= adj[w]
                L_pool |= left
                R_pool = set()
                for w in left:
                    R_pool |= adj[w]
                R_pool |= right

        # Finalize: left and right must be nonempty, disjoint, and all cross edges in remaining.
        if left and right and not (left & right):
            # verify all cross edges are in remaining
            valid = True
            for u in left:
                for v in right:
                    if _edge_key(u, v) not in remaining:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                for u in left:
                    for v in right:
                        remaining.discard(_edge_key(u, v))
                bicliques.append([sorted(left), sorted(right)])
            else:
                # fallback: single edge
                remaining.discard(_edge_key(u0, v0))
                bicliques.append([[u0], [v0]])
        else:
            remaining.discard(_edge_key(u0, v0))
            bicliques.append([[u0], [v0]])

    return bicliques


def _coverage(bicliques):
    cov = set()
    for left, right in bicliques:
        for u in left:
            for v in right:
                cov.add(_edge_key(u, v))
    return cov


def _is_valid(bicliques):
    cov = _coverage(bicliques)
    if cov != ALL_EDGE_SET:
        return False
    for left, right in bicliques:
        if not left or not right:
            return False
        if set(left) & set(right):
            return False
    return True


def _try_merge(bicliques, adj):
    n = len(bicliques)
    for i in range(n):
        for j in range(i + 1, n):
            L1, R1 = bicliques[i]
            L2, R2 = bicliques[j]
            # candidate merged sides: (L1|L2, R1∩R2), (L1∩L2, R1|R2), (L1, R1|R2) if L1⊆L2 etc.
            # Try a few combinations
            combos = [
                (set(L1) | set(L2), set(R1) & set(R2)),
                (set(L1) & set(L2), set(R1) | set(R2)),
                (set(L1), set(R1) | set(R2)) if set(L1) <= set(L2) else None,
                (set(L2), set(R1) & set(R2)) if set(L2) <= set(L1) else None,
                (set(L1) | set(L2), set(R1) & set(R2)),
            ]
            for Lc, Rc in combos:
                if not Lc or not Rc:
                    continue
                if Lc & Rc:
                    continue
                # check all cross edges are graph edges
                ok = True
                for u in Lc:
                    for v in Rc:
                        if v not in adj[u]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    new_bics = bicliques[:i] + bicliques[i + 1:j] + bicliques[j + 1:]
                    new_bics.append([sorted(Lc), sorted(Rc)])
                    if _is_valid(new_bics):
                        return new_bics
    return None


def _try_reassign(bicliques, adj):
    # move an edge from one biclique to another if target can absorb it
    # Represent each biclique's edge set
    for bi in range(len(bicliques)):
        L1, R1 = bicliques[bi]
        edges1 = [_edge_key(u, v) for u in L1 for v in R1]
        for e in edges1:
            u, v = e
            for bj in range(len(bicliques)):
                if bj == bi:
                    continue
                L2, R2 = bicliques[bj]
                # Can we add (u,v) to biclique bj?
                # Need u in L2 or R2 and v in the other side, or we can extend.
                # Simplest: if u in L2 and v in R2 -> already covered (skip, would double cover)
                # We want to reassign: remove from bi, add to bj.
                # For bj to cover (u,v), either u∈L2,v∈R2 (already) or we extend.
                # Try extending bj: add u to L2 (needs u adj all R2) or add v to R2 (needs v adj all L2)
                # or add v to L2 / u to R2.
                options = []
                if v in set(R2) and u not in set(L2):
                    # add u to L2
                    if all(u in adj[w] for w in R2):
                        options.append((set(L2) | {u}, set(R2)))
                if u in set(L2) and v not in set(R2):
                    if all(v in adj[w] for w in L2):
                        options.append((set(L2), set(R2) | {v}))
                if u in set(R2) and v not in set(L2):
                    if all(v in adj[w] for w in R2):
                        options.append((set(L2) | {v}, set(R2)))
                if v in set(L2) and u not in set(R2):
                    if all(u in adj[w] for w in L2):
                        options.append((set(L2), set(R2) | {u}))
                for Lc, Rc in options:
                    if not Lc or not Rc or (Lc & Rc):
                        continue
                    # build new bicliques
                    new_bics = []
                    for k in range(len(bicliques)):
                        if k == bi:
                            L1n, R1n = bicliques[bi]
                            # remove e from bi
                            # If bi is just this edge, remove biclique
                            if len(L1n) == 1 and len(R1n) == 1 and _edge_key(L1n[0], R1n[0]) == e:
                                continue
                            else:
                                # need to shrink; complex. Skip for now unless it's the only edge.
                                new_bics.append(bicliques[k])
                        elif k == bj:
                            new_bics.append([sorted(Lc), sorted(Rc)])
                        else:
                            new_bics.append(bicliques[k])
                    if _is_valid(new_bics):
                        return new_bics
    return None


def _local_search(bicliques, adj, budget=30):
    best = bicliques
    best_len = len(bicliques)
    for _ in range(200):
        if time.time() > _DEADLINE - 5:
            break
        # try merge
        merged = _try_merge(best, adj)
        if merged and len(merged) < best_len:
            best = merged
            best_len = len(best)
            continue
        # try reassign
        reassigned = _try_reassign(best, adj)
        if reassigned and len(reassigned) < best_len:
            best = reassigned
            best_len = len(best)
            continue
        break
    return best


_DEADLINE = None


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 75  # safety margin before 90s kill
    adj = _build_adj()

    # Trivial fallback: star decomposition
    trivial = []
    for u in range(N):
        vs = [v for a, v in EDGES if a == u]
        if vs:
            trivial.append([[u], vs])
    if not trivial:
        # ensure at least one biclique if no edges
        trivial = [[[0], [1]]] if N >= 2 else [[[0], [0]]]
    best = trivial
    best_len = len(trivial)

    rng = random.Random(12345)
    restarts = 0
    while time.time() < _DEADLINE - 10:
        restarts += 1
        seed = rng.randint(0, 10**9)
        bics = _greedy_extract(adj, seed=seed)
        if not bics:
            continue
        if not _is_valid(bics):
            # try to fix by adding missing edges as singletons
            cov = _coverage(bics)
            missing = ALL_EDGE_SET - cov
            for e in missing:
                bics.append([[e[0]], [e[1]]])
            if not _is_valid(bics):
                continue
        if len(bics) < best_len:
            best = bics
            best_len = len(bics)
        # local search
        improved = _local_search(bics, adj)
        if len(improved) < best_len:
            best = improved
            best_len = len(best)
        # early stop if we hit a very good number
        if best_len <= 8:
            break

    # Final validation and type conversion
    if not _is_valid(best):
        # fall back to trivial
        best = trivial

    # Convert to pure Python ints and lists
    result = []
    for left, right in best:
        L = [int(x) for x in left]
        R = [int(x) for x in right]
        result.append([L, R])
    return result
