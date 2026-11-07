"""Biclique partition: grow-based greedy set-cover + enhanced local search."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

_N = P['n']
_ADJ = [set() for _ in range(_N)]
_EDGE_SET = set()
for _a, _b in P['edges']:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
    _EDGE_SET.add((min(_a, _b), max(_a, _b)))


def _star_partition():
    """Trivial valid partition: one biclique per vertex covering edges to higher-index neighbors."""
    res = []
    for u in range(_N):
        nbrs = [v for v in _ADJ[u] if v > u]
        if nbrs:
            res.append(([u], list(nbrs)))
    return res


def _valid_partition(bics):
    """Return True iff bics is an exact, non-overlapping, host-edge biclique partition."""
    covered = set()
    for L, R in bics:
        if not L or not R:
            return False
        if set(L) & set(R):
            return False
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in _EDGE_SET:
                    return False
                if e in covered:
                    return False
                covered.add(e)
    return covered == _EDGE_SET


def _biclique_edges(L, R):
    """Return set of edges in biclique (L,R)."""
    es = set()
    for u in L:
        for v in R:
            es.add((min(u, v), max(u, v)))
    return es


def _greedy_construct(deadline, seed=None):
    """Grow-based greedy set cover with optional randomization."""
    rng = random.Random(seed)
    remaining = set(_EDGE_SET)
    bics = []
    while remaining and time.time() < deadline:
        best_L = None
        best_R = None
        best_count = 0

        # Single-vertex left sides
        for u in range(_N):
            R = [v for v in _ADJ[u] if (min(u, v), max(u, v)) in remaining]
            if len(R) > best_count:
                best_count = len(R)
                best_L = [u]
                best_R = R

        # Seed-edge bicliques with growing L
        edges_list = list(remaining)
        if seed is not None:
            rng.shuffle(edges_list)
        for e in edges_list:
            if time.time() >= deadline:
                break
            a, b = e
            R = [v for v in _ADJ[a] if (min(a, v), max(a, v)) in remaining]
            if not R:
                continue
            Rset = set(R)
            L = [a]
            Lset = set(L)
            changed = True
            while changed:
                changed = False
                for u in _ADJ[b]:
                    if u in Lset:
                        continue
                    if all((min(u, v), max(u, v)) in remaining for v in R):
                        L.append(u)
                        Lset.add(u)
                        changed = True
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R

        if best_L is None or best_count == 0:
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
            best_count = 1

        bics.append((best_L, best_R))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))

    for e in list(remaining):
        bics.append(([e[0]], [e[1]]))
    return bics


def _try_merge(bics, i, j):
    """Try to merge bicliques i and j. Return new list or None."""
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    L = list(set(L1) | set(L2))
    R = list(set(R1) | set(R2))
    if not L or not R:
        return None
    if set(L) & set(R):
        return None
    # Check all cross pairs are host edges
    for u in L:
        for v in R:
            if (min(u, v), max(u, v)) not in _EDGE_SET:
                return None
    # Merged edge set must equal union (no new edges introduced)
    e1 = _biclique_edges(L1, R1)
    e2 = _biclique_edges(L2, R2)
    eu = _biclique_edges(L, R)
    if eu != (e1 | e2):
        return None
    new_bics = [b for k, b in enumerate(bics) if k not in (i, j)]
    new_bics.append((L, R))
    return new_bics


def _try_merge_three(bics, i, j, k):
    """Try to merge bicliques i, j, k. Return new list or None."""
    L = list(set(bics[i][0]) | set(bics[j][0]) | set(bics[k][0]))
    R = list(set(bics[i][1]) | set(bics[j][1]) | set(bics[k][1]))
    if not L or not R:
        return None
    if set(L) & set(R):
        return None
    for u in L:
        for v in R:
            if (min(u, v), max(u, v)) not in _EDGE_SET:
                return None
    e_u = set()
    for idx in (i, j, k):
        e_u |= _biclique_edges(bics[idx][0], bics[idx][1])
    eu = _biclique_edges(L, R)
    if eu != e_u:
        return None
    new_bics = [b for idx, b in enumerate(bics) if idx not in (i, j, k)]
    new_bics.append((L, R))
    return new_bics


def _swap_vertex(bics, i, j, u):
    """
    Try to move vertex u from biclique i to biclique j (or vice versa),
    possibly removing u from its original biclique if it becomes empty.
    Return new bics list or None.
    """
    L1, R1 = bics[i]
    L2, R2 = bics[j]
    s1 = set(L1)
    s2 = set(R1)
    t1 = set(L2)
    t2 = set(R2)

    # Determine which side of biclique i contains u
    if u in s1:
        # u is in L1; try moving to L2 or R2
        candidates = [('L', L2, t1, t2, 1), ('R', R2, t1, t2, 0)]
    elif u in s2:
        # u is in R1; try moving to L2 or R2
        candidates = [('L', L2, t1, t2, 1), ('R', R2, t1, t2, 0)]
    else:
        return None

    for side, target, tA, tB, _ in candidates:
        # Try adding u to target side of biclique j
        # Check u is not already in the other side of biclique j
        if side == 'L':
            if u in t2:
                continue
        else:
            if u in t1:
                continue

        # Build new biclique j
        new_L2 = list(tA)
        new_R2 = list(tB)
        if side == 'L':
            new_L2.append(u)
        else:
            new_R2.append(u)

        if not new_L2 or not new_R2:
            continue
        if set(new_L2) & set(new_R2):
            continue

        # Check all cross pairs in new biclique j are host edges
        ok = True
        for a in new_L2:
            for b in new_R2:
                if (min(a, b), max(a, b)) not in _EDGE_SET:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # Build new biclique i (with u removed)
        if u in s1:
            new_L1 = [x for x in L1 if x != u]
            new_R1 = list(R1)
        else:
            new_L1 = list(L1)
            new_R1 = [x for x in R1 if x != u]

        # If biclique i becomes empty on either side, we can drop it
        drop_i = False
        if not new_L1 or not new_R1:
            drop_i = True
        else:
            # Verify biclique i is still valid (it should be, since we only removed a vertex)
            # But we need to ensure no edge is lost that isn't covered by biclique j
            # Actually, the edges incident to u in biclique i must now be covered by biclique j
            # Let's check: edges of u in original biclique i
            if u in s1:
                # u was in L1, so edges (u, v) for v in R1
                for v in R1:
                    e = (min(u, v), max(u, v))
                    # This edge must be in new biclique j
                    if (u in set(new_L2) and v in set(new_R2)) or (u in set(new_R2) and v in set(new_L2)):
                        pass
                    else:
                        ok = False
                        break
            else:
                # u was in R1, so edges (v, u) for v in L1
                for v in L1:
                    e = (min(v, u), max(v, u))
                    if (u in set(new_L2) and v in set(new_R2)) or (u in set(new_R2) and v in set(new_L2)):
                        pass
                    else:
                        ok = False
                        break
            if not ok:
                continue

        # Verify no overlap between the two bicliques
        if not drop_i:
            if set(new_L1) & set(new_R2) or set(new_R1) & set(new_L2):
                continue
            # Check edge overlap
            e1 = _biclique_edges(new_L1, new_R1)
            e2 = _biclique_edges(new_L2, new_R2)
            if e1 & e2:
                continue

        # Build result
        new_bics = list(bics)
        new_bics[j] = (new_L2, new_R2)
        if drop_i:
            new_bics[i] = None
        else:
            new_bics[i] = (new_L1, new_R1)

        # Filter out None
        result = [b for b in new_bics if b is not None]
        return result

    return None


def _local_search(bics, deadline):
    """Repeatedly merge pairs (and triples) until no improvement or deadline."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        # Try pairwise merges
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _try_merge(bics, i, j)
                if new_bics is not None:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # Try triple merges (only if we have >= 3 bicliques)
        n = len(bics)
        if n >= 3:
            for i in range(n):
                if time.time() >= deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() >= deadline:
                        break
                    for k in range(j + 1, n):
                        new_bics = _try_merge_three(bics, i, j, k)
                        if new_bics is not None:
                            bics = new_bics
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return bics


def _split_repack(bics, idxs, deadline):
    """
    Take bicliques at indices idxs, collect their edges, and re-partition
    them using a fresh greedy pass on just those edges.
    """
    if len(idxs) < 2:
        return bics
    edges_to_repack = set()
    for i in idxs:
        L, R = bics[i]
        for u in L:
            for v in R:
                edges_to_repack.add((min(u, v), max(u, v)))

    # Greedy re-partition of just these edges
    remaining = set(edges_to_repack)
    new_bics = []
    while remaining and time.time() < deadline:
        best_L = None
        best_R = None
        best_count = 0
        for u in range(_N):
            R = [v for v in _ADJ[u] if (min(u, v), max(u, v)) in remaining]
            if len(R) > best_count:
                best_count = len(R)
                best_L = [u]
                best_R = R
        # Also try growing from each remaining edge
        for e in list(remaining):
            a, b = e
            R = [v for v in _ADJ[a] if (min(a, v), max(a, v)) in remaining]
            if not R:
                continue
            L = [a]
            Lset = set(L)
            changed = True
            while changed:
                changed = False
                for u in _ADJ[b]:
                    if u in Lset:
                        continue
                    if all((min(u, v), max(u, v)) in remaining for v in R):
                        L.append(u)
                        Lset.add(u)
                        changed = True
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_L = L
                best_R = R
        if best_L is None or best_count == 0:
            e = next(iter(remaining))
            best_L = [e[0]]
            best_R = [e[1]]
        new_bics.append((best_L, best_R))
        for u in best_L:
            for v in best_R:
                remaining.discard((min(u, v), max(u, v)))
    for e in list(remaining):
        new_bics.append(([e[0]], [e[1]]))

    # Only accept if it reduces the count
    if len(new_bics) < len(idxs):
        other_bics = [b for k, b in enumerate(bics) if k not in idxs]
        other_bics.extend(new_bics)
        return other_bics
    return bics


def _local_search_enhanced(bics, deadline):
    """Merge pass + vertex swaps + split-repack on pairs of bicliques."""
    # First: standard merge pass (pairs + triples)
    bics = _local_search(bics, deadline)

    # Second: vertex swap pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                if time.time() >= deadline:
                    break
                # Try moving each vertex from biclique i to biclique j
                for u in bics[i][0] + bics[i][1]:
                    new_bics = _swap_vertex(bics, i, j, u)
                    if new_bics is not None and len(new_bics) < n:
                        bics = new_bics
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            # Re-run merge after swap
            bics = _local_search(bics, deadline)

    # Third: try split-repack on pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(bics)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i + 1, n):
                new_bics = _split_repack(bics, [i, j], deadline)
                if new_bics is not bics and len(new_bics) < n:
                    bics = new_bics
                    improved = True
                    break
            if improved:
                break
    return bics


def construct():
    deadline = time.time() + 80.0

    best_bics = None
    best_count = float('inf')

    # Multiple greedy restarts with different seeds
    for seed in [None, 42, 123, 7, 999, 31415, 271828, 161803]:
        if time.time() >= deadline - 5.0:
            break
        bics = _greedy_construct(deadline - 2.0, seed=seed)
        bics = _local_search_enhanced(bics, deadline - 1.0)

        if not _valid_partition(bics):
            continue

        if len(bics) < best_count:
            best_count = len(bics)
            best_bics = bics

    if best_bics is None:
        best_bics = _star_partition()

    result = [[list(map(int, L)), list(map(int, R))] for L, R in best_bics]
    return result
# EVOLVE-BLOCK-END
