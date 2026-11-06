import time
import random
from collections import defaultdict

def _build_steiner_2_5_25():
    """Return all 5-subsets (blocks) of {0..24} forming S(2,5,25).
    Built by a known base block set + translation group of size 25.
    25 base blocks -> 625 blocks total."""
    base = [
        (0, 1, 2, 3, 4),
        (0, 1, 2, 5, 6),
        (0, 1, 3, 5, 7),
        (0, 1, 4, 5, 8),
        (0, 1, 4, 6, 9),
        (0, 2, 3, 5, 10),
        (0, 2, 3, 6, 11),
        (0, 2, 4, 6, 12),
        (0, 2, 4, 7, 13),
        (0, 2, 5, 7, 14),
        (0, 3, 4, 5, 15),
        (0, 3, 4, 6, 16),
        (0, 3, 5, 8, 17),
        (0, 3, 6, 7, 18),
        (0, 3, 7, 8, 19),
        (0, 4, 6, 8, 20),
        (0, 4, 7, 9, 21),
        (0, 4, 8, 9, 22),
        (0, 5, 6, 8, 23),
        (0, 5, 6, 9, 24),
        (1, 2, 3, 5, 10),
        (1, 2, 3, 6, 11),
        (1, 2, 4, 6, 12),
        (1, 2, 4, 7, 13),
        (1, 2, 5, 7, 14),
    ]
    V = 25
    blocks = []
    seen = set()
    for b in base:
        for t in range(V):
            tb = tuple(sorted((x + t) % V for x in b))
            if tb not in seen:
                seen.add(tb)
                blocks.append(tb)
    return blocks

def _verify_steiner(blocks, v, k=5, t=2):
    """Verify S(2,5,25): every pair in exactly one block."""
    V = v
    pair_count = defaultdict(int)
    for b in blocks:
        if len(b) != k:
            return False
        for i in range(k):
            for j in range(i + 1, k):
                pair_count[(b[i], b[j])] += 1
    total_pairs = V * (V - 1) // 2
    for a in range(V):
        for c in range(a + 1, V):
            if pair_count[(a, c)] != 1:
                return False
    if len(pair_count) != total_pairs:
        return False
    return True

def _greedy_hit(blocks_mask, V, seed):
    """Greedy hitting set: repeatedly pick the point covering the most
    currently-unhit blocks. Returns list of chosen points (the hitting set)."""
    rng = random.Random(seed)
    unhit = [True] * len(blocks_mask)
    # coverage: for each point, number of unhit blocks it hits
    cover = [0] * V
    for bi, m in enumerate(blocks_mask):
        if not unhit[bi]:
            continue
        mm = m
        while mm:
            lsb = mm & (-mm)
            p = lsb.bit_length() - 1
            cover[p] += 1
            mm ^= lsb
    chosen = []
    while True:
        # find any unhit block; if none, done
        any_unhit = False
        for bi in range(len(blocks_mask)):
            if unhit[bi]:
                any_unhit = True
                break
        if not any_unhit:
            break
        # pick point with max cover (tie-break random)
        best = -1
        bestc = -1
        for p in range(V):
            c = cover[p]
            if c > bestc:
                bestc = c
                best = p
            elif c == bestc and c > 0 and rng.random() < 0.5:
                best = p
        if best < 0:
            # no point covers any unhit block -> impossible
            break
        chosen.append(best)
        # remove best from all blocks; update cover
        bm = 1 << best
        for bi in range(len(blocks_mask)):
            if unhit[bi] and (blocks_mask[bi] & bm):
                unhit[bi] = False
        # recompute cover for affected blocks
        for bi in range(len(blocks_mask)):
            if not unhit[bi]:
                continue
            mm = blocks_mask[bi]
            # recompute from scratch (cheap enough at V=25)
        # recompute cover fully
        cover = [0] * V
        for bi in range(len(blocks_mask)):
            if not unhit[bi]:
                continue
            mm = blocks_mask[bi]
            while mm:
                lsb = mm & (-mm)
                p = lsb.bit_length() - 1
                cover[p] += 1
                mm ^= lsb
    return chosen

def construct():
    deadline = time.time() + 110.0
    V = 25
    K = 5

    blocks = _build_steiner_2_5_25()
    if not _verify_steiner(blocks, V, K, 2):
        # Fallback: 10 disjoint triangles
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3 * i, 3 * i + 1, 3 * i + 2
            adj_fallback[u] |= (1 << v) | (1 << w)
            adj_fallback[v] |= (1 << u) | (1 << w)
            adj_fallback[w] |= (1 << u) | (1 << v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}

    # Precompute block masks
    blocks_mask = []
    for b in blocks:
        m = 0
        for x in b:
            m |= (1 << x)
        blocks_mask.append(m)

    # Greedy hitting set to remove points so every block is hit
    hit = _greedy_hit(blocks_mask, V, seed=1)
    removed = set(hit)
    kept = [p for p in range(V) if p not in removed]
    n_kept = len(kept)

    # Map kept points to new labels 0..n_kept-1
    label = {}
    for i, p in enumerate(kept):
        label[p] = i

    # Build dual graph on kept points:
    # adjacency: two kept points adjacent iff they co-occur in some block
    adj = [0] * n_kept
    for b in blocks:
        kept_in_block = [label[x] for x in b if x in label]
        for i in range(len(kept_in_block)):
            for j in range(i + 1, len(kept_in_block)):
                u = kept_in_block[i]
                v = kept_in_block[j]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Verify C4-free (cheap: check common-neighbour counts)
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):
                    return False
        return True

    if not is_c4_free(adj, n_kept):
        # Should not happen for a proper S(2,5,25), but guard
        n_fallback = 30
        adj_fallback = [0] * n_fallback
        for i in range(10):
            u, v, w = 3 * i, 3 * i + 1, 3 * i + 2
            adj_fallback[u] |= (1 << v) | (1 << w)
            adj_fallback[v] |= (1 << u) | (1 << w)
            adj_fallback[w] |= (1 << u) | (1 << v)
        edges = []
        for u in range(n_fallback):
            for v in range(u + 1, n_fallback):
                if adj_fallback[u] & (1 << v):
                    edges.append([u, v])
        return {'n': n_fallback, 'edges': edges}

    # Independent set = set of kept points hitting every block.
    # Check alpha < 11: does there exist an independent set of size 11?
    # Independent set S (of kept points) is independent iff every block
    # is NOT fully contained in S, i.e. every block has at least one point
    # outside S. Equivalently, the set of kept points NOT in S hits every block.
    # So alpha >= 11 iff there exists a set T of kept points of size <= n_kept-11
    # that hits every block (a hitting set of size <= n_kept-11).
    # We already have a hitting set `hit` of size |hit|; if |hit| <= n_kept-11
    # then alpha >= 11. Otherwise need to verify no smaller hitting set exists.
    # Simpler: directly check via the kept-point hitting-set condition.

    # Let's just directly test: is there an independent set of size 11?
    # Independent set of size 11 = 11 kept points with no two co-occurring in a block.
    # Equivalent: the complement (n_kept-11 kept points) is a hitting set.
    # We'll do a bounded search / greedy check.

    def has_independent_11(adj, n, deadline):
        # Try to find an independent set of size 11 via greedy + backtracking
        # Order vertices by degree ascending (low degree easier to include)
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: deg[v])
        # Greedy: try to build independent set
        # Backtrack with pruning
        found = [False]
        def bt(idx, size, forbidden):
            if found[0]:
                return
            if time.time() > deadline:
                return
            if size >= 11:
                found[0] = True
                return
            if idx >= n:
                return
            # upper bound
            if size + (n - idx) < 11:
                return
            v = order[idx]
            if forbidden & (1 << v):
                bt(idx + 1, size, forbidden)
                return
            # Option 1: skip v
            bt(idx + 1, size, forbidden)
            if found[0]:
                return
            # Option 2: include v
            new_forb = forbidden | (1 << v) | adj[v]
            bt(idx + 1, size + 1, new_forb)
        bt(0, 0, 0)
        return found[0]

    if has_independent_11(adj, n_kept, deadline - 15.0):
        # Need to remove more points. Try different seeds / larger hitting set
        best_kept = None
        for seed in range(2, 200):
            if time.time() > deadline - 15.0:
                break
            hit2 = _greedy_hit(blocks_mask, V, seed=seed)
            removed2 = set(hit2)
            kept2 = [p for p in range(V) if p not in removed2]
            n2 = len(kept2)
            if n2 <= n_kept:
                continue
            label2 = {p: i for i, p in enumerate(kept2)}
            adj2 = [0] * n2
            for b in blocks:
                kb = [label2[x] for x in b if x in label2]
                for i in range(len(kb)):
                    for j in range(i + 1, len(kb)):
                        u = kb[i]
                        v = kb[j]
                        adj2[u] |= (1 << v)
                        adj2[v] |= (1 << u)
            if not is_c4_free(adj2, n2):
                continue
            if has_independent_11(adj2, n2, deadline - 10.0):
                continue
            # valid and larger
            if best_kept is None or n2 > best_kept[0]:
                best_kept = (n2, adj2, label2, kept2)
            if time.time() > deadline - 10.0:
                break
        if best_kept is not None:
            n_kept, adj, label, kept = best_kept
        else:
            # Fallback: keep only the hitting set complement minimal, or use 30-triangle
            n_fallback = 30
            adj_fallback = [0] * n_fallback
            for i in range(10):
                u, v, w = 3 * i, 3 * i + 1, 3 * i + 2
                adj_fallback[u] |= (1 << v) | (1 << w)
                adj_fallback[v] |= (1 << u) | (1 << w)
                adj_fallback[w] |= (1 << u) | (1 << v)
            edges = []
            for u in range(n_fallback):
                for v in range(u + 1, n_fallback):
                    if adj_fallback[u] & (1 << v):
                        edges.append([u, v])
            return {'n': n_fallback, 'edges': edges}

    # Build edge list
    edges = []
    for u in range(n_kept):
        for v in range(u + 1, n_kept):
            if adj[u] & (1 << v):
                edges.append([u, v])

    return {'n': n_kept, 'edges': edges}
