"""Greedy maximal biclique partition of a random graph."""
import time

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 50.0

    # Build adjacency bitsets (undirected)
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Remaining edges as a set of (min, max) tuples
    rem_edges = set()
    for u, v in edges:
        if u == v:
            continue
        rem_edges.add((min(u, v), max(u, v)))

    # Adjacency restricted to remaining edges
    rem_adj = [0] * n
    for (a, b) in rem_edges:
        rem_adj[a] |= (1 << b)
        rem_adj[b] |= (1 << a)

    def popcount(x):
        return bin(x).count('1')

    def iter_bits(x):
        while x:
            lsb = x & (-x)
            yield lsb.bit_length() - 1
            x ^= lsb

    def grow_biclique(seed_l, seed_r):
        """Grow a maximal biclique containing seed_l and seed_r within remaining edges."""
        Lset = {seed_l}
        Rset = set(iter_bits(rem_adj[seed_l]))
        if seed_r not in Rset:
            return [], []
        changed = True
        while changed:
            changed = False
            # Common neighborhood of current L in remaining graph
            common = None
            for l in Lset:
                nb = rem_adj[l]
                if common is None:
                    common = nb
                else:
                    common &= nb
            new_R = set(iter_bits(common))
            if new_R != Rset:
                Rset = new_R
                changed = True
            # Now L can be any vertex adjacent to all of Rset
            if Rset:
                commonL = None
                for r in Rset:
                    nb = rem_adj[r]
                    if commonL is None:
                        commonL = nb
                    else:
                        commonL &= nb
                new_L = set(iter_bits(commonL))
                if new_L != Lset:
                    Lset = new_L
                    changed = True
            else:
                break
        return list(Lset), list(Rset)

    bicliques = []

    while rem_edges:
        if time.time() > deadline:
            break

        # Find best seed: try each remaining edge, grow, pick max coverage
        best = None
        best_count = -1
        seed_list = list(rem_edges)

        for (a, b) in seed_list:
            if time.time() > deadline:
                break
            # Try both orientations
            for (sl, sr) in ((a, b), (b, a)):
                L, R = grow_biclique(sl, sr)
                if not L or not R:
                    continue
                # Count remaining edges covered by this biclique using bitmasks
                cnt = 0
                for l in L:
                    mask = 0
                    for r in R:
                        if l != r:
                            mask |= (1 << r)
                    common_with_rem = mask & rem_adj[l]
                    # Only count edges that are in rem_edges (already handled by rem_adj)
                    # But we need to avoid double counting; each edge (l,r) counted once per l
                    # Since we iterate over L and for each l count neighbors in R that are in rem_adj[l],
                    # and rem_adj is symmetric, each edge is counted exactly once (when l is the smaller? No...)
                    # Actually, for a biclique [L,R], the edges are all (l,r) for l in L, r in R.
                    # We count for each l in L, the number of r in R such that (l,r) is in rem_adj[l].
                    # This counts each cross-edge exactly once (once per left vertex).
                    cnt += popcount(common_with_rem)
                if cnt > best_count:
                    best_count = cnt
                    best = (L, R)
            # Early stop heuristic
            if best_count >= len(rem_edges):
                break

        if best is None or best_count <= 0:
            # Fallback: single edge
            (a, b) = next(iter(rem_edges))
            best = ([a], [b])
            best_count = 1

        L, R = best
        # Remove covered remaining edges
        for l in L:
            for r in R:
                if l == r:
                    continue
                key = (min(l, r), max(l, r))
                if key in rem_edges:
                    rem_edges.discard(key)
                    rem_adj[l] &= ~(1 << r)
                    rem_adj[r] &= ~(1 << l)
        bicliques.append([sorted(L), sorted(R)])

    # Fallback: if deadline broke early, cover remaining edges one by one
    if rem_edges:
        for (a, b) in list(rem_edges):
            bicliques.append([[a], [b]])
            rem_edges.discard((a, b))

    # Ensure types: lists of ints
    out = []
    for L, R in bicliques:
        out.append([[int(x) for x in L], [int(y) for y in R]])
    return out
# EVOLVE-BLOCK-END
