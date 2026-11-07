import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    n = P['n']
    edges = P['edges']
    total_edges = len(edges)

    # Build adjacency mask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Seed Solution (Baseline)
    best_bicliques = []
    for u in range(n):
        right_side = [v for a, v in edges if a == u]
        if right_side:
            best_bicliques.append([[u], right_side])
    best_count = len(best_bicliques)

    def popcount_int(x):
        c = 0
        while x:
            c += bin(x & 0xFFFF).count('1')
            x >>= 16
        return c

    def count_edges(rem_adj):
        s = 0
        for i in range(n):
            s += popcount_int(rem_adj[i])
        return s // 2

    def expand_biclique(rem_adj, u, v, mode='greedy'):
        """Expand a biclique starting from edge (u,v) in rem_adj."""
        L = {u}
        R = {v}
        if mode == 'maximal':
            # Compute true maximal biclique containing (u,v)
            # L candidates: all w connected to every vertex in R
            # R candidates: all w connected to every vertex in L
            mask_R = (1 << u) | (1 << v)  # placeholder, we'll rebuild
            # Actually start with L={u}, R={v}
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                mask_R = 0
                for rv in R:
                    mask_R |= (1 << rv)
                new_L = set()
                for w in range(n):
                    if w not in L and (rem_adj[w] & mask_R) == mask_R:
                        new_L.add(w)
                if new_L:
                    L |= new_L
                    changed = True
                mask_L = 0
                for lu in L:
                    mask_L |= (1 << lu)
                new_R = set()
                for w in range(n):
                    if w not in R and (rem_adj[w] & mask_L) == mask_L:
                        new_R.add(w)
                if new_R:
                    R |= new_R
                    changed = True
        else:
            # Greedy: add one candidate at a time (randomized order)
            changed = True
            for _ in range(n):
                if not changed:
                    break
                changed = False
                mask_R = 0
                for rv in R:
                    mask_R |= (1 << rv)
                candidates_L = []
                for w in range(n):
                    if w not in L and (rem_adj[w] & mask_R) == mask_R:
                        candidates_L.append(w)
                if candidates_L:
                    random.shuffle(candidates_L)
                    for w in candidates_L:
                        L.add(w)
                        changed = True
                mask_L = 0
                for lu in L:
                    mask_L |= (1 << lu)
                candidates_R = []
                for w in range(n):
                    if w not in R:
                        valid = True
                        for lu in L:
                            if not (rem_adj[lu] & (1 << w)):
                                valid = False
                                break
                        if valid:
                            candidates_R.append(w)
                if candidates_R:
                    random.shuffle(candidates_R)
                    for w in candidates_R:
                        R.add(w)
                        changed = True
        return L, R

    def try_greedy(restarts=400):
        nonlocal best_bicliques, best_count
        for r in range(restarts):
            if time.time() > deadline:
                return
            rem_adj = adj[:]
            current_bicliques = []
            edges_left = total_edges

            # Choose strategy for this restart
            use_maximal = (r % 3 == 0)
            use_max_degree = (r % 3 == 1)

            while edges_left > 0:
                if time.time() > deadline:
                    return
                found_edge = False
                u = -1
                v = -1

                if use_max_degree:
                    # Pick edge incident to highest-degree vertex
                    best_deg = -1
                    for i in range(n):
                        deg = popcount_int(rem_adj[i])
                        if deg > best_deg and rem_adj[i]:
                            best_deg = deg
                            u = i
                    if u >= 0:
                        m = rem_adj[u]
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        found_edge = True
                else:
                    # Random edge
                    for _ in range(50):
                        u = random.randint(0, n - 1)
                        mask = rem_adj[u]
                        if mask:
                            m = mask
                            candidates = []
                            while m:
                                lsb = m & -m
                                v_cand = lsb.bit_length() - 1
                                candidates.append(v_cand)
                                m ^= lsb
                            v = random.choice(candidates)
                            if rem_adj[u] & (1 << v):
                                found_edge = True
                                break
                    if not found_edge:
                        for i in range(n):
                            if rem_adj[i]:
                                u = i
                                m = rem_adj[i]
                                lsb = m & -m
                                v = lsb.bit_length() - 1
                                found_edge = True
                                break
                if not found_edge:
                    break

                mode = 'maximal' if use_maximal else 'greedy'
                L, R = expand_biclique(rem_adj, u, v, mode)

                current_bicliques.append([list(L), list(R)])
                mask_L = 0
                for lu in L:
                    mask_L |= (1 << lu)
                mask_R = 0
                for rv in R:
                    mask_R |= (1 << rv)
                for lu in L:
                    rem_adj[lu] &= ~mask_R
                for rv in R:
                    rem_adj[rv] &= ~mask_L
                edges_left -= len(L) * len(R)
                if edges_left < 0:
                    break

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = [[list(l), list(r)] for l, r in current_bicliques]
                if best_count == 1:
                    return

    def can_merge(bi, bj):
        Li, Ri = bi
        Lj, Rj = bj
        if set(Li) & set(Lj) or set(Ri) & set(Rj):
            return False
        if set(Li) & set(Rj) or set(Lj) & set(Ri):
            return False
        for u in Li:
            for v in Ri:
                if not (adj[u] & (1 << v)):
                    return False
        for u in Li:
            for v in Rj:
                if not (adj[u] & (1 << v)):
                    return False
        for u in Lj:
            for v in Ri:
                if not (adj[u] & (1 << v)):
                    return False
        for u in Lj:
            for v in Rj:
                if not (adj[u] & (1 << v)):
                    return False
        return True

    def is_valid_biclique(L, R, adj_mask):
        """Check if (L, R) forms a valid biclique in adj_mask."""
        for u in L:
            for v in R:
                if not (adj_mask[u] & (1 << v)):
                    return False
        return True

    def try_move_vertex(bl):
        """Try to move a vertex from one biclique to another to enable merges."""
        if time.time() > deadline:
            return False
        nb = len(bl)
        for i in range(nb):
            if time.time() > deadline:
                return False
            # Try moving a vertex from bl[i] to bl[j]
            for j in range(nb):
                if i == j:
                    continue
                Li, Ri = bl[i]
                Lj, Rj = bl[j]
                # Try moving a vertex from Li to Lj
                for idx, vert in enumerate(Li):
                    new_Li = Li[:idx] + Li[idx+1:]
                    new_Lj = Lj + [vert]
                    if not new_Li or not new_Lj:
                        continue
                    # Check new biclique i is valid
                    if not is_valid_biclique(new_Li, Ri, adj):
                        continue
                    # Check new biclique j is valid
                    if not is_valid_biclique(new_Lj, Rj, adj):
                        continue
                    # Check no vertex appears in both sides of any biclique
                    # (within-side is fine, but cross-side within same biclique is not)
                    if set(new_Li) & set(Ri) or set(new_Lj) & set(Rj):
                        continue
                    # Check no vertex is in both L and R across bicliques
                    # Actually the constraint is: within each biclique, L and R are disjoint
                    # And all cross pairs must be edges. Let's just check validity.
                    bl[i] = [new_Li, Ri]
                    bl[j] = [new_Lj, Rj]
                    return True
                # Try moving a vertex from Ri to Rj
                for idx, vert in enumerate(Ri):
                    new_Ri = Ri[:idx] + Ri[idx+1:]
                    new_Rj = Rj + [vert]
                    if not new_Ri or not new_Rj:
                        continue
                    if not is_valid_biclique(Li, new_Ri, adj):
                        continue
                    if not is_valid_biclique(Lj, new_Rj, adj):
                        continue
                    if set(Li) & set(new_Ri) or set(Lj) & set(new_Rj):
                        continue
                    bl[i] = [Li, new_Ri]
                    bl[j] = [Lj, new_Rj]
                    return True
        return False

    def local_search_refine(bicliques, max_iters=300):
        if time.time() > deadline:
            return bicliques
        bl = [[list(b[0]), list(b[1])] for b in bicliques]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline:
                return bl
            # Try merges
            for i in range(len(bl)):
                if time.time() > deadline:
                    return bl
                for j in range(i + 1, len(bl)):
                    if time.time() > deadline:
                        return bl
                    if can_merge(bl[i], bl[j]):
                        bl[i] = [bl[i][0] + bl[j][0], bl[i][1] + bl[j][1]]
                        del bl[j]
                        improved = True
                        break
                if improved:
                    break
            # If no merge, try vertex moves
            if not improved:
                if try_move_vertex(bl):
                    improved = True
        return bl

    try_greedy()

    if best_bicliques:
        refined = local_search_refine(best_bicliques, max_iters=500)
        if len(refined) < best_count:
            best_count = len(refined)
            best_bicliques = refined

    if not best_bicliques:
        for u in range(n):
            right_side = [v for a, v in edges if a == u]
            if right_side:
                best_bicliques.append([[u], right_side])

    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    return result
