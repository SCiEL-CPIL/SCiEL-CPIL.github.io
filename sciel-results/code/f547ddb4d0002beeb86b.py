import time

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((a, b))

    total_edges = len(edge_set)

    def mk(u, v):
        return (u, v) if u < v else (v, u)

    def expand_biclique(L, R, remaining):
        """Expand L and R to maximal biclique within remaining edges."""
        L, R = set(L), set(R)
        changed = True
        while changed:
            changed = False
            newL = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = all(mk(w, r) in remaining for r in R)
                if ok:
                    newL.add(w)
            if newL:
                L |= newL
                changed = True
            newR = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = all(mk(w, l) in remaining for l in L)
                if ok:
                    newR.add(w)
            if newR:
                R |= newR
                changed = True
        return L, R

    def is_valid_biclique(L, R, remaining):
        for l in L:
            for r in R:
                if l == r or mk(l, r) not in remaining:
                    return False
        return True

    def biclique_edges(L, R):
        return {mk(l, r) for l in L for r in R if l != r}

    def find_best_biclique(remaining):
        """Find the best biclique to extract from remaining edges."""
        if not remaining:
            return None

        best = None
        best_score = -1

        # Strategy 1: Edge-seed expansion
        for (u, v) in remaining:
            L, R = expand_biclique({u}, {v}, remaining)
            if not is_valid_biclique(L, R, remaining):
                continue
            covered = biclique_edges(L, R)
            score = len(covered)
            if score > best_score:
                best_score = score
                best = (L, R)

        # Strategy 2: Vertex-seed with common neighbors
        for u in range(n):
            nbrs = {v for v in adj[u] if mk(u, v) in remaining}
            if not nbrs:
                continue
            # Try with all neighbors
            L, R = expand_biclique({u}, nbrs, remaining)
            if is_valid_biclique(L, R, remaining):
                covered = biclique_edges(L, R)
                score = len(covered)
                if score > best_score:
                    best_score = score
                    best = (L, R)

            # Try with subsets of neighbors (greedy: keep only those that form a biclique)
            # Start with full neighbor set and trim
            R_trim = set(nbrs)
            # Find maximal subset of nbrs that forms a biclique with {u}
            # A vertex v in nbrs can be in R if it connects to all other R vertices
            changed = True
            while changed:
                changed = False
                for v in list(R_trim):
                    # v must connect to all other vertices in R_trim
                    ok = all(mk(v, w) in remaining for w in R_trim if w != v)
                    if not ok:
                        R_trim.discard(v)
                        changed = True
            if R_trim:
                L2, R2 = expand_biclique({u}, R_trim, remaining)
                if is_valid_biclique(L2, R2, remaining):
                    covered = biclique_edges(L2, R2)
                    score = len(covered)
                    if score > best_score:
                        best_score = score
                        best = (L2, R2)

        return best

    def greedy_partition(remaining, deadline):
        """Greedy: repeatedly extract best biclique."""
        bicliques = []
        rem = set(remaining)
        while rem:
            if time.time() > deadline:
                break
            b = find_best_biclique(rem)
            if b is None:
                u, v = next(iter(rem))
                bicliques.append([u, v])
                rem.discard(mk(u, v))
                continue
            L, R = b
            bicliques.append([list(L), list(R)])
            for l in L:
                for r in R:
                    rem.discard(mk(l, r))
        # Handle any leftover edges
        if rem:
            for e in rem:
                u, v = e
                bicliques.append([u, v])
        return bicliques

    def compute_covered_edges(bicliques):
        covered = set()
        for L, R in bicliques:
            for l in L:
                for r in R:
                    covered.add(mk(l, r))
        return covered

    def try_local_moves(bicliques, deadline):
        """Try to reduce biclique count via moves and merges."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Try merging pairs
            for i in range(len(bicliques)):
                if time.time() > deadline:
                    return
                for j in range(i + 1, len(bicliques)):
                    if time.time() > deadline:
                        return
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]

                    # Try all 4 merge orientations
                    candidates = [
                        (set(L1) | set(L2), set(R1) | set(R2)),
                        (set(L1) | set(R2), set(R1) | set(L2)),
                        (set(L1) | set(L2), set(R1) | set(R2)),
                        (set(R1) | set(R2), set(L1) | set(L2)),
                    ]
                    seen = set()
                    for Lnew, Rnew in candidates:
                        key = (frozenset(Lnew), frozenset(Rnew))
                        if key in seen or not Lnew or not Rnew:
                            continue
                        seen.add(key)
                        # Check biclique validity
                        ok = True
                        for a in Lnew:
                            for b in Rnew:
                                if a == b or mk(a, b) not in edge_set:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # Check that merged edges == original edges
                        merged_e = biclique_edges(Lnew, Rnew)
                        orig_e = biclique_edges(L1, R1) | biclique_edges(L2, R2)
                        if merged_e == orig_e:
                            bicliques[i] = [sorted(Lnew), sorted(Rnew)]
                            bicliques.pop(j)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Try moving a vertex from one biclique to another
            if not improved:
                for i in range(len(bicliques)):
                    if time.time() > deadline:
                        return
                    for side in (0, 1):
                        L, R = bicliques[i]
                        source = L if side == 0 else R
                        target = R if side == 0 else L
                        for v in list(source):
                            # Try moving v from source to target
                            new_source = [x for x in source if x != v]
                            new_target = list(target) + [v]
                            if not new_source or not new_target:
                                continue
                            # Check biclique validity
                            ok = True
                            for a in new_source:
                                for b in new_target:
                                    if a == b or mk(a, b) not in edge_set:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if not ok:
                                continue
                            # Check edge preservation
                            old_e = biclique_edges(source, target)
                            new_e = biclique_edges(new_source, new_target)
                            if new_e != old_e:
                                continue
                            # Apply
                            if side == 0:
                                bicliques[i] = [new_source, new_target]
                            else:
                                bicliques[i] = [new_target, new_source]
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break

            # Try moving a vertex from biclique i to biclique j
            if not improved and len(bicliques) >= 2:
                for i in range(len(bicliques)):
                    if time.time() > deadline:
                        return
                    for j in range(len(bicliques)):
                        if i == j or time.time() > deadline:
                            continue
                        Li, Ri = bicliques[i]
                        Lj, Rj = bicliques[j]
                        # Try moving v from Li to Lj, or from Li to Rj, etc.
                        for src_side, dst_bic, dst_side in [(0, j, 0), (0, j, 1), (1, j, 0), (1, j, 1)]:
                            if time.time() > deadline:
                                return
                            src = Li if src_side == 0 else Ri
                            dst = Lj if dst_side == 0 else Rj
                            for v in list(src):
                                # v must not already be in dst
                                if v in dst:
                                    continue
                                new_src = [x for x in src if x != v]
                                new_dst = list(dst) + [v]
                                if not new_src or not new_dst:
                                    continue
                                # Check biclique i still valid
                                other_i = Ri if src_side == 0 else Li
                                ok_i = True
                                for a in new_src:
                                    for b in other_i:
                                        if a == b or mk(a, b) not in edge_set:
                                            ok_i = False
                                            break
                                    if not ok_i:
                                        break
                                if not ok_i:
                                    continue
                                # Check biclique j still valid
                                other_j = Rj if dst_side == 0 else Lj
                                ok_j = True
                                for a in new_dst:
                                    for b in other_j:
                                        if a == b or mk(a, b) not in edge_set:
                                            ok_j = False
                                            break
                                    if not ok_j:
                                        break
                                if not ok_j:
                                    continue
                                # Check edge preservation for both bicliques
                                old_ei = biclique_edges(src, other_i)
                                new_ei = biclique_edges(new_src, other_i)
                                old_ej = biclique_edges(dst, other_j)
                                new_ej = biclique_edges(new_dst, other_j)
                                if new_ei != old_ei or new_ej != old_ej:
                                    continue
                                # Apply
                                if src_side == 0:
                                    bicliques[i] = [new_src, other_i]
                                else:
                                    bicliques[i] = [other_i, new_src]
                                if dst_side == 0:
                                    bicliques[j] = [new_dst, other_j]
                                else:
                                    bicliques[j] = [other_j, new_dst]
                                improved = True
                                break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

    def reseed_partition(remaining, deadline):
        """Alternative seeding: pick edges in different order."""
        bicliques = []
        rem = set(remaining)
        while rem:
            if time.time() > deadline:
                break
            # Pick edge with highest degree sum
            best_e = None
            best_deg = -1
            for e in rem:
                u, v = e
                deg = len(adj[u]) + len(adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_e = e
            if best_e is None:
                break
            u, v = best_e
            L, R = expand_biclique({u}, {v}, rem)
            if not is_valid_biclique(L, R, rem) or not L or not R:
                bicliques.append([u, v])
                rem.discard(mk(u, v))
                continue
            bicliques.append([list(L), list(R)])
            for l in L:
                for r in R:
                    rem.discard(mk(l, r))
        if rem:
            for e in rem:
                u, v = e
                bicliques.append([u, v])
        return bicliques

    # Main algorithm
    remaining = set(edge_set)

    # Initial greedy partition
    bicliques = greedy_partition(remaining, deadline - 10)

    # Local refinement
    try_local_moves(bicliques, deadline - 5)

    # Try alternative seeding and see if it's better
    if time.time() < deadline - 8 and len(bicliques) > 2:
        alt = reseed_partition(set(edge_set), deadline - 8)
        try_local_moves(alt, deadline - 5)
        if len(alt) < len(bicliques):
            bicliques = alt

    # More local refinement passes
    for _ in range(2):
        if time.time() > deadline - 4:
            break
        try_local_moves(bicliques, deadline - 4)

    # Final safety: ensure full coverage
    covered = compute_covered_edges(bicliques)
    for e in edge_set:
        if e not in covered:
            u, v = e
            bicliques.append([u, v])

    # Convert to exact Python types
    result = []
    for L, R in bicliques:
        Lc = [int(x) for x in L]
        Rc = [int(x) for x in R]
        if Lc and Rc:
            result.append([Lc, Rc])
    return result
# EVOLVE-BLOCK-END
