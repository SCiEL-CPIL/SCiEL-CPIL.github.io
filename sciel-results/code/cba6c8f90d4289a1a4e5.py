# EVOLVE-BLOCK-START
import time

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Full edge set for validation
    all_edges = set()
    for u, v in P['edges']:
        all_edges.add((min(u, v), max(u, v)))

    remaining_edges = set(all_edges)
    bicliques = []

    def expand_biclique(u, v, rem_edges):
        """Expand from seed edge (u,v) to maximal biclique in remaining subgraph."""
        L = {u}
        R = {v}
        changed = True
        while changed:
            changed = False
            # Expand L
            candidate_L = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for r in R:
                    key = (min(w, r), max(w, r))
                    if key not in rem_edges:
                        ok = False
                        break
                if ok:
                    candidate_L.add(w)
            if candidate_L:
                L |= candidate_L
                changed = True
            # Expand R
            candidate_R = set()
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    key = (min(w, l), max(w, l))
                    if key not in rem_edges:
                        ok = False
                        break
                if ok:
                    candidate_R.add(w)
            if candidate_R:
                R |= candidate_R
                changed = True
        R = R - L
        return L, R

    while remaining_edges:
        if time.time() > deadline:
            break

        # Build adjacency for remaining subgraph
        rem_adj = [set() for _ in range(n)]
        for a, b in remaining_edges:
            rem_adj[a].add(b)
            rem_adj[b].add(a)

        # Score edges: use product of degrees as a proxy for biclique size
        scored_edges = []
        for a, b in remaining_edges:
            da = len(rem_adj[a])
            db = len(rem_adj[b])
            common = len(rem_adj[a] & rem_adj[b])
            # Heuristic: prefer edges in dense regions
            score = da * db + 10 * common
            scored_edges.append((score, a, b))
        scored_edges.sort(reverse=True)

        # Try top K seed edges
        best_biclique = None
        best_count = 0
        K = min(50, len(scored_edges))

        for idx in range(K):
            if time.time() > deadline:
                break
            _, u, v = scored_edges[idx]
            L, R = expand_biclique(u, v, remaining_edges)
            if not L or not R:
                continue
            count = len(L) * len(R)
            if count > best_count:
                best_count = count
                best_biclique = (L, R)

        if best_biclique is None:
            e = next(iter(remaining_edges))
            best_biclique = ({e[0]}, {e[1]})

        L, R = best_biclique

        for u in L:
            for v in R:
                key = (min(u, v), max(u, v))
                remaining_edges.discard(key)

        bicliques.append([sorted(int(x) for x in L), sorted(int(x) for x in R)])

    if remaining_edges:
        for e in remaining_edges:
            bicliques.append([[int(e[0])], [int(e[1])]])

    # Post-hoc: try to merge pairs of bicliques
    # Two bicliques [L1,R1] and [L2,R2] can be merged if:
    # Option A: L1∪L2 and R1∪R2 form a valid biclique (all cross edges exist in graph)
    # Option B: L1 and R1∪R2 form a valid biclique, and L2 is subset of L1 or vice versa
    # Simplest check: try merging by combining lefts and rights
    def can_merge(L1, R1, L2, R2):
        """Check if two bicliques can be merged into one valid biclique."""
        # Try: merged L = L1 ∪ L2, merged R = R1 ∪ R2
        # All cross pairs must be in all_edges
        for u in L1:
            for v in R1:
                if (min(u,v), max(u,v)) not in all_edges:
                    return None
            for v in R2:
                if (min(u,v), max(u,v)) not in all_edges:
                    return None
        for u in L2:
            for v in R2:
                if (min(u,v), max(u,v)) not in all_edges:
                    return None
            for v in R1:
                if (min(u,v), max(u,v)) not in all_edges:
                    return None
        # Also need L and R to be disjoint
        merged_L = set(L1) | set(L2)
        merged_R = set(R1) | set(R2)
        if merged_L & merged_R:
            return None
        if not merged_L or not merged_R:
            return None
        return (merged_L, merged_R)

    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(bicliques):
            j = i + 1
            while j < len(bicliques):
                L1, R1 = set(bicliques[i][0]), set(bicliques[i][1])
                L2, R2 = set(bicliques[j][0]), set(bicliques[j][1])
                result = can_merge(L1, R1, L2, R2)
                if result is not None:
                    mL, mR = result
                    bicliques[i] = [sorted(int(x) for x in mL), sorted(int(x) for x in mR)]
                    bicliques.pop(j)
                    merged = True
                    break
                j += 1
            if merged:
                break
            i += 1

    return bicliques
# EVOLVE-BLOCK-END
