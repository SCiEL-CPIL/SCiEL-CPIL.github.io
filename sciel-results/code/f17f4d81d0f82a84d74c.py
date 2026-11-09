import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start = time.time()
    deadline = start + 80

    n = P['n']
    edges = P['edges']

    # Build global adjacency as bitmasks
    adj_mask = [0] * n
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add((u, v))

    total_edges = len(edge_set)
    all_mask = (1 << n) - 1

    def popcount(x):
        return bin(x).count('1')

    def bits_to_list(x):
        lst = []
        i = 0
        while x:
            if x & 1:
                lst.append(i)
            x >>= 1
            i += 1
        return lst

    def set_to_mask(s):
        m = 0
        for v in s:
            m |= (1 << v)
        return m

    def mask_to_set(m):
        s = set()
        i = 0
        while m:
            if m & 1:
                s.add(i)
            m >>= 1
            i += 1
        return s

    def find_max_biclique(rem_adj, rem_edge_count):
        """Find largest biclique using bitmask intersection. Returns (L_mask, R_mask) or None."""
        best_L = 0
        best_R = 0
        best_size = 0

        # For each vertex u, consider it as part of L
        # R candidates: neighbors of u in remaining graph
        # Then expand: L = {v : N(v) ⊇ R}, R = {v : N(v) ⊇ L}
        for u in range(n):
            R_cand = rem_adj[u]
            if R_cand == 0:
                continue

            L = 1 << u
            R = R_cand

            # Iterate to fixpoint
            for _ in range(20):  # bounded iterations
                # Expand L: vertices connected to ALL of R
                common = all_mask
                r = R
                while r:
                    low = r & (-r)
                    idx = low.bit_length() - 1
                    common &= rem_adj[idx]
                    r &= r - 1

                new_L = common & ~(L | R)
                if new_L:
                    L |= new_L

                # Expand R: vertices connected to ALL of L
                common = all_mask
                l = L
                while l:
                    low = l & (-l)
                    idx = low.bit_length() - 1
                    common &= rem_adj[idx]
                    l &= l - 1

                new_R = common & ~(L | R)
                if new_R:
                    R |= new_R

                if not new_L and not new_R:
                    break

            if not L or not R:
                continue
            if L & R:
                continue

            size = popcount(L) * popcount(R)
            if size > best_size:
                best_size = size
                best_L = L
                best_R = R

        if best_size == 0:
            return None
        return (best_L, best_R)

    def greedy_partition(edge_set, rng, use_weights=False):
        """Greedy: repeatedly find largest biclique in remaining edges."""
        bicliques = []
        remaining = set(edge_set)

        while remaining:
            if time.time() > deadline:
                break

            # Build remaining adjacency
            rem_adj = [0] * n
            for (u, v) in remaining:
                rem_adj[u] |= (1 << v)
                rem_adj[v] |= (1 << u)

            bc = find_max_biclique(rem_adj, len(remaining))

            if bc is None:
                # Fallback: single edges
                for (u, v) in remaining:
                    bicliques.append((1 << u, 1 << v))
                remaining = set()
                break

            L, R = bc
            bicliques.append((L, R))

            # Remove all edges between L and R
            to_remove = set()
            l = L
            while l:
                low = l & (-l)
                u = low.bit_length() - 1
                r = R
                while r:
                    low_r = r & (-r)
                    v = low_r.bit_length() - 1
                    if u != v:
                        e = (min(u, v), max(u, v))
                        if e in remaining:
                            to_remove.add(e)
                    r &= r - 1
                l &= l - 1
            remaining -= to_remove

        return bicliques

    def try_merge_all(bicliques, edge_set, adj_mask):
        """Repeatedly try to merge pairs of bicliques until no more merges possible."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_bics = len(bicliques)
            for i in range(n_bics):
                if time.time() > deadline:
                    return bicliques
                for j in range(i + 1, n_bics):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]

                    candidates = []

                    # Union both sides
                    newL = L1 | L2
                    newR = R1 | R2
                    if not (newL & newR) and newL and newR:
                        candidates.append((newL, newR))

                    # Cross merge
                    newL = L1 | R2
                    newR = R1 | L2
                    if not (newL & newR) and newL and newR:
                        candidates.append((newL, newR))

                    for newL, newR in candidates:
                        # Verify all cross edges exist in original graph
                        valid = True
                        l = newL
                        while l and valid:
                            low = l & (-l)
                            u = low.bit_length() - 1
                            r = newR
                            while r and valid:
                                low_r = r & (-r)
                                v = low_r.bit_length() - 1
                                if u != v:
                                    if not (adj_mask[u] & (1 << v)):
                                        valid = False
                                r &= r - 1
                            l &= l - 1

                        if valid:
                            result = list(bicliques)
                            result.pop(j)
                            result.pop(i)
                            result.append((newL, newR))
                            return result

        return bicliques

    # Run multiple greedy strategies
    best_result = None
    best_count = total_edges + 1

    rng = random.Random(12345)

    for restart in range(40):
        if time.time() > deadline:
            break

        bicliques = greedy_partition(edge_set, rng)
        bicliques = try_merge_all(bicliques, edge_set, adj_mask)

        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_result = bicliques

        if best_count <= 10:
            break

    # Final merge pass on best result
    if best_result is not None:
        best_result = try_merge_all(best_result, edge_set, adj_mask)

    # Convert to required format
    result = []
    for L, R in best_result:
        result.append([bits_to_list(L), bits_to_list(R)])

    # Validate
    covered_edges = set()
    valid = True
    for bc in result:
        if not isinstance(bc, list) or len(bc) != 2:
            valid = False
            break
        left, right = bc
        if not isinstance(left, list) or not isinstance(right, list):
            valid = False
            break
        if len(left) == 0 or len(right) == 0:
            valid = False
            break
        if set(left) & set(right):
            valid = False
            break
        for l in left:
            if not isinstance(l, int) or isinstance(l, bool):
                valid = False
                break
            for r in right:
                if not isinstance(r, int) or isinstance(r, bool):
                    valid = False
                    break
                if l == r:
                    valid = False
                    break
                e = (min(l, r), max(l, r))
                if e not in edge_set:
                    valid = False
                    break
                if e in covered_edges:
                    valid = False
                    break
                covered_edges.add(e)
        if not valid:
            break

    if not valid or covered_edges != edge_set:
        # Fallback: one biclique per vertex
        result = []
        for u in range(n):
            neighbors = [v for v in range(n) if adj_mask[u] & (1 << v)]
            if neighbors:
                result.append([[u], neighbors])

    return result
