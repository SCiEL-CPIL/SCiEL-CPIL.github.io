import time
import random
import itertools

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    # Build adjacency bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))

    total_edges = len(edge_set)

    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def expand_biclique(L_mask, R_mask, rem_adj):
        """Expand a partial biclique to maximal within the remaining edge set."""
        changed = True
        while changed:
            changed = False
            # Expand L: find u not in L such that u connected to all in R
            for u in range(n):
                if (L_mask >> u) & 1:
                    continue
                if R_mask != 0 and (rem_adj[u] & R_mask) == R_mask:
                    L_mask |= (1 << u)
                    changed = True
            # Expand R: find v not in R such that v connected to all in L
            for v in range(n):
                if (R_mask >> v) & 1:
                    continue
                if L_mask != 0 and (rem_adj[v] & L_mask) == L_mask:
                    R_mask |= (1 << v)
                    changed = True
        return L_mask, R_mask

    def is_biclique(L, R):
        """Check if all pairs in L x R are edges in the original graph."""
        for u in L:
            for v in R:
                if not (adj[u] >> v) & 1:
                    return False
        return True

    def biclique_edges(L, R):
        """Get the set of canonical edges in L x R."""
        s = set()
        for u in L:
            for v in R:
                if u < v:
                    s.add((u, v))
                else:
                    s.add((v, u))
        return s

    def merge_pass(bicliques):
        """Try to merge pairs of bicliques if their union forms a valid biclique."""
        changed = True
        while changed:
            changed = False
            new_bicliques = []
            i = 0
            while i < len(bicliques):
                B1 = bicliques[i]
                merged = False
                for j in range(i + 1, len(bicliques)):
                    if bicliques[j] is None:
                        continue
                    B2 = bicliques[j]
                    L1, R1 = B1
                    L2, R2 = B2

                    L_new = list(set(L1) | set(L2))
                    R_new = list(set(R1) | set(R2))

                    # Check if L_new x R_new is a biclique in G
                    if not is_biclique(L_new, R_new):
                        continue

                    # Check that the union of edges equals the new biclique's edges
                    E1 = biclique_edges(L1, R1)
                    E2 = biclique_edges(L2, R2)
                    E_new = biclique_edges(L_new, R_new)

                    if E1 | E2 == E_new:
                        # Valid merge
                        bicliques[j] = None
                        B1 = [L_new, R_new]
                        merged = True
                        changed = True
                        break

                if B1 is not None:
                    new_bicliques.append(B1)
                i += 1
            bicliques = [b for b in new_bicliques if b is not None]

        return bicliques

    def get_random_edge(rem_adj):
        """Pick a random edge from the remaining edge set."""
        candidates = [u for u in range(n) if rem_adj[u] != 0]
        if not candidates:
            return None
        u = random.choice(candidates)
        mask = rem_adj[u]
        vs = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vs.append(v)
            m ^= lsb
        if not vs:
            return None
        v = random.choice(vs)
        return (u, v)

    def greedy_cover(rem_adj, max_restarts=50):
        """Greedy cover using bitmask expansion. Returns list of bicliques as [L_list, R_list]."""
        best_bicliques = None
        best_count = float('inf')

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            current_bicliques = []
            cur_adj = rem_adj[:]
            edges_left = 0
            for u in range(n):
                edges_left += cur_adj[u].bit_count()
            edges_left //= 2  # each edge counted twice

            while edges_left > 0:
                if time.time() > deadline - 5:
                    return None

                best_local = None
                best_size = 0

                # Try multiple random starting edges, pick the one with largest expansion
                for _ in range(8):
                    if time.time() > deadline - 5:
                        return None
                    edge = get_random_edge(cur_adj)
                    if edge is None:
                        break
                    u, v = edge
                    L_mask = (1 << u)
                    R_mask = (1 << v)
                    L_mask, R_mask = expand_biclique(L_mask, R_mask, cur_adj)
                    size = L_mask.bit_count() * R_mask.bit_count()
                    if size > best_size:
                        best_size = size
                        best_local = (L_mask, R_mask)

                if not best_local:
                    edge = get_random_edge(cur_adj)
                    if edge is None:
                        break
                    u, v = edge
                    best_local = ((1 << u), (1 << v))
                    best_size = 1

                L_mask, R_mask = best_local
                current_bicliques.append([mask_to_list(L_mask), mask_to_list(R_mask)])

                # Update remaining adjacency
                l_list = mask_to_list(L_mask)
                r_list = mask_to_list(R_mask)
                for u in l_list:
                    cur_adj[u] &= ~R_mask
                for v in r_list:
                    cur_adj[v] &= ~L_mask

                edges_left -= best_size
                if edges_left < 0:
                    edges_left = 0

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    break

        return best_bicliques

    def enumerate_bicliques(adj, max_size=5):
        """Enumerate all bicliques up to a certain size using edge-based enumeration."""
        bicliques = set()
        edges_list = list(edge_set)
        
        # For each edge, try to extend it to a maximal biclique
        for u, v in edges_list:
            L_mask = (1 << u)
            R_mask = (1 << v)
            L_mask, R_mask = expand_biclique(L_mask, R_mask, adj)
            L_list = mask_to_list(L_mask)
            R_list = mask_to_list(R_mask)
            bicliques.add((frozenset(L_list), frozenset(R_list)))
        
        return bicliques

    def greedy_with_candidates(candidates, rem_adj, max_restarts=30):
        """Greedy cover but only using bicliques from the candidate set."""
        best_bicliques = None
        best_count = float('inf')

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            current_bicliques = []
            cur_adj = rem_adj[:]
            edges_left = 0
            for u in range(n):
                edges_left += cur_adj[u].bit_count()
            edges_left //= 2

            while edges_left > 0:
                if time.time() > deadline - 5:
                    return None

                best_local = None
                best_size = 0

                # Find the best candidate biclique that covers remaining edges
                for L_set, R_set in candidates:
                    L_mask = 0
                    R_mask = 0
                    for u in L_set:
                        L_mask |= (1 << u)
                    for v in R_set:
                        R_mask |= (1 << v)
                    
                    # Check if this biclique covers any remaining edges
                    covered = 0
                    for u in L_set:
                        for v in R_set:
                            if (cur_adj[u] >> v) & 1:
                                covered += 1
                    
                    if covered > best_size:
                        best_size = covered
                        best_local = (L_set, R_set)

                if not best_local:
                    break

                L_set, R_set = best_local
                current_bicliques.append([list(L_set), list(R_set)])

                # Update remaining adjacency
                for u in L_set:
                    cur_adj[u] &= ~R_set
                for v in R_set:
                    cur_adj[v] &= ~L_set

                edges_left -= best_size
                if edges_left < 0:
                    edges_left = 0

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = current_bicliques
                if best_count == 1:
                    break

        return best_bicliques

    # Phase 1: Initial greedy with multiple restarts
    best_bicliques = None
    best_count = float('inf')

    # Seed: one biclique per vertex covering its outgoing edges
    seed_bicliques = []
    for u in range(n):
        right_side = [v for (u2, v) in edge_set if u2 == u]
        if right_side:
            seed_bicliques.append([[u], right_side])
    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)

    # Run greedy multiple times
    for attempt in range(30):
        if time.time() > deadline - 20:
            break
        result = greedy_cover(adj, max_restarts=30)
        if result is not None and len(result) < best_count:
            best_count = len(result)
            best_bicliques = result
            if best_count <= 5:
                break

    # Phase 2: Merge pass
    if best_bicliques is not None:
        merged = merge_pass(best_bicliques)
        if len(merged) < best_count:
            best_bicliques = merged
            best_count = len(merged)

    # Phase 3: Local search - un-cover random subsets and re-greedy-cover
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False

        # Merge pass
        merged = merge_pass(best_bicliques)
        if len(merged) < best_count:
            best_bicliques = merged
            best_count = len(merged)
            improved = True
            continue

        # Pick random subset of bicliques to re-cover
        k = random.randint(2, min(5, len(best_bicliques)))
        if k < 2:
            k = len(best_bicliques)
        if k == 0:
            break

        indices = random.sample(range(len(best_bicliques)), k)
        edges_to_recover = set()
        for idx in indices:
            L, R = best_bicliques[idx]
            edges_to_recover |= biclique_edges(L, R)

        remaining_bicliques = [best_bicliques[i] for i in range(len(best_bicliques)) if i not in set(indices)]

        # Build a temporary adjacency for just these edges
        temp_adj = [0] * n
        for (u, v) in edges_to_recover:
            temp_adj[u] |= (1 << v)
            temp_adj[v] |= (1 << u)

        new_subset = greedy_cover(temp_adj, max_restarts=20)

        if new_subset is not None:
            candidate = remaining_bicliques + new_subset
            if len(candidate) < best_count:
                best_bicliques = candidate
                best_count = len(candidate)
                improved = True
                # Immediate merge
                merged = merge_pass(best_bicliques)
                if len(merged) < best_count:
                    best_bicliques = merged
                    best_count = len(merged)

    # Final merge
    best_bicliques = merge_pass(best_bicliques)

    # Format output
    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])

    return result
