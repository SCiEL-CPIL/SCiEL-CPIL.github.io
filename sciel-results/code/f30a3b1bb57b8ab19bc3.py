import random
import time
import itertools

DEADLINE = 140.0  # safety margin before 150s kill


def _has_c5(adj_matrix, n):
    """Check if graph has a 5-cycle using bitmasks."""
    # Precompute adjacency as integers
    adj = [0] * n
    for i in range(n):
        mask = 0
        row = adj_matrix[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        adj[i] = mask

    # Check all 5-tuples in increasing order
    # For a 5-cycle v0-v1-v2-v3-v4-v0, we need:
    # v0<v1<v2<v3<v4, edges: v0v1, v1v2, v2v3, v3v4, v4v0
    # We also need to avoid chords? No, a 5-cycle as subgraph means there exist
    # 5 distinct vertices forming a cycle. It doesn't require them to be an induced 5-cycle.
    # So we just need to check if there's a cyclic ordering.

    # More efficient: for each pair (i,j) as start, enumerate paths of length 4
    # Actually, let's just do it with itertools.combinations for n <= 35
    # C(35,5) = 324632, which is manageable

    for combo in itertools.combinations(range(n), 5):
        a, b, c, d, e = combo
        # Check if there's a 5-cycle using these 5 vertices
        # The 5-cycle must use all 5 vertices in some cyclic order
        # There are (5-1)!/2 = 12 distinct cycles on 5 labeled vertices
        # But we can be smarter: try all permutations of the 5 vertices as a cycle
        # Actually, a cycle on 5 vertices: fix one vertex, arrange the other 4
        # But that's 24 checks per combination, so 324632 * 24 ~ 7.8M, might be slow

        # Alternative: check if the induced subgraph on these 5 vertices has a Hamiltonian cycle
        # For 5 vertices, a Hamiltonian cycle exists iff... let's just check all 12 cycles

        # The 12 distinct 5-cycles on vertices {0,1,2,3,4} (up to rotation and reversal):
        # (0,1,2,3,4), (0,1,2,4,3), (0,1,3,2,4), (0,1,3,4,2), (0,1,4,2,3), (0,1,4,3,2)
        # and their reversals are the same cycles, so 12 = 4!/2... wait, (5-1)!/2 = 12
        # Let me list them: fix 0 first. Then arrange 1,2,3,4 in a cycle: (4-1)!/2 = 3... no.
        # Number of distinct cycles on 5 labeled vertices = (5-1)!/2 = 12

        # Let's just try all 12 permutations
        cycles = [
            (0, 1, 2, 3, 4),
            (0, 1, 2, 4, 3),
            (0, 1, 3, 2, 4),
            (0, 1, 3, 4, 2),
            (0, 1, 4, 2, 3),
            (0, 1, 4, 3, 2),
            (0, 2, 1, 3, 4),
            (0, 2, 1, 4, 3),
            (0, 2, 3, 1, 4),
            (0, 2, 3, 4, 1),
            (0, 2, 4, 1, 3),
            (0, 2, 4, 3, 1),
        ]
        for cyc in cycles:
            v = [combo[cyc[k]] for k in range(5)]
            # Check edges v0-v1, v1-v2, v2-v3, v3-v4, v4-v0
            if (adj[v[0]] & (1 << v[1])) and \
               (adj[v[1]] & (1 << v[2])) and \
               (adj[v[2]] & (1 << v[3])) and \
               (adj[v[3]] & (1 << v[4])) and \
               (adj[v[4]] & (1 << v[0])):
                return True
    return False


def _has_c5_fast(adj, n):
    """Faster C5 check using bitmask adjacency."""
    # For n up to 35, check all 5-subsets
    # Precompute adj as list of ints
    for combo in itertools.combinations(range(n), 5):
        a, b, c, d, e = combo
        # Check all 12 cycles
        # (0,1,2,3,4): a-b-c-d-e-a
        if (adj[a] & (1 << b)) and (adj[b] & (1 << c)) and (adj[c] & (1 << d)) and (adj[d] & (1 << e)) and (adj[e] & (1 << a)):
            return True
        # (0,1,2,4,3): a-b-c-e-d-a
        if (adj[a] & (1 << b)) and (adj[b] & (1 << c)) and (adj[c] & (1 << e)) and (adj[e] & (1 << d)) and (adj[d] & (1 << a)):
            return True
        # (0,1,3,2,4): a-b-d-c-e-a
        if (adj[a] & (1 << b)) and (adj[b] & (1 << d)) and (adj[d] & (1 << c)) and (adj[c] & (1 << e)) and (adj[e] & (1 << a)):
            return True
        # (0,1,3,4,2): a-b-d-e-c-a
        if (adj[a] & (1 << b)) and (adj[b] & (1 << d)) and (adj[d] & (1 << e)) and (adj[e] & (1 << c)) and (adj[c] & (1 << a)):
            return True
        # (0,1,4,2,3): a-b-e-c-d-a
        if (adj[a] & (1 << b)) and (adj[b] & (1 << e)) and (adj[e] & (1 << c)) and (adj[c] & (1 << d)) and (adj[d] & (1 << a)):
            return True
        # (0,1,4,3,2): a-b-e-d-c-a
        if (adj[a] & (1 << b)) and (adj[b] & (1 << e)) and (adj[e] & (1 << d)) and (adj[d] & (1 << c)) and (adj[c] & (1 << a)):
            return True
        # (0,2,1,3,4): a-c-b-d-e-a
        if (adj[a] & (1 << c)) and (adj[c] & (1 << b)) and (adj[b] & (1 << d)) and (adj[d] & (1 << e)) and (adj[e] & (1 << a)):
            return True
        # (0,2,1,4,3): a-c-b-e-d-a
        if (adj[a] & (1 << c)) and (adj[c] & (1 << b)) and (adj[b] & (1 << e)) and (adj[e] & (1 << d)) and (adj[d] & (1 << a)):
            return True
        # (0,2,3,1,4): a-c-d-b-e-a
        if (adj[a] & (1 << c)) and (adj[c] & (1 << d)) and (adj[d] & (1 << b)) and (adj[b] & (1 << e)) and (adj[e] & (1 << a)):
            return True
        # (0,2,3,4,1): a-c-d-e-b-a
        if (adj[a] & (1 << c)) and (adj[c] & (1 << d)) and (adj[d] & (1 << e)) and (adj[e] & (1 << b)) and (adj[b] & (1 << a)):
            return True
        # (0,2,4,1,3): a-c-e-b-d-a
        if (adj[a] & (1 << c)) and (adj[c] & (1 << e)) and (adj[e] & (1 << b)) and (adj[b] & (1 << d)) and (adj[d] & (1 << a)):
            return True
        # (0,2,4,3,1): a-c-e-d-b-a
        if (adj[a] & (1 << c)) and (adj[c] & (1 << e)) and (adj[e] & (1 << d)) and (adj[d] & (1 << b)) and (adj[b] & (1 << a)):
            return True
    return False


def _independence_number(adj, n):
    """Compute the independence number using bitmask DP for n <= 35."""
    # For each vertex, the non-neighbors mask
    non_neigh = []
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i][j]:
                mask |= (1 << j)
        non_neigh.append(mask)

    # Use a recursive approach with memoization or just greedy + branch and bound
    # For n up to 35, we can use a simple approach:
    # Try to find independent set of size 9
    def _has_independent_set(size, candidates_mask, remaining):
        if remaining == 0:
            return True
        if candidates_mask == 0:
            return False
        # If not enough candidates
        if candidates_mask.bit_count() < remaining:
            return False
        # Pick a vertex with max degree in the induced subgraph (to reduce search)
        # Actually for independent set, pick vertex that eliminates most candidates
        best_v = -1
        best_count = -1
        cm = candidates_mask
        while cm:
            v = cm & (-cm)
            idx = v.bit_length() - 1
            # If we pick idx, we can only pick from non_neigh[idx] & candidates_mask
            new_candidates = non_neigh[idx] & candidates_mask & ~(1 << idx)
            cnt = new_candidates.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_v = idx
            cm ^= v

        # Branch: pick best_v or don't pick best_v
        # Pick best_v
        new_mask = candidates_mask & ~(1 << best_v)
        new_candidates = non_neigh[best_v] & new_mask
        if _has_independent_set(size, new_candidates, remaining - 1):
            return True
        # Don't pick best_v
        if _has_independent_set(size, new_mask, remaining):
            return True
        return False

    if _has_independent_set(9, (1 << n) - 1, 9):
        return 9  # at least 9
    return 8  # less than 9


def _has_independent_set_9(adj, n):
    """Check if there's an independent set of size 9."""
    non_neigh = []
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i][j]:
                mask |= (1 << j)
        non_neigh.append(mask)

    def _search(candidates_mask, remaining):
        if remaining == 0:
            return True
        if candidates_mask == 0 or candidates_mask.bit_count() < remaining:
            return False
        # Pick vertex to branch on
        # Choose vertex that maximizes pruning
        cm = candidates_mask
        best_v = -1
        best_eliminate = -1
        while cm:
            v = cm & (-cm)
            idx = v.bit_length() - 1
            # If we don't pick idx, we remove idx from candidates
            # If we pick idx, we can only use non_neigh[idx]
            # For branching, we want to pick a vertex where one branch is very constrained
            eliminate = (candidates_mask & ~(1 << idx)).bit_count()
            if eliminate > best_eliminate:
                best_eliminate = eliminate
                best_v = idx
            cm ^= v

        # Branch 1: pick best_v
        if best_v >= 0:
            new_cand = non_neigh[best_v] & (candidates_mask & ~(1 << best_v))
            if _search(new_cand, remaining - 1):
                return True
            # Branch 2: don't pick best_v
            if _search(candidates_mask & ~(1 << best_v), remaining):
                return True
        return False

    return _search((1 << n) - 1, 9)


def _edges_from_adj(adj, n):
    """Extract edge list from adjacency matrix."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges


def _is_valid(adj, n):
    """Check if graph is valid: no C5 and no independent set of size 9."""
    if _has_c5_fast(_adj_to_bitmask(adj, n), n):
        return False
    if _has_independent_set_9(adj, n):
        return False
    return True


def _adj_to_bitmask(adj, n):
    """Convert adjacency matrix to list of bitmasks."""
    result = []
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        result.append(mask)
    return result


def construct():
    start_time = time.time()
    best_n = 28
    best_edges = [[4 * b + i, 4 * b + j] for b in range(7) for i in range(4) for j in range(i + 1, 4)]

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Try random search for larger n
    for n in range(35, 27, -1):
        if time_left() < 30:
            break
        for attempt in range(500):
            if time_left() < 15:
                break
            # Generate random graph
            # Use a moderate edge probability
            p = 0.5
            adj = [[False] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i][j] = adj[j][i] = True

            # Quick check for C5
            if _has_c5_fast(_adj_to_bitmask(adj, n), n):
                continue

            # Check independence number
            if _has_independent_set_9(adj, n):
                continue

            # Valid!
            best_n = n
            best_edges = _edges_from_adj(adj, n)
            return {'n': n, 'edges': best_edges}

    # If we didn't find anything better, try local improvement on the 28-vertex base
    # or just return the base
    return {'n': best_n, 'edges': best_edges}
