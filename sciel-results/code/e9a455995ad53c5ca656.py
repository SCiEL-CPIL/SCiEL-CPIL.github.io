"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 16
    total_pairs = n * (n - 1) // 2  # 120

    # Edge index mapping
    edge_idx = {}
    pair_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = len(pair_list)
            pair_list.append((i, j))

    # Q3 structure: 8 vertices, 12 edges
    q3_vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all 8-vertex subsets
    all_8sets = list(combinations(range(16), 8))

    # Precompute for each 8-set: all 8! assignments of Q3 vertices to the 8-set vertices
    # For each assignment, store the 12 edge indices (in the global edge_idx) that form the Q3
    # This is expensive but done once. We'll use a smarter approach:
    # For each 8-set, enumerate all 8! = 40320 bijections from Q3 vertices to 8-set vertices
    # For each bijection, check if all 12 Q3 edges map to actual graph edges
    # We'll precompute the edge masks for each 8-set

    # Actually, let's use a faster approach: for each 8-set, precompute which 
    # 12-edge subsets of the 28 possible edges within those 8 vertices form a Q3.
    # There are C(28,12) = 30M which is too many.
    # Better: for each 8-set, enumerate all 40320 assignments and collect the 12-edge bitmasks.

    # Precompute: for each 8-set, the set of 12-edge bitmasks (as frozensets of local edge positions)
    # that form a Q3. We'll store as a set of tuples of 12 local edge indices.

    # Local edge indexing within an 8-set: 8 vertices -> 28 edges
    # For a given 8-set vs = (v0,...,v7), local edge (i,j) maps to global edge (vs[i], vs[j])

    # Precompute all Q3 edge patterns as local edge index sets
    # A Q3 pattern is a set of 12 local edge indices (within 8 labeled vertices 0-7)
    # such that the induced graph on those 12 edges is exactly Q3.
    # Actually, any Q3 subgraph on 8 vertices uses exactly 12 edges.
    # The 12 edges of Q3 on vertices 0-7 are fixed: q3_edges above.
    # But we need all 8! permutations of vertex labels.

    # For each permutation p of [0,1,...,7], the Q3 edges become:
    # (p[0],p[1]), (p[0],p[2]), ..., etc.
    # Convert to local edge indices.

    # Local edge index for pair (a,b) in 8 vertices:
    # We'll use a lookup: local_edge_idx[(a,b)] for 0<=a<b<=7
    local_edge_idx = {}
    local_pair_list = []
    for i in range(8):
        for j in range(i + 1, 8):
            local_edge_idx[(i, j)] = len(local_pair_list)
            local_pair_list.append((i, j))
    # 28 local edges

    # Generate all Q3 edge patterns (sets of 12 local edge indices)
    q3_patterns = set()
    from itertools import permutations
    for perm in permutations(range(8)):
        # perm[i] = which vertex of the 8-set corresponds to Q3 vertex i
        # Q3 edge (a,b) becomes (perm[a], perm[b]) in local coords
        edge_set = set()
        for a, b in q3_edges:
            la, lb = perm[a], perm[b]
            if la > lb:
                la, lb = lb, la
            edge_set.add(local_edge_idx[(la, lb)])
        q3_patterns.add(frozenset(edge_set))

    # q3_patterns is a set of frozensets of 12 local edge indices
    num_patterns = len(q3_patterns)

    # For each 8-set, precompute the mapping from local edge index to global edge index
    # Then for each Q3 pattern, we can check if all 12 global edges are present.

    # To speed up: for each 8-set, precompute a 28-bit mask of which local edges are present
    # Then a Q3 pattern is present iff (pattern_mask & present_mask) == pattern_mask
    # i.e., all 12 edges of the pattern are in the present set.

    # Precompute pattern masks as integers (12 bits set out of 28)
    pattern_masks = []
    for pat in q3_patterns:
        mask = 0
        for e in pat:
            mask |= (1 << e)
        pattern_masks.append(mask)

    # For each 8-set, store the 28 local edges mapped to global edge indices
    # And we'll compute the present mask on the fly during search.

    # For efficiency, precompute for each 8-set: list of 28 global edge indices
    eightset_local_to_global = []
    for vs in all_8sets:
        mapping = []
        for i in range(8):
            for j in range(i + 1, 8):
                u, v = vs[i], vs[j]
                if u > v:
                    u, v = v, u
                mapping.append(edge_idx[(u, v)])
        eightset_local_to_global.append(mapping)

    def is_q3_free(edge_mask):
        """Check if the graph with edges represented by edge_mask (120-bit int) is Q3-free."""
        for idx, vs in enumerate(all_8sets):
            mapping = eightset_local_to_global[idx]
            # Compute present mask for this 8-set
            present = 0
            for le, ge in enumerate(mapping):
                if (edge_mask >> ge) & 1:
                    present |= (1 << le)
            # Check if any Q3 pattern is a subset of present
            for pm in pattern_masks:
                if (present & pm) == pm:
                    return False
        return True

    # Baseline: K3,13
    k313_mask = 0
    for a in range(3):
        for b in range(3, 16):
            u, v = min(a,b), max(a,b)
            k313_mask |= (1 << edge_idx[(u, v)])
    best_mask = k313_mask
    best_count = bin(k313_mask).count('1')

    # Greedy: try to add edges to K3,13
    current_mask = k313_mask
    candidate_edges = [(i, j) for i in range(16) for j in range(i + 1, 16)
                       if not ((k313_mask >> edge_idx[(i, j)]) & 1)]

    def edge_priority(e):
        u, v = e
        if u < 3 and v < 3:
            return 0
        if u >= 3 and v >= 3:
            return 1
        return 2

    candidate_edges.sort(key=edge_priority)

    for edge in candidate_edges:
        if time.time() > deadline - 30:
            break
        u, v = edge
        ge = edge_idx[(u, v)]
        new_mask = current_mask | (1 << ge)
        if is_q3_free(new_mask):
            current_mask = new_mask

    if bin(current_mask).count('1') > best_count:
        best_count = bin(current_mask).count('1')
        best_mask = current_mask

    # Simulated annealing to improve
    if time.time() < deadline - 10:
        random.seed(42)
        current = best_mask
        current_count = bin(current).count('1')
        best_sa = current
        best_sa_count = current_count
        T = 2.0
        cooling = 0.9995
        max_iters = 50000

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            # Try adding a random missing edge
            if random.random() < 0.6:
                # Add an edge
                missing = [(i, j) for i in range(16) for j in range(i+1, 16)
                           if not ((current >> edge_idx[(i,j)]) & 1)]
                if not missing:
                    continue
                u, v = random.choice(missing)
                ge = edge_idx[(u, v)]
                new_mask = current | (1 << ge)
                if is_q3_free(new_mask):
                    current = new_mask
                    current_count = bin(current).count('1')
                    if current_count > best_sa_count:
                        best_sa_count = current_count
                        best_sa = current
                else:
                    # Reject
                    pass
            else:
                # Swap: remove one edge, add another
                present = [(i, j) for i in range(16) for j in range(i+1, 16)
                           if (current >> edge_idx[(i,j)]) & 1]
                missing = [(i, j) for i in range(16) for j in range(i+1, 16)
                           if not ((current >> edge_idx[(i,j)]) & 1)]
                if not present or not missing:
                    continue
                u1, v1 = random.choice(present)
                u2, v2 = random.choice(missing)
                ge1 = edge_idx[(u1, v1)]
                ge2 = edge_idx[(u2, v2)]
                new_mask = (current & ~(1 << ge1)) | (1 << ge2)
                if is_q3_free(new_mask):
                    new_count = bin(new_mask).count('1')
                    delta = new_count - current_count
                    if delta >= 0 or random.random() < pow(2.718281828, delta / max(T, 0.01)):
                        current = new_mask
                        current_count = new_count
                        if current_count > best_sa_count:
                            best_sa_count = current_count
                            best_sa = current
            T *= cooling

        if best_sa_count > best_count:
            best_count = best_sa_count
            best_mask = best_sa

    result = []
    for (u, v) in pair_list:
        ge = edge_idx[(u, v)]
        if (best_mask >> ge) & 1:
            result.append([u, v])
    return result
