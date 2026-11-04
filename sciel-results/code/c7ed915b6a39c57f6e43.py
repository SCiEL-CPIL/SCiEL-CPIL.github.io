import numpy as np
import time
import random
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11
    time_budget_mis = 80.0  # seconds for MIS search
    time_budget_add = 55.0  # seconds for adding lower-weight vectors

    # Build the full candidate set:
    # 1) All weight-4 vectors (330 supports * 16 signs = 5280)
    # 2) All weight-3 vectors (165 supports * 8 signs = 1320)
    # 3) All weight-2 vectors (55 supports * 4 signs = 220)
    # 4) All weight-1 vectors (11 supports * 2 signs = 22)
    # Total: 6842 candidates

    # We want to select a subset S such that for all u, v in S, u != v:
    #   ||u - v||^2 >= max_{w in S} ||w||^2
    #
    # If we restrict to weight-4 vectors only, max_norm^2 = 4, and we need dist^2 >= 4.
    # For two weight-4 vectors with supports S, T:
    #   dist^2 = 4 + 4 - 2*u.v = 8 - 2*u.v
    #   Need 8 - 2*u.v >= 4 => u.v <= 2
    #   u.v = sum_{i in S∩T} s_i * t_i, |S∩T| = m
    #   If m <= 2: max u.v = 2. OK.
    #   If m = 3: u.v can be 3 (all signs agree). Forbidden.
    #
    # If we include weight-3 vectors (norm^2 = 3):
    #   max_norm^2 is still 4 (from weight-4 vectors).
    #   dist^2 between weight-4 (norm^2=4) and weight-3 (norm^2=3):
    #     dist^2 = 4 + 3 - 2*u.v = 7 - 2*u.v
    #     Need >= 4 => u.v <= 1.5 => u.v <= 1
    #   u.v = sum over intersection. If |S∩T| = m, u.v ranges in [-m, m] step 2.
    #   m=0: 0. OK. m=1: ±1. OK. m=2: -2,0,2. u.v=2 forbidden.
    #   m=3: weight-3 support is subset of weight-4 support. u.v = ±3,±1. u.v=3 forbidden, u.v=1 OK.
    #
    # If we include weight-2 vectors (norm^2 = 2):
    #   dist^2 between weight-4 and weight-2: 4+2-2*u.v = 6-2*u.v >= 4 => u.v <= 1
    #   dist^2 between weight-3 and weight-2: 3+2-2*u.v = 5-2*u.v >= 4 => u.v <= 0.5 => u.v <= 0
    #   dist^2 between weight-2 and weight-2: 2+2-2*u.v = 4-2*u.v >= 4 => u.v <= 0
    #
    # If we include weight-1 vectors (norm^2 = 1):
    #   dist^2 between weight-4 and weight-1: 4+1-2*u.v = 5-2*u.v >= 4 => u.v <= 0.5 => u.v <= 0
    #   dist^2 between weight-3 and weight-1: 3+1-2*u.v = 4-2*u.v >= 4 => u.v <= 0
    #   dist^2 between weight-2 and weight-1: 2+1-2*u.v = 3-2*u.v >= 4 => u.v <= -0.5 => u.v <= -1
    #   dist^2 between weight-1 and weight-1: 1+1-2*u.v = 2-2*u.v >= 4 => u.v <= -1
    #
    # This is getting complex. Let's use a unified approach:
    # Build the full conflict graph and solve MIS on it.
    # With 6842 nodes, this is too large for exact MIS, but we can use greedy + local search.
    #
    # Actually, let's think about this differently. The key insight from the current program
    # is that weight-4 vectors with supports having pairwise intersection <= 2 give a valid
    # configuration. The MIS on 330 4-subsets with conflict when |S∩T|>=3 gives ~35 supports,
    # yielding 35*16 = 560 vectors.
    #
    # Can we do better by including some weight-3, weight-2, or weight-1 vectors?
    # Let's estimate: if we have 35 weight-4 supports, each weight-3 vector with support T
    # conflicts with a weight-4 vector with support S iff |S∩T| >= 2 AND the signs are
    # "too aligned". Actually, for weight-4 vs weight-3, we need u.v <= 1.
    # If |S∩T| = 2, u.v can be -2, 0, 2. u.v=2 is forbidden.
    # If |S∩T| = 1, u.v = ±1. OK.
    # If |S∩T| = 0, u.v = 0. OK.
    # If |S∩T| = 3, T is subset of S. u.v = ±3, ±1. u.v=3 forbidden.
    #
    # So a weight-3 vector with support T is compatible with a weight-4 vector with support S
    # if either |S∩T| <= 1, or |S∩T| = 2 with u.v <= 0 (i.e., signs not both +1 or both -1
    # on both intersection points), or |S∩T| = 3 with u.v <= 1 (i.e., not all signs agree).
    #
    # This is very complex to handle in a unified graph. Let me try a different approach:
    # Focus on improving the weight-4 MIS solution, and then try to add compatible
    # lower-weight vectors greedily.

    # Step 1: Find a large independent set of 4-subsets with pairwise intersection <= 2
    supports4 = list(combinations(range(D), 4))
    N4 = len(supports4)
    sup4_sets = [set(s) for s in supports4]

    # Build conflict graph for 4-subsets: conflict if |S∩T| >= 3
    adj4 = [set() for _ in range(N4)]
    for i in range(N4):
        for j in range(i + 1, N4):
            if len(sup4_sets[i] & sup4_sets[j]) >= 3:
                adj4[i].add(j)
                adj4[j].add(i)

    def greedy_mis(adj, n, rng, deadline):
        """Greedy minimum-degree MIS."""
        selected = set()
        avail = set(range(n))
        while avail:
            if time.time() > deadline:
                break
            min_deg = float('inf')
            candidates = []
            for u in avail:
                deg = len(adj[u] & avail)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
            u = rng.choice(candidates)
            selected.add(u)
            avail -= {u} | (adj[u] & avail)
        return selected

    def local_search_mis(adj, n, iset, rng, deadline):
        """1-2 swap local search for MIS."""
        iset = set(iset)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v in list(iset):
                if time.time() > deadline:
                    break
                sel_minus_v = iset - {v}
                potential = []
                for x in range(n):
                    if x in iset or x == v:
                        continue
                    if not (adj[x] & sel_minus_v):
                        potential.append(x)
                L = len(potential)
                found = False
                for i in range(L):
                    u1 = potential[i]
                    for j in range(i + 1, L):
                        u2 = potential[j]
                        if u2 not in adj[u1]:
                            iset.remove(v)
                            iset.add(u1)
                            iset.add(u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break
        return iset

    # Find good MIS on 4-subset conflict graph
    best_iset4 = set()
    best_size = 0
    rng = np.random.default_rng(42)

    mis_deadline = start_time + time_budget_mis
    for r in range(60):
        if time.time() > mis_deadline:
            break
        iset = greedy_mis(adj4, N4, rng, mis_deadline)
        iset = local_search_mis(adj4, N4, iset, rng, mis_deadline)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset4 = iset.copy()

    # Generate all weight-4 vectors from the selected supports
    vectors = []
    for i in best_iset4:
        sup = supports4[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)

    vec_array = np.array(vectors, dtype=np.int8)
    vec_norm2 = np.sum(vec_array.astype(np.int64) ** 2, axis=1)

    # Step 2: Try to add weight-3 vectors using vectorized compatibility checks
    # Compatibility: for candidate v3 (norm^2=3) vs existing vec (norm^2=n2):
    #   dist^2 = n2 + 3 - 2*dot >= max_norm2 (=4 if any weight-4 exists)
    #   => dot <= (n2 + 3 - 4) / 2
    #   For n2=4: dot <= 1; n2=3: dot <= 1; n2=2: dot <= 0; n2=1: dot <= 0

    supports3 = list(combinations(range(D), 3))
    N3 = len(supports3)

    # Pre-compute all weight-3 candidates as a matrix
    w3_candidates = []
    w3_supports = []
    for T in supports3:
        for signs in product([1, -1], repeat=3):
            v3 = np.zeros(D, dtype=np.int8)
            for idx, s in zip(T, signs):
                v3[idx] = s
            w3_candidates.append(v3)
            w3_supports.append(T)
    w3_matrix = np.array(w3_candidates, dtype=np.int8)  # shape (1320, 11)

    # Compute dot products: (1320, k)
    dots_w3 = w3_matrix.astype(np.int64) @ vec_array.T.astype(np.int64)  # (1320, k)

    # Determine which candidates are compatible
    # For each existing vector j with norm2 n2_j, need dot[i,j] <= (n2_j + 3 - 4)/2
    # i.e., dot[i,j] <= (n2_j - 1) / 2
    # n2=4 -> 1.5 -> floor to 1; n2=3 -> 1; n2=2 -> 0.5 -> 0; n2=1 -> 0
    thresholds = ((vec_norm2 - 1) // 2).astype(np.int64)  # shape (k,)
    # dots_w3[i, j] <= thresholds[j] for all j
    compatible_w3 = np.all(dots_w3 <= thresholds[np.newaxis, :], axis=1)  # shape (1320,)

    for i in range(N3 * 8):
        if time.time() > start_time + time_budget_mis + time_budget_add:
            break
        if compatible_w3[i]:
            vectors.append(w3_candidates[i])
            vec_array = np.vstack([vec_array, w3_candidates[i].reshape(1, -1)])
            vec_norm2 = np.append(vec_norm2, 3)

    # Step 3: Try to add weight-2 vectors using vectorized checks
    supports2 = list(combinations(range(D), 2))
    N2 = len(supports2)

    w2_candidates = []
    for S2 in supports2:
        for signs in product([1, -1], repeat=2):
            v2 = np.zeros(D, dtype=np.int8)
            for idx, s in zip(S2, signs):
                v2[idx] = s
            w2_candidates.append(v2)
    w2_matrix = np.array(w2_candidates, dtype=np.int8)  # shape (220, 11)

    # Recompute dots with updated vec_array
    dots_w2 = w2_matrix.astype(np.int64) @ vec_array.T.astype(np.int64)  # (220, k')

    # For weight-2 candidate (norm^2=2) vs existing vec (norm^2=n2):
    #   dist^2 = n2 + 2 - 2*dot >= 4 => dot <= (n2 - 2) / 2
    #   n2=4 -> 1; n2=3 -> 0.5 -> 0; n2=2 -> 0; n2=1 -> -0.5 -> -1
    thresholds_w2 = ((vec_norm2 - 2) // 2).astype(np.int64)
    compatible_w2 = np.all(dots_w2 <= thresholds_w2[np.newaxis, :], axis=1)

    for i in range(N2 * 4):
        if time.time() > start_time + time_budget_mis + time_budget_add:
            break
        if compatible_w2[i]:
            vectors.append(w2_candidates[i])
            vec_array = np.vstack([vec_array, w2_candidates[i].reshape(1, -1)])
            vec_norm2 = np.append(vec_norm2, 2)

    # Step 4: Try to add weight-1 vectors
    w1_candidates = []
    for d in range(D):
        for s in [1, -1]:
            v1 = np.zeros(D, dtype=np.int8)
            v1[d] = s
            w1_candidates.append(v1)
    w1_matrix = np.array(w1_candidates, dtype=np.int8)  # shape (22, 11)

    dots_w1 = w1_matrix.astype(np.int64) @ vec_array.T.astype(np.int64)

    # For weight-1 candidate (norm^2=1) vs existing vec (norm^2=n2):
    #   dist^2 = n2 + 1 - 2*dot >= 4 => dot <= (n2 - 3) / 2
    #   n2=4 -> 0.5 -> 0; n2=3 -> 0; n2=2 -> -0.5 -> -1; n2=1 -> -1
    thresholds_w1 = ((vec_norm2 - 3) // 2).astype(np.int64)
    compatible_w1 = np.all(dots_w1 <= thresholds_w1[np.newaxis, :], axis=1)

    for i in range(22):
        if time.time() > start_time + time_budget_mis + time_budget_add:
            break
        if compatible_w1[i]:
            vectors.append(w1_candidates[i])
            vec_array = np.vstack([vec_array, w1_candidates[i].reshape(1, -1)])
            vec_norm2 = np.append(vec_norm2, 1)

    if not vectors:
        return np.array([[]], dtype=np.int8)

    return np.array(vectors, dtype=np.int8)
