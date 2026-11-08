"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid_set(vectors):
    """Check that min pairwise distance >= max norm for the set of vectors."""
    if len(vectors) == 0:
        return True
    if len(vectors) == 1:
        return True
    norms_sq = np.sum(vectors * vectors, axis=1)
    max_norm_sq = np.max(norms_sq)
    if max_norm_sq == 0:
        return False
    # Compute pairwise squared distances
    # dist^2 = |a|^2 + |b|^2 - 2*a.b
    dots = vectors @ vectors.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    return min_dist_sq >= max_norm_sq


def _score(vectors):
    """Return the number of vectors if valid, else 0."""
    if _is_valid_set(vectors):
        return len(vectors)
    return 0


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_vectors = []
    best_score = 0

    # Start with the seed: all weight-2 vectors with entries ±1
    # These have norm^2 = 2
    seed_vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    seed_vectors.append(v)
    seed_vectors = np.array(seed_vectors, dtype=np.int64)

    # The seed set itself: check validity
    if _is_valid_set(seed_vectors):
        best_vectors = seed_vectors.copy()
        best_score = len(seed_vectors)

    # Now try to add more vectors. Strategy:
    # 1. Try weight-4 vectors (4 non-zero entries, each ±1 or ±2)
    # 2. Use spectral/graph approach to find compatible subsets
    # 3. Greedy local search to improve

    # First, let's understand the structure of the seed.
    # For weight-2 vectors with entries ±1: norm^2 = 2
    # Two such vectors are compatible if dist^2 >= 2, i.e., |a-b|^2 >= 2
    # |a-b|^2 = |a|^2 + |b|^2 - 2a.b = 4 - 2a.b
    # So we need 4 - 2a.b >= 2 => a.b <= 1
    # For weight-2 ±1 vectors, a.b can be: -2, -1, 0, 1, 2
    # a.b = 2 means same vector (not possible in set)
    # a.b = 1 means they share one coordinate with same sign
    # So incompatible pairs share exactly one coordinate with the same sign.

    # The conflict graph: vertices are the 4*C(11,2) = 220 weight-2 ±1 vectors.
    # Two are adjacent if they share a coordinate with the same sign.

    # Let's build the conflict graph and find a maximum independent set via greedy.
    # But actually, the seed set IS all 220 vectors. Are they all mutually compatible?
    # No! If v1 = (1,1,0,...) and v2 = (1,0,1,...), they share coordinate 0 with sign +1,
    # so a.b = 1, dist^2 = 4-2 = 2 >= 2. OK, that's fine.
    # If v1 = (1,1,0,...) and v2 = (1,1,0,...) same vector - not in set.
    # If v1 = (1,1,0,...) and v2 = (1,-1,0,...), a.b = 1-1=0, dist^2=4>=2. Fine.
    # If v1 = (1,1,0,...) and v2 = (-1,1,0,...), a.b = -1+1=0, dist^2=4. Fine.
    # Wait, when is a.b = 1? When they share exactly one coordinate with same sign.
    # v1 = (1,1,0,...), v2 = (1,0,1,...): share coord 0, both +1. a.b = 1*1 + 1*0 = 1.
    # dist^2 = 2+2-2(1) = 2. This equals max_norm_sq=2. So it's valid (>=).
    # So actually ALL pairs of distinct weight-2 ±1 vectors have dist^2 >= 2?
    # Let's check: minimum a.b for distinct vectors.
    # They can share 0, 1, or 2 coordinates.
    # If they share 2 coordinates: they're the same or opposite. If same, not distinct.
    # If opposite: v1=(1,1,0..), v2=(-1,-1,0..): a.b = -1-1 = -2. dist^2 = 2+2-2(-2)=8. Fine.
    # If share 1 coordinate: a.b = ±1 (the shared coord contribution). dist^2 = 4-2(±1) = 2 or 6. Fine.
    # If share 0 coordinates: a.b = 0. dist^2 = 4. Fine.
    # So the entire seed set of 220 vectors is valid! Score = 220.

    # Now can we add more vectors? Let's try adding weight-4 vectors.
    # A weight-4 vector with entries ±1 has norm^2 = 4.
    # For it to be compatible with all seed vectors (norm^2=2), we need:
    # dist^2 >= max(4, 2) = 4 (since the new vector has larger norm)
    # dist^2 = |a|^2 + |b|^2 - 2a.b = 4 + 2 - 2a.b = 6 - 2a.b >= 4
    # => a.b <= 1
    # So we need weight-4 ±1 vectors whose dot product with every weight-2 ±1 vector is <= 1.

    # Let's generate candidate weight-4 vectors and filter.
    candidates = []
    for idx in range(len(seed_vectors)):
        sv = seed_vectors[idx]
        # For a candidate c with norm^2=4, need c·sv <= 1 for all sv
        # This is a strong constraint. Let's just generate and test.
        pass

    # Generate all weight-4 vectors with entries in {-1, 1}
    # Number: C(11,4) * 2^4 = 330 * 16 = 5280
    import itertools
    weight4_candidates = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for coord, sign in zip(support, signs):
                v[coord] = sign
            weight4_candidates.append(v)
    weight4_candidates = np.array(weight4_candidates, dtype=np.int64)

    # Filter: keep only those compatible with ALL seed vectors
    # c·sv <= 1 for all sv in seed
    # And c·c' <= 1 for all c,c' in the selected weight-4 set (for mutual compatibility)
    # Also need c·c' such that dist^2 >= 4: 4+4-2c·c' >= 4 => c·c' <= 2

    # First filter against seed
    dots_with_seed = weight4_candidates @ seed_vectors.T  # (5280, 220)
    compatible_with_seed = np.all(dots_with_seed <= 1, axis=1)
    filtered = weight4_candidates[compatible_with_seed]

    if len(filtered) > 0:
        # Now find a large subset of filtered that are mutually compatible
        # Two weight-4 vectors c1, c2: need c1·c2 <= 2 (for dist^2 >= 4)
        # Build conflict graph among filtered
        n_filtered = len(filtered)
        if n_filtered > 0:
            dots_ff = filtered @ filtered.T
            # Conflict if dot > 2
            conflict = (dots_ff > 2)
            np.fill_diagonal(conflict, False)

            # Greedy: pick vertices with fewest conflicts first
            degrees = np.sum(conflict, axis=1)
            order = np.argsort(degrees)
            selected = []
            available = np.ones(n_filtered, dtype=bool)
            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                available[conflict[idx]] = False
                if time.time() > deadline:
                    break

            if len(selected) > 0:
                new_vectors = np.vstack([seed_vectors, filtered[selected]])
                if _is_valid_set(new_vectors):
                    if len(new_vectors) > best_score:
                        best_vectors = new_vectors
                        best_score = len(new_vectors)

    # Try adding weight-3 vectors with entries in {-1, 0, 1}
    # Weight-3 vectors: 3 non-zero entries, each ±1. Norm^2 = 3.
    # Need dist^2 >= max(3, 2) = 3 for compatibility with seed
    # dist^2 = 3 + 2 - 2a.b = 5 - 2a.b >= 3 => a.b <= 1
    # Mutual: 3+3-2a.b >= 3 => a.b <= 3 (always true for ±1 entries, max dot is 3)
    # Actually max dot for two weight-3 ±1 vectors is 3 (same vector), min is -3.
    # So mutual compatibility: a.b <= 3, always satisfied for distinct vectors? 
    # No: if a.b = 3, they're the same. If a.b = 2, dist^2 = 6-4=2 < 3. Not compatible!
    # So need a.b <= 2 for mutual compatibility.

    weight3_candidates = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = np.zeros(D, dtype=np.int64)
            for coord, sign in zip(support, signs):
                v[coord] = sign
            weight3_candidates.append(v)
    weight3_candidates = np.array(weight3_candidates, dtype=np.int64)

    # Filter against seed: a.b <= 1
    dots_w3_seed = weight3_candidates @ seed_vectors.T
    compatible_w3 = np.all(dots_w3_seed <= 1, axis=1)
    filtered_w3 = weight3_candidates[compatible_w3]

    if len(filtered_w3) > 0:
        n_f3 = len(filtered_w3)
        dots_w3 = filtered_w3 @ filtered_w3.T
        conflict_w3 = (dots_w3 > 2)
        np.fill_diagonal(conflict_w3, False)
        degrees_w3 = np.sum(conflict_w3, axis=1)
        order_w3 = np.argsort(degrees_w3)
        selected_w3 = []
        available_w3 = np.ones(n_f3, dtype=bool)
        for idx in order_w3:
            if not available_w3[idx]:
                continue
            selected_w3.append(idx)
            available_w3[conflict_w3[idx]] = False
            if time.time() > deadline:
                break

        if len(selected_w3) > 0:
            # Try combining with best so far
            if len(best_vectors) > 0:
                trial = np.vstack([best_vectors, filtered_w3[selected_w3]])
            else:
                trial = filtered_w3[selected_w3]
            if _is_valid_set(trial):
                if len(trial) > best_score:
                    best_vectors = trial
                    best_score = len(trial)

    # Try weight-4 vectors with entries in {-2, -1, 1, 2} - more general
    # But this explodes. Let's try a different approach: local search on the combined set.

    # Alternative: try to find a larger set using a different construction.
    # The Leech lattice or related constructions might give better results.
    # For now, let's try a simple local search: randomly add/remove vectors to improve.

    # Actually, let's try including vectors with larger norms.
    # A vector with norm^2 = N can coexist with others if all pairwise dist^2 >= N_max.
    # If we have vectors of different norms, the constraint is the MAX norm.

    # Let's try: keep only vectors of the same norm for simplicity.
    # We already have 220 vectors of norm^2=2.
    # Can we find more than 220 vectors of norm^2=2 that are mutually compatible?
    # All weight-2 ±1 vectors are the only integer vectors of norm^2=2 (up to sign).
    # Actually, (1,1,0,...) and (0,1,1,...) etc. - yes, that's all of them.
    # So 220 is the max for norm^2=2.

    # What about norm^2=4? Vectors like (2,0,0,...) have norm^2=4.
    # Or (1,1,1,1,0,...) weight-4 with ±1.
    # Let's try (2,0,...,0) type vectors: D such vectors (and their negatives).
    # These have norm^2=4. Two such vectors: dist^2 = 4+4-2(±4 or 0) = 8 or 16.
    # (2,0,..) and (-2,0,..): dist^2 = 8. (2,0,..) and (2,0,..,different coord): dist^2=8.
    # So all 2D=22 vectors of type (±2,0,...,0) are mutually compatible.
    # But they need to be compatible with the seed (norm^2=2):
    # dist^2 = 4+2-2(±2*±1) = 6-2(±2) = 2 or 10.
    # If the dot is 2 (same sign, same coord): dist^2 = 2 < 4. NOT compatible!
    # So (2,0,...) is incompatible with (1,0,1,...) type? No, (2,0,...)·(1,1,0,..) = 2.
    # dist^2 = 4+2-4 = 2 < 4. Incompatible.
    # So we can't mix norm^2=2 and norm^2=4 easily.

    # Let's just return the best we found.
    if len(best_vectors) == 0:
        best_vectors = seed_vectors
        best_score = len(seed_vectors)

    return best_vectors
# EVOLVE-BLOCK-END
