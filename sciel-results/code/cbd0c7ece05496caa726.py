"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from scipy.optimize import linear_sum_assignment


def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin

    def elapsed_ok():
        return time.time() < deadline

    # Start with the seed: all 2-coordinate vectors (±1 on two coords, 0 elsewhere)
    seed_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                seed_vectors.append(v)
    vectors = np.array(seed_vectors, dtype=np.int64)

    # Current best norm squared and min pairwise distance squared
    def compute_stats(vecs):
        norms_sq = np.sum(vecs * vecs, axis=1)
        max_norm_sq = int(np.max(norms_sq))
        # Min pairwise distance: use the fact that dist(u,v)^2 = ||u||^2 + ||v||^2 - 2 u·v
        # For efficiency, compute Gram matrix
        G = vecs @ vecs.T
        dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * G
        np.fill_diagonal(dists_sq, np.inf)
        min_dist_sq = int(np.min(dists_sq))
        return max_norm_sq, min_dist_sq

    max_norm_sq, min_dist_sq = compute_stats(vectors)

    # Try to add more vectors by searching for vectors that satisfy:
    # ||v||^2 <= max_norm_sq  (so v doesn't increase max norm)
    # ||v - u||^2 >= max_norm_sq for all existing u
    # We search over small integer vectors.

    # Candidate pool: all integer vectors with coords in {-2,-1,0,1} (norm_sq <= D)
    # That's 3^11 = 177147 candidates, manageable.
    # Actually let's be a bit more generous: coords in {-2,-1,0,1,2} -> 5^11 = 48828125, too many.
    # Stick with {-1,0,1}^D minus zero vector.

    # Generate all nonzero vectors in {-1,0,1}^D
    # 3^11 = 177147, subtract 1 for zero = 177146
    # We already have the 2-coord ones. Let's generate all and filter.

    # For efficiency, precompute existing vectors as a matrix and check constraints.
    # For a candidate v, we need:
    #   ||v||^2 <= max_norm_sq
    #   min_u (||v - u||^2) >= max_norm_sq
    #   i.e., min_u (||v||^2 + ||u||^2 - 2 v·u) >= max_norm_sq
    #   i.e., min_u (||u||^2 - 2 v·u) >= max_norm_sq - ||v||^2

    # Let's iterate over candidates sorted by some heuristic (e.g., higher norm first, or just all)
    # and greedily add if feasible.

    # Generate all {-1,0,1}^D vectors excluding zero
    # Use itertools.product
    all_candidates = []
    for coord_tuple in itertools.product([-1, 0, 1], repeat=D):
        if any(c != 0 for c in coord_tuple):
            all_candidates.append(np.array(coord_tuple, dtype=np.int64))

    # This is 177146 vectors. For each, checking against all existing vectors is O(N_existing * D).
    # N_existing starts at 220 (55 pairs * 4 signs). As we add more, it grows.
    # 177146 * 220 * 11 ≈ 43 million ops for first pass, should be OK in NumPy if vectorized.

    # Vectorized feasibility check:
    # existing = vectors (k x D)
    # For candidate v (1 x D):
    #   v_norm_sq = sum(v^2)
    #   If v_norm_sq > max_norm_sq, skip.
    #   dots = existing @ v  (k,)
    #   dists_sq = v_norm_sq + norms_sq_existing - 2 * dots
    #   If min(dists_sq) >= max_norm_sq, feasible.

    # Let's convert all_candidates to a matrix for batch processing, but memory might be an issue.
    # 177146 * 11 * 8 bytes ≈ 15.6 MB, fine.

    cand_matrix = np.array(all_candidates, dtype=np.int64)  # (M, D)
    M = cand_matrix.shape[0]
    cand_norms_sq = np.sum(cand_matrix * cand_matrix, axis=1)  # (M,)

    # Filter candidates with norm_sq <= max_norm_sq
    valid_mask = cand_norms_sq <= max_norm_sq
    cand_matrix = cand_matrix[valid_mask]
    cand_norms_sq = cand_norms_sq[valid_mask]
    M = cand_matrix.shape[0]

    # Sort candidates by norm_sq descending (prefer adding higher-norm vectors? or maybe ascending to pack more?)
    # Actually, to maximize count, we want vectors that are "compatible" with each other.
    # Greedy: process in some order, add if feasible.
    # Let's try sorting by norm_sq ascending first (smaller vectors might fit more easily between existing ones? 
    # But existing already include all 2-coord vectors with norm_sq=2. So candidates with norm_sq=2 are already present.
    # Candidates with norm_sq=1: single 1s. There are 22 of them (11 coords * 2 signs).
    # Candidates with norm_sq=3: 3 coords set to ±1. C(11,3)*8 = 165*8 = 1320.
    # etc.

    # Let's sort by norm_sq, then by lexicographic order for determinism.
    order = np.argsort(cand_norms_sq)
    cand_matrix = cand_matrix[order]
    cand_norms_sq = cand_norms_sq[order]

    # Now greedily add
    added = 0
    for idx in range(M):
        if not elapsed_ok():
            break
        v = cand_matrix[idx]
        v_norm_sq = cand_norms_sq[idx]

        # Check against existing vectors
        # dots = vectors @ v
        dots = vectors @ v
        dists_sq = v_norm_sq + np.sum(vectors * vectors, axis=1) - 2 * dots
        if np.min(dists_sq) >= max_norm_sq:
            vectors = np.vstack([vectors, v])
            added += 1
            # Update max_norm_sq if needed (it shouldn't increase since v_norm_sq <= max_norm_sq)

    # After greedy, try a few rounds of re-optimization:
    # 1. Remove vectors that are "redundant" (not needed to maintain min distance) -- complex, skip.
    # 2. Try adding more from a different candidate set, e.g., vectors with a 2 in them.

    # Second candidate pool: vectors with coords in {-2,-1,0,1} but at least one 2 or -2
    # This is more complex to generate. Let's try a different approach:
    # Local search: for pairs of existing vectors, see if we can "shift" one slightly.
    # Or: try adding vectors of the form e_i * a + e_j * b where a,b in {-2,-1,1,2}
    # But we already have all 1,1 combinations. Let's try 2,1 and 2,2.

    # Generate 2-coord vectors with larger entries
    extra_candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (-2, -1, 1, 2):
            for sj in (-2, -1, 1, 2):
                if si == 0 and sj == 0:
                    continue
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                extra_candidates.append(v)

    # Also single-coord vectors with magnitude 2
    for i in range(D):
        for si in (-2, 2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = si
            extra_candidates.append(v)

    extra_matrix = np.array(extra_candidates, dtype=np.int64)
    extra_norms_sq = np.sum(extra_matrix * extra_matrix, axis=1)
    valid_extra = extra_norms_sq <= max_norm_sq
    extra_matrix = extra_matrix[valid_extra]
    extra_norms_sq = extra_norms_sq[valid_extra]

    for idx in range(extra_matrix.shape[0]):
        if not elapsed_ok():
            break
        v = extra_matrix[idx]
        v_norm_sq = extra_norms_sq[idx]
        dots = vectors @ v
        dists_sq = v_norm_sq + np.sum(vectors * vectors, axis=1) - 2 * dots
        if np.min(dists_sq) >= max_norm_sq:
            vectors = np.vstack([vectors, v])

    # Final check: verify the configuration is valid
    final_stats = compute_stats(vectors)
    if final_stats[1] < final_stats[0]:
        # Invalid - should not happen, but just in case, fall back to seed
        return np.array(seed_vectors, dtype=np.int64)

    return vectors.astype(np.int64)

# EVOLVE-BLOCK-END

