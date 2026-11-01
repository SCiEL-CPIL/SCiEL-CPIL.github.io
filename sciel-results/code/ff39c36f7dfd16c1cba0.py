"""Kissing number in dimension 11 - D11 lattice constructive approach."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate all D11 lattice shortest vectors (squared norm 4, even coordinate sum)
    # Type A: weight-4 vectors with entries in {-1, 0, 1} and even number of minus signs
    # Type B: weight-2 vectors with entries in {-2, 0, 2}
    d11_vectors = []

    # Type A: weight 4, even number of negative signs
    for positions in itertools.combinations(range(D), 4):
        for n_neg in [0, 2, 4]:  # even number of negative signs
            for neg_positions in itertools.combinations(range(4), n_neg):
                v = [0] * D
                for i, pos in enumerate(positions):
                    if i in neg_positions:
                        v[pos] = -1
                    else:
                        v[pos] = 1
                d11_vectors.append(v)

    # Type B: weight 2, entries +/-2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([2, -2], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            d11_vectors.append(v)

    d11_array = np.array(d11_vectors, dtype=np.int64)
    N_d11 = len(d11_array)

    # Verify the configuration is valid: min pairwise distance >= max norm
    # For D11 shortest vectors, all have norm^2 = 4, and max |dot| = 2
    # so min distance^2 = 4 - 2*2 = 0... wait, let me verify properly
    # Actually: distance^2 = |a|^2 + |b|^2 - 2*a.b = 4 + 4 - 2*(a.b)
    # We need distance^2 >= 4 (max norm^2), so 8 - 2*(a.b) >= 4, i.e., a.b <= 2
    # And we also need distance^2 >= 0, so a.b <= 4 (always true for unit-ish vectors)
    # The key constraint is a.b <= 2 for all pairs

    # Let's verify by computing the conflict matrix
    dots = d11_array @ d11_array.T
    np.fill_diagonal(dots, 0)
    max_dot = np.max(dots)

    # If max_dot > 2, we need to filter. But for D11, max_dot should be 2.
    # If there are conflicts, we fall back to greedy selection.
    if max_dot > 2:
        # Fallback: greedy independent set
        conflicts = dots > 2
        np.fill_diagonal(conflicts, False)
        neighbors = [np.where(conflicts[i])[0] for i in range(N_d11)]
        neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

        rng = np.random.default_rng(42)
        in_set = np.zeros(N_d11, dtype=bool)
        alive = np.ones(N_d11, dtype=bool)
        degrees = neighbor_counts.copy()

        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)

        final_d11 = d11_array[in_set]
    else:
        # All D11 vectors are mutually compatible
        final_d11 = d11_array

    # Add W1 vectors: weight 1, entry +/-2 (norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [2, -2]:
            v = [0] * D
            v[pos] = sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    # Check if W1 vectors are compatible with D11 vectors
    # dot(W1, D11): W1 has one entry of +/-2, D11 has entries in {-2,-1,0,1,2}
    # The dot is +/-2 * (one entry of D11)
    # For Type A D11: entries are -1,0,1, so dot is in {-2,0,2} -> max |dot| = 2 OK
    # For Type B D11: entries are -2,0,2, so dot can be +/-4 -> CONFLICT if same position
    # So W1 vectors conflict with Type B D11 vectors at the same position

    # Let's check compatibility of W1 with final_d11
    if len(final_d11) > 0:
        cross_dots = vectors_v1 @ final_d11.T
        w1_conflicts = cross_dots > 2  # shape (22, N_d11)

        # For each W1 vector, check if it conflicts with any D11 vector
        w1_compatible = ~np.any(w1_conflicts, axis=1)
        compatible_v1 = vectors_v1[w1_compatible]
    else:
        compatible_v1 = vectors_v1

    final_vectors = np.vstack([final_d11, compatible_v1])

    return final_vectors
