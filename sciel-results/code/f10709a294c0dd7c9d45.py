import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # Greedy MIS with dynamic degree tracking
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int64)
    rng = np.random.default_rng(42)

    # Precompute neighbor lists
    neighbor_lists = []
    for i in range(N4):
        neighbor_lists.append(np.where(conflicts[i])[0])

    while np.any(alive) and time.time() < deadline - 5:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break

        # Find min degree among alive nodes
        alive_degrees = degrees[alive_indices]
        min_deg = np.min(alive_degrees)
        cands = alive_indices[alive_degrees == min_deg]
        chosen = cands[rng.integers(len(cands))]

        selected_indices.append(chosen)

        # Collect all nodes to remove: chosen + its alive neighbors
        neighbors = neighbor_lists[chosen]
        to_remove_set = {chosen}
        for n in neighbors:
            if alive[n]:
                to_remove_set.add(n)

        to_remove = np.array(list(to_remove_set))
        alive[to_remove] = False

        # Update degrees: for each remaining alive node, subtract number of removed neighbors
        remaining = np.where(alive)[0]
        if len(remaining) > 0:
            removed_neighbor_counts = conflicts[remaining][:, to_remove].sum(axis=1)
            degrees[remaining] -= removed_neighbor_counts

    result_v4 = vectors_v4[selected_indices]

    # V1 vectors: check compatibility with selected V4
    # V1 has norm^2 = 4, V4 has norm^2 = 4
    # dot(V1, V4): V1 has one entry ±2, V4 has four entries ±1
    # If V1's nonzero position is among V4's nonzero positions, dot = ±2
    # Otherwise dot = 0
    # So dot ≤ 2 always holds. All V1 are compatible with all V4.
    # V1 vs V1: two V1 vectors either are same (dot=4>2, conflict) or different positions (dot=0) or opposite sign same position (dot=-4)
    # So we can take at most one V1 per position (either +2 or -2), but not both.
    # Actually, same position same sign: dot = 4 > 2, conflict
    # Same position opposite sign: dot = -4, OK
    # Different positions: dot = 0, OK
    # So we can take all 22 V1 vectors! Wait no: (2,0,...) and (2,0,...) are the same vector, we only have one of each.
    # We have 11 positions × 2 signs = 22 vectors. Two vectors conflict if they're at the same position with same sign.
    # But we only have one vector per (position, sign) pair, so no two V1 vectors are identical.
    # Two V1 vectors at same position: one is +2, other is -2, dot = -4 ≤ 2, OK.
    # Two V1 vectors at different positions: dot = 0 ≤ 2, OK.
    # So all 22 V1 vectors are mutually compatible!

    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors
