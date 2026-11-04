import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    D = 11

    # Type A: 22 vectors (one entry +/-2)
    vecs_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vecs_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vecs_A.append(v2)
    vecs_A = np.array(vecs_A, dtype=np.int8)

    # Type D: all norm-2 integer vectors with four +/-1 entries
    # C(11,4) * 16 = 330 * 16 = 5280 vectors
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Generate all Type D vectors
    all_vecs = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vecs.append(v)
    all_vecs = np.array(all_vecs, dtype=np.int8)
    N_total = len(all_vecs)

    # Key insight: Two Type D vectors are compatible iff their dot product <= 2.
    # Dot product = sum over shared positions of (sign_i * sign_j).
    # If supports share k positions, dot ranges from -k to k.
    # Compatible iff dot <= 2.
    # k=0,1,2: always compatible.
    # k=3: compatible iff dot != 3 (i.e., not all 3 signs agree).
    # k=4 (same support): compatible iff dot != 4 (i.e., not all 4 signs agree).

    # Strategy: Use a fast greedy with random restarts.
    # For speed, precompute the support of each vector and use support-based conflict detection.

    # Precompute support masks
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # For each vector, its support index
    vec_support_idx = np.repeat(np.arange(N_S), 16)

    # Precompute popcount
    popcount = np.zeros(2048, dtype=np.int8)
    for i in range(2048):
        popcount[i] = bin(i).count('1')

    # For fast conflict checking, precompute for each pair of supports with intersection size 3 or 4,
    # the sign patterns that conflict.

    # Actually, let me just use a vectorized greedy with batching.

    def greedy_select(rng, batch_size=200):
        """
        Greedy: process vectors in random order, add if compatible.
        Use vectorized dot product for speed.
        """
        order = np.arange(N_total)
        rng.shuffle(order)

        selected_indices = []
        selected_vecs = np.empty((0, D), dtype=np.int8)

        for batch_start in range(0, N_total, batch_size):
            if time.time() > deadline - 15:
                break
            batch_end = min(batch_start + batch_size, N_total)
            batch = order[batch_start:batch_end]
            batch_vecs = all_vecs[batch]

            if len(selected_vecs) == 0:
                selected_indices.extend(batch.tolist())
                selected_vecs = batch_vecs.copy()
                continue

            # Compute dot products: (batch_size, n_selected)
            dots = batch_vecs @ selected_vecs.T

            # Compatible if all dot products <= 2
            compatible = np.all(dots <= 2, axis=1)

            new_vecs = batch_vecs[compatible]
            new_indices = batch[compatible]

            if len(new_vecs) > 0:
                selected_indices.extend(new_indices.tolist())
                selected_vecs = np.vstack([selected_vecs, new_vecs])

        return selected_indices, selected_vecs

    # Run multiple random restarts
    rng = np.random.RandomState(42)
    best_count = 0
    best_selected = None

    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 25:
            break
        sel_idx, sel_vecs = greedy_select(rng)
        count = len(sel_idx)
        if count > best_count:
            best_count = count
            best_selected = sel_vecs.copy()

    # Local search: try to add more vectors
    if best_selected is not None and len(best_selected) > 0:
        # Find vectors not in the selected set
        selected_set = set(best_selected.tolist())
        remaining = [i for i in range(N_total) if i not in selected_set]
        remaining = np.array(remaining, dtype=np.int32)
        rng.shuffle(remaining)

        # Try to add remaining vectors
        for idx in remaining:
            if time.time() > deadline - 10:
                break
            v = all_vecs[idx]
            dot = best_selected @ v
            if np.all(dot <= 2):
                best_selected = np.vstack([best_selected, v.reshape(1, D)])

    # Final result: Type A + selected Type D
    if best_selected is not None and len(best_selected) > 0:
        final_vectors = np.vstack([vecs_A, best_selected])
    else:
        final_vectors = vecs_A

    return final_vectors
