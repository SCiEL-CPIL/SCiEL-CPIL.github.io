import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type D: Vectors with four non-zero entries +/- 1
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)

    # Build support masks using bit operations
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)

    # Precompute conflict matrix: two 4-subsets conflict if they share >= 3 elements
    # For 4-subsets, sharing 3 means they differ in exactly 1 element
    # Use popcount to check intersection size
    conflict = np.zeros((N_S, N_S), dtype=bool)

    # For each pair, compute intersection size via bit AND and popcount
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = support_masks[i] & support_masks[j]
            # Count bits set in inter
            popcount = bin(inter).count('1')
            if popcount >= 3:
                conflict[i, j] = True
                conflict[j, i] = True

    # Build adjacency list for faster lookups
    adj = [np.where(conflict[i])[0] for i in range(N_S)]

    # Degree of each node
    degrees = np.array([len(adj[i]) for i in range(N_S)], dtype=np.int64)

    rng = np.random.RandomState(42)
    py_rng = random.Random(42)

    best_support_set = set()
    best_size = 0
    best_support_list = []

    # Greedy with minimum degree selection
    def greedy_min_degree():
        x = np.zeros(N_S, dtype=bool)
        remaining = set(range(N_S))
        current_degrees = degrees.copy()

        while remaining:
            if time.time() > deadline - 30:
                break
            # Find node with minimum degree among remaining
            min_deg = min(current_degrees[i] for i in remaining)
            candidates = [i for i in remaining if current_degrees[i] == min_deg]
            # Pick randomly from candidates for diversity
            v = py_rng.choice(candidates)
            x[v] = True
            remaining.discard(v)
            # Remove neighbors
            for nb in adj[v]:
                if nb in remaining:
                    remaining.discard(nb)
                    for nb2 in adj[nb]:
                        if nb2 in remaining:
                            current_degrees[nb2] -= 1
            for nb in adj[v]:
                current_degrees[nb] -= 1

        return x

    # Simulated Annealing - vectorized
    T0 = 1.0
    T_end = 0.005
    num_iterations = 50000
    cooling = (T_end / T0) ** (1.0 / num_iterations)

    max_restarts = 150

    for restart in range(max_restarts):
        if time.time() > deadline - 35:
            break

        # Initialize with greedy min-degree
        x = greedy_min_degree()
        current_size = int(np.sum(x))

        best_x = x.copy()
        best_current_size = current_size

        T = T0
        for it in range(num_iterations):
            if time.time() > deadline - 35:
                break

            in_set = np.where(x)[0]
            not_in_set = np.where(~x)[0]

            # Compute blocked nodes (neighbors of in-set)
            if len(in_set) > 0:
                blocked = np.zeros(N_S, dtype=bool)
                blocked[in_set] = True
                for v in in_set:
                    blocked[adj[v]] = True
            else:
                blocked = np.zeros(N_S, dtype=bool)

            add_candidates = np.where(~x & ~blocked)[0]
            remove_candidates = in_set

            if len(add_candidates) == 0 and len(remove_candidates) == 0:
                break

            move_type = py_rng.random()
            if move_type < 0.45 and len(add_candidates) > 0:
                v = py_rng.choice(add_candidates)
                x[v] = True
                current_size += 1
                if current_size > best_current_size:
                    best_current_size = current_size
                    best_x = x.copy()
            elif move_type < 0.75 and len(remove_candidates) > 0:
                v = py_rng.choice(remove_candidates)
                x[v] = False
                current_size -= 1
                if py_rng.random() >= np.exp(-1.0 / T):
                    x[v] = True
                    current_size += 1
            else:
                # Swap
                if len(add_candidates) > 0 and len(remove_candidates) > 0:
                    v_remove = py_rng.choice(remove_candidates)
                    v_add = py_rng.choice(add_candidates)
                    if v_remove != v_add:
                        # Check if v_add conflicts with any in-set node except v_remove
                        in_set_no_remove = [v for v in in_set if v != v_remove]
                        if len(in_set_no_remove) > 0:
                            if np.any(conflict[v_add, in_set_no_remove]):
                                T *= cooling
                                continue
                        x[v_remove] = False
                        x[v_add] = True

            T *= cooling

        if best_current_size > best_size:
            best_size = best_current_size
            best_support_set = set(np.where(best_x)[0].tolist())
            best_support_list = list(best_support_set)

    # Generate Type D vectors for the selected supports
    selected_vectors_D = []
    for s_idx in best_support_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(support, signs):
                v[idx] = sign
            selected_vectors_D.append(v)
    selected_vectors_D = np.array(selected_vectors_D, dtype=np.int64)

    # Greedy fill: try to add more Type D vectors from non-selected supports
    current_set_vectors = list(selected_vectors_D)
    current_set_vectors_set = set(map(tuple, current_set_vectors))
    current_arr = np.array(current_set_vectors, dtype=np.int64)

    # Generate ALL Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)

    indices_all = np.arange(N_all_D)
    rng.shuffle(indices_all)

    for i in indices_all:
        if time.time() > deadline - 10:
            break
        v = all_vectors_D[i]
        if tuple(v) in current_set_vectors_set:
            continue
        dots = current_arr @ v
        if np.any(dots >= 3):
            continue
        current_set_vectors.append(v)
        current_set_vectors_set.add(tuple(v))
        current_arr = np.vstack([current_arr, v.reshape(1, -1)])

    final_vectors_D = np.array(current_set_vectors, dtype=np.int64)
    final_vectors = np.vstack([vectors_A, final_vectors_D])
    return final_vectors
