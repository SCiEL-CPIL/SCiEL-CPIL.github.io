# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin before 600s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    grid_sq = (grid ** 2).sum(axis=1).astype(np.float64)

    # Pre-compute all pairwise differences and squared norm differences
    # diff[i,j] = grid[i] - grid[j]  (shape N, N, 3)
    # sq_diff[i,j] = grid_sq[i] - grid_sq[j]  (shape N, N)
    grid_f = grid.astype(np.float64)
    diff_all = grid_f[:, np.newaxis, :] - grid_f[np.newaxis, :, :]  # (N, N, 3)
    sq_diff_all = grid_sq[:, np.newaxis] - grid_sq[np.newaxis, :]  # (N, N)

    # Pre-compute combination indices for m up to a reasonable limit
    # combos_cache[m] is a numpy array of shape (C(m,4), 4)
    max_m = 80
    combos_cache = {}
    for m in range(4, max_m + 1):
        idx = np.arange(m)
        combos = list(itertools.combinations(idx, 4))
        if combos:
            combos_cache[m] = np.array(combos, dtype=np.int32)
        else:
            combos_cache[m] = np.empty((0, 4), dtype=np.int32)

    def can_add_fast(arr_indices, p_idx):
        """
        Fast validity check using precomputed differences.
        arr_indices: list/array of indices of points already in the set.
        p_idx: index of candidate point to add.
        Returns True if adding p_idx keeps the set valid.
        """
        m = len(arr_indices)
        if m < 4:
            return True

        arr_indices_arr = np.array(arr_indices, dtype=np.int64)

        # Extract rows for each existing point vs p_idx
        # row_i = (grid[arr_indices[i]] - grid[p_idx], grid_sq[arr_indices[i]] - grid_sq[p_idx])
        diffs = diff_all[arr_indices_arr, p_idx]  # (m, 3)
        sq_diffs = sq_diff_all[arr_indices_arr, p_idx]  # (m,)
        rows = np.column_stack([diffs, sq_diffs])  # (m, 4)

        if m not in combos_cache:
            idx = np.arange(m)
            combos = list(itertools.combinations(idx, 4))
            if not combos:
                return True
            combo_arr = np.array(combos, dtype=np.int32)
        else:
            combo_arr = combos_cache[m]

        if combo_arr.shape[0] == 0:
            return True

        sub = rows[combo_arr]  # (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_fill(order):
        """Greedy fill: add points in given order if valid."""
        arr_indices = []
        for idx in order:
            if time.time() > deadline - 30:
                break
            if can_add_fast(arr_indices, idx):
                arr_indices.append(idx)
        return arr_indices

    def local_search(arr_indices, max_rounds=100):
        """
        Swap-based local search.
        """
        if len(arr_indices) < 4:
            return arr_indices

        current = list(arr_indices)
        current_set = set(current)
        improved = True
        rounds = 0

        while improved and rounds < max_rounds and time.time() < deadline - 30:
            improved = False
            rounds += 1
            for i in range(len(current)):
                if time.time() > deadline - 30:
                    break
                # Remove point i
                remaining = [x for x in current if x != current[i]]
                
                # Try to find a replacement
                candidates = [j for j in range(N) if j not in set(remaining)]
                rng.shuffle(candidates)
                
                # Limit candidates to check
                for j in candidates[:150]:
                    if can_add_fast(remaining, j):
                        current = remaining + [j]
                        improved = True
                        break
                if improved:
                    break  # restart scan after a swap
        return current

    # --- Main search loop ---
    rng = np.random.default_rng(42)
    best_set = []
    best_size = 0

    # Precompute distance from center for biased sampling
    center = np.array([3.5, 3.5, 3.5])
    dist_from_center = np.sum((grid_f - center) ** 2, axis=1)
    
    # Precompute a few weight vectors for different biases
    weight_corn = dist_from_center + 1e-9
    weight_corn /= weight_corn.sum()
    weight_center = (dist_from_center.max() + 1.0 - dist_from_center) + 1e-9
    weight_center /= weight_center.sum()
    weight_uniform = np.ones(N) / N

    num_restarts = 300

    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break

        # Vary ordering strategy
        r = restart % 5
        if r == 0:
            order = rng.permutation(N)
        elif r == 1:
            order = rng.choice(N, size=N, replace=False, p=weight_corn)
        elif r == 2:
            order = rng.choice(N, size=N, replace=False, p=weight_center)
        elif r == 3:
            # Mix of corner and center bias
            mix = 0.5 * weight_corn + 0.5 * weight_center
            order = rng.choice(N, size=N, replace=False, p=mix)
        else:
            order = rng.permutation(N)

        # Greedy fill
        current = greedy_fill(order)

        # Local search improvement
        if len(current) >= 4:
            current = local_search(current, max_rounds=80)

        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)

        # Simulated Annealing phase
        if len(current) >= 4 and time.time() < deadline - 60:
            sa_set = list(current)
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            sa_steps = 0
            sa_max_steps = 15000

            # Precompute set for O(1) membership
            sa_set_set = set(sa_set)

            while sa_steps < sa_max_steps and time.time() < deadline - 30:
                sa_steps += 1
                move_type = rng.random()

                if move_type < 0.45:
                    # Try add
                    candidates = [i for i in range(N) if i not in sa_set_set]
                    if candidates:
                        move_idx = int(rng.choice(candidates))
                        if can_add_fast(sa_set, move_idx):
                            sa_set.append(move_idx)
                            sa_set_set.add(move_idx)
                            if len(sa_set) > best_size:
                                best_size = len(sa_set)
                                best_set = list(sa_set)
                elif move_type < 0.75:
                    # Try remove
                    if sa_set:
                        move_pos = int(rng.choice(len(sa_set)))
                        move_idx = sa_set[move_pos]
                        # Accept removal with some probability
                        if rng.random() < np.exp(-0.3 / T):
                            sa_set.pop(move_pos)
                            sa_set_set.remove(move_idx)
                else:
                    # Try swap
                    if sa_set and len(sa_set) < N:
                        out_pos = int(rng.choice(len(sa_set)))
                        out_idx = sa_set[out_pos]
                        candidates = [i for i in range(N) if i not in sa_set_set]
                        if candidates:
                            in_idx = int(rng.choice(candidates))
                            remaining = [x for x in sa_set if x != out_idx]
                            if can_add_fast(remaining, in_idx):
                                sa_set[out_pos] = in_idx
                                sa_set_set.discard(out_idx)
                                sa_set_set.add(in_idx)

                T *= cooling
                if T < min_T:
                    T = min_T

    # Convert best set to array
    if best_size > 0:
        result = np.array([grid[i] for i in best_set], dtype=np.int64)
    else:
        result = np.empty((0, 3), dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
