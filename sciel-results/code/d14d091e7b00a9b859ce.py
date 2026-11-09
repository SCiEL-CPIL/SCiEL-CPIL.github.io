"""Combined: Vectorized SA with batching + multi-phase refinement."""
import time
import numpy as np
import random
import math


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs as 0-indexed tuples
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)

    # Build per-position list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx in range(M):
        for pos in aps_arr[idx]:
            pos_aps[pos].append(idx)
    pos_aps_arr = [np.array(p, dtype=np.int32) if p else np.array([], dtype=np.int32) for p in pos_aps]

    # Thue-Morse initialization
    colors = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono_vec(cols):
        c = cols[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))

    def compute_mono_flags(cols):
        c = cols[aps_arr]
        return (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])

    current_mono = count_mono_vec(colors)
    best_colors = colors.copy()
    best_mono = current_mono
    mono_flag = compute_mono_flags(colors)

    # Phase 1: Greedy polish
    def greedy_polish(cols, mono_flags, cur_count, time_budget):
        cols = cols.copy()
        mono_flags = mono_flags.copy()
        improved = True
        while improved and (time.time() - start_time) < time_budget:
            improved = False
            for pos in range(n):
                if (time.time() - start_time) > time_budget:
                    break
                ap_indices = pos_aps_arr[pos]
                if len(ap_indices) == 0:
                    continue
                old_mono = mono_flags[ap_indices]
                cols[pos] = 1 - cols[pos]
                new_colors = cols[aps_arr[ap_indices]]
                new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                           (new_colors[:, 1] == new_colors[:, 2]) & \
                           (new_colors[:, 2] == new_colors[:, 3])
                delta = int(np.sum(new_mono) - np.sum(old_mono))
                if delta < 0:
                    mono_flags[ap_indices] = new_mono
                    cur_count += delta
                    improved = True
                else:
                    cols[pos] = 1 - cols[pos]
        return cols, mono_flags, cur_count

    colors, mono_flag, current_mono = greedy_polish(colors, mono_flag, current_mono, deadline - 100.0)
    if current_mono < best_mono:
        best_mono = current_mono
        best_colors = colors.copy()

    # Phase 2: Batched Simulated Annealing with restarts
    rng = np.random.RandomState(42)
    T = 2.0
    min_T = 0.05
    cooling = 0.9997
    batch_size = 16  # flips per temperature step

    while (time.time() - start_time) < deadline - 5.0:
        # Batch of random positions
        positions = rng.randint(0, n, size=batch_size)
        batch_delta = 0
        accepted_positions = []
        accepted_ap_indices = []
        accepted_new_mono = []
        accepted_old_mono = []

        for pos in positions:
            ap_indices = pos_aps_arr[pos]
            if len(ap_indices) == 0:
                continue
            old_mono = mono_flag[ap_indices]
            colors[pos] = 1 - colors[pos]
            new_colors = colors[aps_arr[ap_indices]]
            new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                       (new_colors[:, 1] == new_colors[:, 2]) & \
                       (new_colors[:, 2] == new_colors[:, 3])
            d = int(np.sum(new_mono) - np.sum(old_mono))

            if d <= 0 or (T > 0.1 and rng.rand() < math.exp(-d / T)):
                batch_delta += d
                accepted_positions.append(pos)
                accepted_ap_indices.append(ap_indices)
                accepted_new_mono.append(new_mono)
                accepted_old_mono.append(old_mono)
            else:
                colors[pos] = 1 - colors[pos]

        # Apply all accepted flips at once
        if accepted_positions:
            for i, pos in enumerate(accepted_positions):
                mono_flag[accepted_ap_indices[i]] = accepted_new_mono[i]
            current_mono += batch_delta
            if current_mono < best_mono:
                best_mono = current_mono
                best_colors = colors.copy()

        T *= cooling
        if T < min_T:
            # Restart with perturbation
            T = 1.5
            for _ in range(10):
                p2 = rng.randint(n)
                ap_indices = pos_aps_arr[p2]
                if len(ap_indices) == 0:
                    continue
                old_mono = mono_flag[ap_indices]
                colors[p2] = 1 - colors[p2]
                new_colors = colors[aps_arr[ap_indices]]
                new_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                           (new_colors[:, 1] == new_colors[:, 2]) & \
                           (new_colors[:, 2] == new_colors[:, 3])
                d = int(np.sum(new_mono) - np.sum(old_mono))
                if d <= 0 or rng.rand() < 0.2:
                    mono_flag[ap_indices] = new_mono
                    current_mono += d
                    if current_mono < best_mono:
                        best_mono = current_mono
                        best_colors = colors.copy()
                else:
                    colors[p2] = 1 - colors[p2]

    # Phase 3: Final greedy polish
    if (time.time() - start_time) < deadline - 2.0:
        colors, mono_flag, current_mono = greedy_polish(best_colors, compute_mono_flags(best_colors), best_mono, deadline - 1.0)
        if current_mono < best_mono:
            best_mono = current_mono
            best_colors = colors.copy()

    # Verify
    actual = count_mono_vec(best_colors)
    if actual < best_mono:
        best_mono = actual

    return [int(c) for c in best_colors]
