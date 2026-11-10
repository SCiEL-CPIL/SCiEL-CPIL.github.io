import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 140

    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)  # shape (total_aps, 4)

    # For each position, precompute which APs contain it and the position index
    pos_ap_indices = []
    pos_ap_positions = []
    pos_ap_others = []  # For each position, the other 3 indices in each AP

    for p in range(N):
        indices = []
        positions = []
        others = []
        for i in range(total_aps):
            row = aps[i]
            for j in range(4):
                if row[j] == p:
                    indices.append(i)
                    positions.append(j)
                    others.append([row[k] for k in range(4) if k != j])
                    break
        pos_ap_indices.append(np.array(indices, dtype=np.int32))
        pos_ap_positions.append(np.array(positions, dtype=np.int8))
        pos_ap_others.append(np.array(others, dtype=np.int32))  # shape (n_aps, 3)

    def count_mono(colors):
        c = colors[aps]
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def fast_delta(colors, p):
        old_color = colors[p]
        ap_idxs = pos_ap_indices[p]
        n_aps = len(ap_idxs)
        if n_aps == 0:
            return 0

        others = pos_ap_others[p]  # shape (n_aps, 3)
        other_colors = colors[others]  # shape (n_aps, 3)

        # Before flip: mono if all 3 others have same color AND that color == old_color
        all_same = (other_colors[:, 0] == other_colors[:, 1]) & (other_colors[:, 1] == other_colors[:, 2])
        mono_before = all_same & (other_colors[:, 0] == old_color)

        # After flip: new_color = 1 - old_color
        new_color = 1 - old_color
        mono_after = all_same & (other_colors[:, 0] == new_color)

        return int(mono_after.sum()) - int(mono_before.sum())

    best_colors = None
    best_count = float('inf')

    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()

    # Multiple restarts with different initializations
    num_restarts = 20

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Different initialization strategies
        if restart == 0:
            colors = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 1:
            colors = np.random.randint(0, 2, size=N).astype(np.int8)
        elif restart == 2:
            colors = np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 3:
            colors = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 4:
            colors = np.array([(i // 5) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 5:
            colors = np.array([(i // 6) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 6:
            # Thicker: 000000111111...
            colors = np.array([(i // 7) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 7:
            # Thicker: 0000000011111111...
            colors = np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 8:
            # Thicker: 00000000001111111111...
            colors = np.array([(i // 10) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 9:
            # Thicker: 0000000000000011111111111111...
            colors = np.array([(i // 12) % 2 for i in range(N)], dtype=np.int8)
        else:
            if best_colors is not None:
                colors = best_colors.copy()
                n_flips = max(5, N // 8)
                flip_positions = np.random.choice(N, size=n_flips, replace=False)
                colors[flip_positions] ^= 1
            else:
                colors = np.random.randint(0, 2, size=N).astype(np.int8)

        current_count = count_mono(colors)
        update_best(colors, current_count)

        # Simulated annealing
        T = 2.5
        T_min = 0.03
        cooling = 0.9997
        max_sa_steps = 100000

        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break

            p = np.random.randint(0, N)
            delta = fast_delta(colors, p)

            if delta <= 0:
                colors[p] ^= 1
                current_count += delta
                update_best(colors, current_count)
            else:
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        colors[p] ^= 1
                        current_count += delta

            T = max(T_min, T * cooling)

        # Local search: hill climbing
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for p in range(N):
                if time.time() > deadline - 3:
                    break
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)

        # Second local search pass
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for p in range(N):
                if time.time() > deadline - 2:
                    break
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)

    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]
