import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)  # shape (M, 4)
    M = len(aps_arr)

    # Build per-position list of AP indices for delta computation
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    pos_aps_arr = [np.array(lst, dtype=np.int32) for lst in pos_aps]

    # Precompute for each position, which column (0-3) it occupies in each affected AP
    # pos_col[pos] = array of column indices where pos appears in each affected AP
    pos_col = []
    for pos in range(n):
        cols = []
        ap_indices = pos_aps_arr[pos]
        if len(ap_indices) > 0:
            affected = aps_arr[ap_indices]  # shape (K, 4)
            # Find column where each row equals pos
            col_arr = np.argmax(affected == pos, axis=1)
            cols.append(col_arr)
        pos_col.append(cols[0] if cols else np.array([], dtype=np.int32))

    # Thue-Morse initialization: c[i] = popcount(i) % 2
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(pos, col):
        """Vectorized delta computation for flipping position pos."""
        ap_indices = pos_aps_arr[pos]
        K = len(ap_indices)
        if K == 0:
            return 0
        affected = aps_arr[ap_indices]  # shape (K, 4)
        colors = col[affected]  # shape (K, 4)
        col_idx = pos_col[pos]  # shape (K,)
        # Current mono status
        cur_mono = (colors[:, 0] == colors[:, 1]) & (colors[:, 1] == colors[:, 2]) & (colors[:, 2] == colors[:, 3])
        cur_mono_count = int(np.sum(cur_mono))
        # After flip: flip the specific column in each row
        # Use advanced indexing to flip
        flipped = colors.copy()
        rows = np.arange(K)
        flipped[rows, col_idx] = 1 - flipped[rows, col_idx]
        new_mono = (flipped[:, 0] == flipped[:, 1]) & (flipped[:, 1] == flipped[:, 2]) & (flipped[:, 2] == flipped[:, 3])
        new_mono_count = int(np.sum(new_mono))
        return new_mono_count - cur_mono_count

    def greedy_polish(col):
        """Run greedy local search until no improvement."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        while improved and (time.time() - start_time) < deadline - 5.0:
            improved = False
            for pos in range(n):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
        return col, cur_count

    # Phase 1: Greedy polish from Thue-Morse
    c, best_count = greedy_polish(c)
    best_c = c.copy()

    # Phase 2: Simulated annealing with restarts to escape local minima
    T = 1.0
    cooling = 0.9998
    min_T = 0.01
    stuck_counter = 0
    no_improve_steps = 0
    max_no_improve = 3000

    while (time.time() - start_time) < deadline - 3.0:
        # Pick random position
        pos = random.randint(0, n - 1)
        d = get_delta(pos, c)

        # Acceptance criterion
        accept = False
        if d < 0:
            accept = True
            stuck_counter = 0
            no_improve_steps = 0
        elif d == 0:
            accept = True
        else:
            if T > 0.05:
                prob = math.exp(-d / T)
            else:
                prob = 0.0
            if random.random() < prob:
                accept = True
                stuck_counter = 0

        if accept:
            c[pos] = 1 - c[pos]
            best_count += d
            if best_count < count_mono(best_c):
                best_c = c.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            stuck_counter += 1
            no_improve_steps += 1

        # Cooling
        T *= cooling
        if T < min_T:
            # Restart with perturbation
            T = 1.0
            stuck_counter = 0
            # Perturb: flip several random positions
            for _ in range(8):
                p2 = random.randint(0, n - 1)
                d2 = get_delta(p2, c)
                if d2 <= 0 or (T > 0.1 and random.random() < 0.1):
                    c[p2] = 1 - c[p2]
                    best_count += d2
                    if best_count < count_mono(best_c):
                        best_c = c.copy()
                    no_improve_steps = 0

        # If stuck too long, do a larger perturbation
        if stuck_counter > 2000:
            T = 1.5
            stuck_counter = 0
            for _ in range(12):
                p2 = random.randint(0, n - 1)
                d2 = get_delta(p2, c)
                if d2 <= 0 or random.random() < 0.15:
                    c[p2] = 1 - c[p2]
                    best_count += d2
                    if best_count < count_mono(best_c):
                        best_c = c.copy()
                    no_improve_steps = 0

        # If no improvement for a long time, restart from best_c with perturbation
        if no_improve_steps > max_no_improve:
            c = best_c.copy()
            best_count = count_mono(c)
            T = 1.0
            no_improve_steps = 0
            # Perturb
            for _ in range(6):
                p2 = random.randint(0, n - 1)
                d2 = get_delta(p2, c)
                if d2 <= 0 or random.random() < 0.1:
                    c[p2] = 1 - c[p2]
                    best_count += d2

    # Phase 3: Final greedy polish from best_c
    if (time.time() - start_time) < deadline - 2.0:
        final_c, final_count = greedy_polish(best_c)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
