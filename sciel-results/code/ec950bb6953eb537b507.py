import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    M = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # Build per-position list of AP indices and the indices of the OTHER elements in those APs
    pos_aps = [[] for _ in range(n)]
    pos_others = [[] for _ in range(n)]
    
    for idx, ap in enumerate(aps):
        for k in range(4):
            p = ap[k]
            pos_aps[p].append(idx)
            others = [ap[j] for j in range(4) if j != k]
            pos_others[p].append(others)

    # Precompute others as numpy arrays for faster delta computation
    pos_others_arr = [np.array(lst, dtype=np.int32) if lst else np.empty((0,3), dtype=np.int32) for lst in pos_others]

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
        """Calculate delta in mono count if we flip col[pos]."""
        others_arr = pos_others_arr[pos]
        if len(others_arr) == 0:
            return 0
        c1 = col[others_arr[:, 0]]
        c2 = col[others_arr[:, 1]]
        c3 = col[others_arr[:, 2]]
        all_same = (c1 == c2) & (c2 == c3)
        if not np.any(all_same):
            return 0
        same_val = c1[all_same]
        matches = (col[pos] == same_val)
        # For each AP where others are all same:
        # delta contribution: -1 if col[pos]==val, +1 otherwise
        return int(np.sum(np.where(matches, -1, 1)))

    def get_delta_vectorized(col):
        """Calculate delta for all positions at once using vectorized operations."""
        deltas = np.zeros(n, dtype=np.int32)
        for pos in range(n):
            others_arr = pos_others_arr[pos]
            if len(others_arr) == 0:
                continue
            c1 = col[others_arr[:, 0]]
            c2 = col[others_arr[:, 1]]
            c3 = col[others_arr[:, 2]]
            all_same = (c1 == c2) & (c2 == c3)
            if np.any(all_same):
                same_val = c1[all_same]
                matches = (col[pos] == same_val)
                deltas[pos] = np.sum(np.where(matches, -1, 1))
        return deltas

    def greedy_polish(col, max_passes=100):
        """Run greedy local search until no improvement."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        passes = 0
        while improved and passes < max_passes and (time.time() - start_time) < deadline - 5.0:
            improved = False
            passes += 1
            for pos in range(n):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
        return col, cur_count

    def greedy_polish_random_order(col, max_passes=100):
        """Run greedy local search with random position ordering."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        passes = 0
        while improved and passes < max_passes and (time.time() - start_time) < deadline - 5.0:
            improved = False
            passes += 1
            order = list(range(n))
            random.shuffle(order)
            for pos in order:
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

    # Phase 1.5: Try random-order polish from best
    c_ro, count_ro = greedy_polish_random_order(best_c, max_passes=50)
    if count_ro < best_count:
        best_c = c_ro.copy()
        best_count = count_ro

    # Phase 2: Simulated Annealing with improved restarts
    T = 2.0
    cooling = 0.99995
    min_T = 0.03
    no_improve_steps = 0
    max_no_improve = 1500
    restart_count = 0
    best_restart_T = 1.5
    cur_count = count_mono(c)
    best_sa_count = count_mono(best_c)

    while (time.time() - start_time) < deadline - 3.0:
        # Pick random position
        pos = random.randint(0, n - 1)
        d = get_delta(pos, c)

        accept = False
        if d < 0:
            accept = True
            no_improve_steps = 0
        elif d == 0:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(-d / T)
                if random.random() < prob:
                    accept = True
                    no_improve_steps = 0

        if accept:
            c[pos] = 1 - c[pos]
            cur_count += d
            if cur_count < best_sa_count:
                best_c = c.copy()
                best_sa_count = cur_count
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        T *= cooling

        # Restart strategy if stuck
        if no_improve_steps > max_no_improve:
            restart_count += 1
            # Reset to best
            c = best_c.copy()
            cur_count = best_sa_count
            # Adaptive temperature: start lower for later restarts
            T = max(best_restart_T / (1 + 0.3 * restart_count), 0.3)
            no_improve_steps = 0
            # Perturb: flip random positions based on restart count
            num_flips = min(3 + restart_count, 10)
            for _ in range(num_flips):
                p2 = random.randint(0, n - 1)
                c[p2] = 1 - c[p2]
            cur_count = count_mono(c)
            # Polish to get back on track
            c, cur_count = greedy_polish(c, max_passes=20)
            if cur_count < best_sa_count:
                best_c = c.copy()
                best_sa_count = cur_count

    # Phase 3: Final intensive polish with multiple orderings
    best_count = count_mono(best_c)
    
    if (time.time() - start_time) < deadline - 2.0:
        # Sequential polish
        final_c, final_count = greedy_polish(best_c, max_passes=200)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    if (time.time() - start_time) < deadline - 1.0:
        # Random-order polish
        final_c, final_count = greedy_polish_random_order(best_c, max_passes=100)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Phase 4: Multi-start polish - try several random perturbations from best
    if (time.time() - start_time) < deadline - 1.0:
        for attempt in range(5):
            if (time.time() - start_time) > deadline - 1.0:
                break
            perturbed = best_c.copy()
            num_flips = random.randint(2, 6)
            for _ in range(num_flips):
                p = random.randint(0, n - 1)
                perturbed[p] = 1 - perturbed[p]
            polished_c, polished_count = greedy_polish(perturbed, max_passes=50)
            if polished_count < best_count:
                best_c = polished_c
                best_count = polished_count

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
