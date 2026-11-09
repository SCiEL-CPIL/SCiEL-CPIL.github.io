import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
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
        flipped = colors.copy()
        rows = np.arange(K)
        flipped[rows, col_idx] = 1 - flipped[rows, col_idx]
        new_mono = (flipped[:, 0] == flipped[:, 1]) & (flipped[:, 1] == flipped[:, 2]) & (flipped[:, 2] == flipped[:, 3])
        new_mono_count = int(np.sum(new_mono))
        return new_mono_count - cur_mono_count

    def greedy_polish(col, max_iters=200):
        """Run greedy local search until no improvement or max_iters reached."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        iters = 0
        while improved and iters < max_iters and (time.time() - start_time) < deadline - 5.0:
            improved = False
            iters += 1
            # Shuffle positions to avoid bias towards low indices
            order = np.random.permutation(n)
            for pos in order:
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
        return col, cur_count

    def sa_polish(col, max_iters=300):
        """Simulated annealing polish: accept worse moves with probability exp(-delta/T)."""
        col = col.copy()
        cur_count = count_mono(col)
        T = 3.0
        T_min = 0.01
        cooling = 0.995
        improved = True
        iters = 0
        while iters < max_iters and (time.time() - start_time) < deadline - 5.0:
            iters += 1
            T *= cooling
            if T < T_min:
                T = T_min
            order = np.random.permutation(n)
            any_change = False
            for pos in order:
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    any_change = True
                elif d > 0 and random.random() < math.exp(-d / T):
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    any_change = True
            if not any_change:
                break
        return col, cur_count

    # Phase 1: Generate diverse initial solutions and polish them
    best_c = None
    best_count = float('inf')

    # 1. Thue-Morse
    c_tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    c_polished, count = greedy_polish(c_tm)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # 2. Random
    for _ in range(5):
        c_rand = np.random.randint(0, 2, n).astype(np.int8)
        c_polished, count = greedy_polish(c_rand)
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()

    # 3. Block patterns
    for block_size in [2, 4, 8, 16, 32]:
        c_block = np.zeros(n, dtype=np.int8)
        for i in range(n):
            c_block[i] = (i // block_size) % 2
        c_polished, count = greedy_polish(c_block)
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()

    # 4. Rademacher-like: based on parity of floor(log2(i))
    c_rad = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c_rad[i] = int(math.log2(i + 1)) % 2
    c_polished, count = greedy_polish(c_rad)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # 5. Walsh-Hadamard row (first non-trivial)
    c_walsh = np.array([1 if (i & 1) else 0 for i in range(n)], dtype=np.int8)
    c_polished, count = greedy_polish(c_walsh)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # Phase 2: Iterated Local Search (ILS) with SA
    # We will repeatedly perturb the best solution and polish it.
    
    base_perturb = 3
    max_perturb = 12
    
    while (time.time() - start_time) < deadline - 10.0:
        # Perturb the best solution
        c_current = best_c.copy()
        
        # Randomly decide how many bits to flip
        time_elapsed = time.time() - start_time
        time_remaining = deadline - 10.0 - time_elapsed
        if time_remaining < 10.0:
            num_flips = random.randint(5, 12)
        else:
            num_flips = random.randint(base_perturb, max_perturb)
        
        # Flip random positions
        positions_to_flip = np.random.choice(n, num_flips, replace=False)
        for p in positions_to_flip:
            c_current[p] = 1 - c_current[p]
        
        # Polish the perturbed solution - use SA for better escape
        c_polished, count = sa_polish(c_current)
        
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()
            # Do a final greedy polish on the improved solution
            c_polished2, count2 = greedy_polish(c_polished)
            if count2 < best_count:
                best_count = count2
                best_c = c_polished2.copy()

    # Phase 3: Final Polish with both methods
    if (time.time() - start_time) < deadline - 3.0:
        final_c, final_count = greedy_polish(best_c)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    if (time.time() - start_time) < deadline - 2.0:
        final_c2, final_count2 = sa_polish(best_c)
        if final_count2 < best_count:
            best_c = final_c2
            best_count = final_count2

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
