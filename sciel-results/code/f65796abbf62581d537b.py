import time
import random
import numpy as np
from scipy.optimize import differential_evolution

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_len = max_sum - min_sum + 1  # 129

    # Precompute diagonal correction vectors for each target sum
    # diag_vec[s - min_sum] has 1 at position s//2 if s is even
    diag_vecs = []
    for s in range(min_sum, max_sum + 1):
        v = np.zeros(n + 1, dtype=np.float64)
        if s % 2 == 0:
            mid = s // 2
            if 0 <= mid <= n:
                v[mid] = 1.0
        diag_vecs.append(v)
    diag_matrix = np.array(diag_vecs)  # shape (129, 129)

    def fast_objective(x_bin):
        """
        Vectorized objective using convolution.
        x_bin is a binary array of length 129.
        Returns (peak, uncovered_count).
        """
        if x_bin.sum() == 0:
            return float('inf'), 129

        conv = np.convolve(x_bin, x_bin, mode='full')
        counts = conv[min_sum:max_sum + 1].astype(np.float64)

        # diag_correction[i] = x_bin[s//2] if s even else 0
        diag_correction = diag_matrix @ x_bin.astype(np.float64)

        unordered_counts = (counts + diag_correction) / 2.0
        unordered_counts = np.round(unordered_counts).astype(np.int32)

        uncovered = int(np.sum(unordered_counts == 0))
        if uncovered > 0:
            return float('inf'), uncovered

        peak = int(np.max(unordered_counts))
        return peak, 0

    def compute_peak_only(x_bin):
        """Faster version that just returns peak (assumes coverage)."""
        conv = np.convolve(x_bin, x_bin, mode='full')
        counts = conv[min_sum:max_sum + 1].astype(np.float64)
        diag_correction = diag_matrix @ x_bin.astype(np.float64)
        unordered_counts = (counts + diag_correction) / 2.0
        unordered_counts = np.round(unordered_counts).astype(np.int32)
        if np.any(unordered_counts == 0):
            return float('inf')
        return int(np.max(unordered_counts))

    # Phase 1: Greedy initialization - start with symmetric structure
    # Use a two-interval approach: [0, a] U [b, 128]
    best_peak = float('inf')
    best_A = None

    for a in range(0, 70):
        for b in range(60, 129):
            if a >= b:
                continue
            A_set = set(range(0, a + 1)) | set(range(b, 129))
            x_bin = np.zeros(n + 1, dtype=np.int8)
            for x in A_set:
                x_bin[x] = 1
            peak, unc = fast_objective(x_bin)
            if peak < best_peak:
                best_peak = peak
                best_A = A_set.copy()

    if best_A is None:
        best_A = set(range(0, 70))
        best_peak = 100

    # Phase 2: Greedy refinement - add/remove to fix gaps and reduce peak
    x_cur = np.zeros(n + 1, dtype=np.int8)
    for x in best_A:
        x_cur[x] = 1

    # Greedy: try adding elements to reduce peak
    for _ in range(200):
        if time.time() > deadline - 30:
            break
        cur_peak, cur_unc = fast_objective(x_cur)
        if cur_unc > 0:
            # Need to add elements to cover gaps
            conv = np.convolve(x_cur, x_cur, mode='full')
            counts = conv[min_sum:max_sum + 1].astype(np.float64)
            diag_correction = diag_matrix @ x_cur.astype(np.float64)
            unordered_counts = (counts + diag_correction) / 2.0
            unordered_counts = np.round(unordered_counts).astype(np.int32)
            gaps = np.where(unordered_counts == 0)[0]
            if len(gaps) == 0:
                break
            # Try adding each zero element, pick best
            best_add_peak = cur_peak if cur_peak != float('inf') else 1000
            best_add_idx = -1
            zeros = np.where(x_cur == 0)[0]
            for idx in zeros:
                x_test = x_cur.copy()
                x_test[idx] = 1
                p, u = fast_objective(x_test)
                if u == 0 and p < best_add_peak:
                    best_add_peak = p
                    best_add_idx = idx
            if best_add_idx >= 0:
                x_cur[best_add_idx] = 1
            else:
                break
        else:
            # Try removing elements to reduce peak
            best_rem_peak = cur_peak
            best_rem_idx = -1
            ones = np.where(x_cur == 1)[0]
            for idx in ones:
                x_test = x_cur.copy()
                x_test[idx] = 0
                p, u = fast_objective(x_test)
                if u == 0 and p < best_rem_peak:
                    best_rem_peak = p
                    best_rem_idx = idx
            if best_rem_idx >= 0:
                x_cur[best_rem_idx] = 0
            else:
                break

    # Update best
    peak_greedy, unc_greedy = fast_objective(x_cur)
    if unc_greedy == 0 and peak_greedy < best_peak:
        best_peak = peak_greedy
        best_A = set(np.where(x_cur > 0)[0].tolist())

    # Phase 3: Differential Evolution on binary vector
    # Use a penalty-based objective for DE
    def de_objective(x_float):
        """Objective for differential evolution. x_float is in [0,1]^129."""
        # Round to binary
        x_bin = (x_float > 0.5).astype(np.int8)
        if x_bin.sum() == 0:
            return 1e6
        conv = np.convolve(x_bin, x_bin, mode='full')
        counts = conv[min_sum:max_sum + 1].astype(np.float64)
        diag_correction = diag_matrix @ x_bin.astype(np.float64)
        unordered_counts = (counts + diag_correction) / 2.0
        unordered_counts = np.round(unordered_counts).astype(np.int32)
        uncovered = int(np.sum(unordered_counts == 0))
        if uncovered > 0:
            return 1000 + uncovered * 10
        peak = int(np.max(unordered_counts))
        return float(peak)

    # Initial population from best found
    x_init = np.zeros(n + 1, dtype=np.float64)
    for x in best_A:
        x_init[x] = 1.0

    bounds = [(0, 1)] * (n + 1)

    # Run DE with limited budget
    de_start = time.time()
    try:
        result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=0.1,
            mutation=(0.3, 0.8),
            recombination=0.7,
            polish=False,
            init='so',
            x0=x_init,
            updating='immediate',
            callback=None
        )
    except Exception:
        result = None

    if result is not None and result.fun < 1000:
        x_de = (result.x > 0.5).astype(np.int8)
        peak_de, unc_de = fast_objective(x_de)
        if unc_de == 0 and peak_de < best_peak:
            best_peak = peak_de
            best_A = set(np.where(x_de > 0)[0].tolist())

    # Phase 4: Simulated annealing refinement
    random.seed(42)
    np.random.seed(42)

    x_current = np.zeros(n + 1, dtype=np.int8)
    for x in best_A:
        x_current[x] = 1

    x_best = x_current.copy()
    best_peak_sa = best_peak

    T_start = 5.0
    T_end = 0.05
    cooling_rate = 0.9995
    T = T_start

    while time.time() < deadline - 5 and T > T_end:
        if time.time() > deadline - 7:
            break

        move_type = random.random()
        x_proposed = x_current.copy()

        if move_type < 0.3:
            # Add an element
            candidates = np.where(x_current == 0)[0]
            if len(candidates) > 0:
                idx = random.choice(candidates)
                x_proposed[idx] = 1
        elif move_type < 0.6:
            # Remove an element
            if x_current.sum() > 15:
                candidates = np.where(x_current == 1)[0]
                if len(candidates) > 0:
                    idx = random.choice(candidates)
                    x_proposed[idx] = 0
        else:
            # Swap
            if x_current.sum() > 15:
                ones = np.where(x_current == 1)[0]
                zeros = np.where(x_current == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    idx_remove = random.choice(ones)
                    idx_add = random.choice(zeros)
                    x_proposed[idx_remove] = 0
                    x_proposed[idx_add] = 1

        peak_prop, unc_prop = fast_objective(x_proposed)
        peak_cur, unc_cur = fast_objective(x_current)

        obj_cur = peak_cur + 1000 * unc_cur
        obj_prop = peak_prop + 1000 * unc_prop

        delta_obj = obj_prop - obj_cur

        if delta_obj < 0 or (T > 0 and random.random() < np.exp(-delta_obj / T)):
            x_current = x_proposed
            if peak_prop < best_peak_sa and unc_prop == 0:
                best_peak_sa = peak_prop
                x_best = x_proposed.copy()

        T *= cooling_rate

    # Phase 5: Final local search polish
    x_cur = x_best.copy()
    cur_peak, _ = fast_objective(x_cur)
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 2 and no_improve < max_no_improve:
        move_type = random.random()
        x_new = x_cur.copy()

        if move_type < 0.35:
            candidates = np.where(x_cur == 0)[0]
            if len(candidates) == 0:
                break
            idx = random.choice(candidates)
            x_new[idx] = 1
        elif move_type < 0.65:
            if x_cur.sum() <= 15:
                no_improve += 1
                continue
            candidates = np.where(x_cur == 1)[0]
            if len(candidates) == 0:
                break
            idx = random.choice(candidates)
            x_new[idx] = 0
        else:
            if x_cur.sum() <= 15:
                no_improve += 1
                continue
            ones = np.where(x_cur == 1)[0]
            zeros = np.where(x_cur == 0)[0]
            if len(ones) == 0 or len(zeros) == 0:
                break
            idx_rem = random.choice(ones)
            idx_add = random.choice(zeros)
            x_new[idx_rem] = 0
            x_new[idx_add] = 1

        new_peak, new_unc = fast_objective(x_new)
        if new_unc == 0 and new_peak < cur_peak:
            x_cur = x_new
            cur_peak = new_peak
            no_improve = 0
        else:
            no_improve += 1

    final_A = set(np.where(x_cur > 0)[0].tolist())
    if len(final_A) == 0:
        final_A = best_A

    return sorted(final_A)
