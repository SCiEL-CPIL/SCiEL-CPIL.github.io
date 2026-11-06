import numpy as np
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85

    # Precompute all arithmetic progressions as index lists (0-based)
    all_progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                all_progs.append(idx)
    Np_total = len(all_progs)

    # Select critical progressions for fast SA
    critical_ids = set()
    for pid, idx in enumerate(all_progs):
        k = len(idx)
        if k < 5:
            continue
        s = idx[1] - idx[0] if k > 1 else 1
        if s == 1:
            critical_ids.add(pid)
        elif k >= 10 and s <= 5:
            critical_ids.add(pid)
        elif k >= 20:
            critical_ids.add(pid)
    critical_pids = list(critical_ids)
    Np_crit = len(critical_pids)

    # Build membership lists for incremental updates
    # For each position i, list of prog indices that contain it
    pos_to_crit = [[] for _ in range(n)]
    pos_to_full = [[] for _ in range(n)]
    for j, pid in enumerate(critical_pids):
        for i in all_progs[pid]:
            pos_to_crit[i].append(j)
    for pid in range(Np_total):
        for i in all_progs[pid]:
            pos_to_full[i].append(pid)

    # Precompute the full set membership for evaluation
    # We'll use a different approach: precompute for each position, the list of all progs it belongs to

    rng = np.random.RandomState(42)

    def compute_sums_from_signs(signs, progs, pos_to):
        """Compute sums for all progs given signs, using pos_to mapping."""
        sums = np.zeros(len(progs), dtype=np.float32)
        for i in range(n):
            s_val = signs[i]
            for pidx in pos_to[i]:
                sums[pidx] += s_val
        return sums

    def max_abs(sums):
        return float(np.max(np.abs(sums))) if len(sums) > 0 else 0.0

    # Phase 1: Multi-restart SA on critical set with histogram-based max tracking
    best_signs = None
    best_full_score = np.inf

    n_restarts = 8
    iters_per_restart = 15000

    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break

        signs = rng.choice([-1, 1], size=n).astype(np.float32)
        sums = np.zeros(Np_crit, dtype=np.float32)
        for i in range(n):
            for pidx in pos_to_crit[i]:
                sums[pidx] += signs[i]

        current_score = max_abs(sums)

        # Histogram-based max tracking: maintain counts of abs(sums) values
        # Since sums are integers (signs are +/-1, progs have integer lengths),
        # sums are integers. Range: -len(prog) to len(prog), so abs in [0, max_len]
        max_len = max(len(all_progs[pid]) for pid in critical_pids)
        hist = np.zeros(max_len + 1, dtype=np.int32)
        for v in sums:
            hist[abs(int(v))] += 1

        def get_max_from_hist(hist):
            for m in range(len(hist) - 1, -1, -1):
                if hist[m] > 0:
                    return m
            return 0

        current_score = get_max_from_hist(hist)

        T_start = 5.0
        T_end = 0.01

        for it in range(iters_per_restart):
            if time.time() > deadline - 15:
                break

            T = T_start * (1.0 - it / float(iters_per_restart)) + T_end

            i = rng.randint(n)
            sign_i = signs[i]

            # Flip position i: affects all progs containing i
            affected = pos_to_crit[i]
            if len(affected) == 0:
                continue

            delta = -2.0 * sign_i

            # Update sums and histogram
            new_sums = sums.copy()
            new_hist = hist.copy()

            for pidx in affected:
                old_val = sums[pidx]
                new_val = old_val + delta
                new_sums[pidx] = new_val
                new_hist[abs(int(old_val))] -= 1
                new_hist[abs(int(new_val))] += 1

            new_score = get_max_from_hist(new_hist)
            delta_score = new_score - current_score

            if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
                signs[i] = -sign_i
                sums = new_sums
                hist = new_hist
                current_score = new_score

                if best_signs is None:
                    best_signs = signs.copy()

        # Evaluate this candidate on full set
        if best_signs is not None:
            full_sums = np.zeros(Np_total, dtype=np.float32)
            for i in range(n):
                for pidx in pos_to_full[i]:
                    full_sums[pidx] += signs[i]
            full_score = max_abs(full_sums)
            if full_score < best_full_score:
                best_full_score = full_score
                best_signs = signs.copy()

    # Phase 2: Polish using full set with incremental updates
    if best_signs is not None:
        signs = best_signs.copy()
    else:
        signs = rng.choice([-1, 1], size=n).astype(np.float32)

    full_sums = np.zeros(Np_total, dtype=np.float32)
    for i in range(n):
        for pidx in pos_to_full[i]:
            full_sums[pidx] += signs[i]

    current_score = max_abs(full_sums)

    # Build histogram for full set
    max_len_full = max(len(p) for p in all_progs)
    hist_full = np.zeros(max_len_full + 1, dtype=np.int32)
    for v in full_sums:
        hist_full[abs(int(v))] += 1

    def get_max_full(hist):
        for m in range(len(hist) - 1, -1, -1):
            if hist[m] > 0:
                return m
        return 0

    current_score = get_max_full(hist_full)

    n_iters_full = 8000
    T_start = 2.0
    T_end = 0.01

    for it in range(n_iters_full):
        if time.time() > deadline - 3:
            break

        T = T_start * (1.0 - it / float(n_iters_full)) + T_end

        i = rng.randint(n)
        sign_i = signs[i]

        affected = pos_to_full[i]
        if len(affected) == 0:
            continue

        delta = -2.0 * sign_i

        new_sums = full_sums.copy()
        new_hist = hist_full.copy()

        for pidx in affected:
            old_val = full_sums[pidx]
            new_val = old_val + delta
            new_sums[pidx] = new_val
            new_hist[abs(int(old_val))] -= 1
            new_hist[abs(int(new_val))] += 1

        new_score = get_max_full(new_hist)
        delta_score = new_score - current_score

        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            signs[i] = -sign_i
            full_sums = new_sums
            hist_full = new_hist
            current_score = new_score

    # Final evaluation
    final_sums = np.zeros(Np_total, dtype=np.float32)
    for i in range(n):
        for pidx in pos_to_full[i]:
            final_sums[pidx] += signs[i]
    final_score = max_abs(final_sums)

    if final_score < best_full_score:
        best_full_score = final_score

    return [int(x) for x in signs]
