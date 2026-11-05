# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(pt, perm):
    """Makespan of a job permutation on a flow shop. O(n*m)."""
    m = pt.shape[1]
    n = perm.shape[0]
    if n == 0:
        return 0.0
    # C[i, k] = completion time of job i on machine k
    C = np.empty((n, m), dtype=np.float64)
    C[0, 0] = pt[perm[0], 0]
    for k in range(1, m):
        C[0, k] = C[0, k - 1] + pt[perm[0], k]
    for i in range(1, n):
        C[i, 0] = C[i - 1, 0] + pt[perm[i], 0]
        C[i, 1] = max(C[i, 0], C[i - 1, 1]) + pt[perm[i], 1]
        for k in range(2, m):
            C[i, k] = max(C[i, k - 1], C[i - 1, k]) + pt[perm[i], k]
    return float(C[-1, -1])


def _makespan_vec(pt, perm):
    """Vectorized makespan (NumPy). O(n*m) but faster for moderate sizes."""
    n, m = pt.shape
    if n == 0:
        return 0.0
    P = pt[perm]  # (n, m)
    if m == 1:
        return float(P.sum())
    C = np.empty((n, m), dtype=np.float64)
    C[0, :] = np.cumsum(P[0, :])
    C[1:, 0] = C[:-1, 0] + P[1:, 0]
    for k in range(1, m):
        C[1:, k] = np.maximum(C[1:, k - 1], C[:-1, k]) + P[1:, k]
    return float(C[-1, -1])


def _neighborhood_moves(n):
    """Yield (i, j) swap positions for the neighborhood, i<j, i and j adjacent-ish + random."""
    return None  # placeholder; moves generated on the fly


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape
    if n <= 1:
        return list(range(n))
    if m == 1:
        # single machine: order doesn't matter for makespan; return identity
        return list(range(n))

    deadline = time.time() + 150.0
    safety = 146.0  # stop new iterations by this absolute time
    rng = np.random.default_rng(20240607)

    # Choose makespan evaluator based on size.
    use_vec = (n * m) <= 200000
    mk = _makespan_vec if use_vec else _makespan

    # ---- Initial solutions: NEH (LPT total) + a few random starts ----
    def neh_order():
        return np.argsort(-pt.sum(axis=1))

    best_perm = neh_order().copy()
    best_ms = mk(pt, best_perm)

    # Tabu list: set of (i,j) swaps recently performed (store as i*n+j for i<j)
    tabu_tenure = max(8, int(0.12 * n))
    tabu = set()
    tabu_history = []  # list of moves for cycling out

    # SA parameters
    T0 = max(1e-6, 0.10 * best_ms)
    T_end = max(1e-9, T0 * 1e-5)
    # We'll run until deadline; adapt cooling to elapsed.

    # Precompute for fast random move selection
    # We'll use adjacent swaps + occasional longer swaps.
    cur_perm = best_perm.copy()
    cur_ms = best_ms
    T = T0

    # Estimate iterations per second dynamically: run a small calibration.
    t_cal = time.time()
    cal_iters = 0
    while time.time() - t_cal < 0.05 and time.time() < deadline:
        i = int(rng.integers(0, n - 1))
        j = i + 1
        cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]
        ms = mk(pt, cur_perm)
        cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]
        cal_iters += 1
    if cal_iters == 0:
        cal_iters = 1
    # Rough iters/sec (adjacent swap is cheap); use a conservative factor.
    ips = cal_iters / 0.05

    # Total budget in iterations
    # Reserve ~2s for finalization.
    remaining = max(1.0, (deadline - 2.0) - time.time())
    max_iters = int(ips * remaining * 0.9)
    max_iters = max(max_iters, 100)

    # Cooling schedule over max_iters
    if max_iters > 1:
        cool = (T_end / T0) ** (1.0 / max_iters)
    else:
        cool = 1.0

    # Tabu move key helper
    def tabu_key(i, j):
        if i > j:
            i, j = j, i
        return i * n + j

    # Track best
    for it in range(max_iters):
        if time.time() > safety:
            break

        # Choose a move: mostly adjacent swap, sometimes a 2-opt reversal or longer swap
        r = rng.random()
        if r < 0.7:
            i = int(rng.integers(0, n - 1))
            j = i + 1
        elif r < 0.85:
            # 2-opt: reverse a segment
            a = int(rng.integers(0, n - 1))
            b = int(rng.integers(0, n - 1))
            if a > b:
                a, b = b, a
            if b - a < 1:
                a = int(rng.integers(0, n - 1))
                b = min(a + 1, n - 1)
            i, j = a, b  # treat as segment reversal
        else:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                j = (j + 1) % n

        # Apply move
        if r < 0.85 and (r >= 0.7):
            # 2-opt reversal
            a, b = i, j
            cur_perm[a:b + 1] = cur_perm[a:b + 1][::-1]
            new_ms = mk(pt, cur_perm)
            # tabu key for segment (use a,b)
            tk = tabu_key(a, b)
        else:
            # swap
            cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]
            new_ms = mk(pt, cur_perm)
            tk = tabu_key(i, j)

        # Tabu management
        improving = new_ms < cur_ms
        # If move is in tabu and not improving (no aspiration), revert.
        if tk in tabu and not (improving and new_ms < best_ms):
            # revert
            if r < 0.85 and (r >= 0.7):
                a, b = i, j
                cur_perm[a:b + 1] = cur_perm[a:b + 1][::-1]
            else:
                cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]
            # cool and continue
            T *= cool
            continue

        # Accept/reject
        if improving or (T > 0 and rng.random() < np.exp(-(new_ms - cur_ms) / max(T, 1e-12))):
            cur_ms = new_ms
            if new_ms < best_ms:
                best_ms = new_ms
                best_perm = cur_perm.copy()
            # add to tabu
            tabu.add(tk)
            tabu_history.append(tk)
            if len(tabu_history) > tabu_tenure:
                old = tabu_history.pop(0)
                tabu.discard(old)
        else:
            # revert
            if r < 0.85 and (r >= 0.7):
                a, b = i, j
                cur_perm[a:b + 1] = cur_perm[a:b + 1][::-1]
            else:
                cur_perm[i], cur_perm[j] = cur_perm[j], cur_perm[i]

        T *= cool

    # Final: a short deterministic improvement pass (first-improvement adjacent swaps)
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n - 1):
                if time.time() > deadline - 5.0:
                    break
                cur_perm[i], cur_perm[i + 1] = cur_perm[i + 1], cur_perm[i]
                ms = mk(pt, cur_perm)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_perm = cur_perm.copy()
                    improved = True
                else:
                    cur_perm[i], cur_perm[i + 1] = cur_perm[i + 1], cur_perm[i]

    return list(best_perm)
# EVOLVE-BLOCK-END
