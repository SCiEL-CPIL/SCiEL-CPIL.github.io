"""Erdos minimum overlap solver.

Structural change: replace the trivial constant-0.5 baseline with a real
search.  The constant vector is provably optimal for the *average* overlap,
but the objective here is the *maximum* overlap over discrete shifts.
A non-uniform vector can push the peak below the constant value while
keeping some other shifts higher, lowering max-overlap.

We use a time-budgeted iterative projected-gradient search:
  1. Start from a random feasible vector (in [0,1], sum = N/2).
  2. At each iteration compute the overlap vector O(k) for k=0..N-1,
     find the shift k* that achieves the maximum, and take the negative
     subgradient with respect to h_i at that shift.
  3. Project back onto the box [0,1] and onto the simplex {sum = N/2}.
  4. Track the best (lowest max-overlap) vector seen.

An analytic lower bound from the autocorrelation energy is used to stop
early if we reach the bound.
"""

import numpy as np
import time

N = 100


# ---- helpers ------------------------------------------------------------

def _overlap_max(h):
    """Return (max_overlap, argmax_shift) for discrete circular shifts.

    O(k) = sum_i h_i * (1 - h_{(i+k) % N})
         = sum_i h_i  -  sum_i h_i * h_{(i+k) % N}
         = (N/2)       -  autocorr[k]

    where autocorr[k] = sum_i h_i * h_{(i+k) % N}.
    """
    # FFT-based autocorrelation (linear, then wrap)
    H = np.fft.fft(h)
    ac = np.real(np.fft.ifft(H * np.conj(H)))
    # ac[i] = sum_j h[j] * h[(j-i) mod N]  ->  circular autocorr
    # We need C[k] = sum_i h_i * h_{(i+k) % N}
    # ac[k] as defined above = sum_j h[j] * h[(j-k) mod N] = sum_i h[(i+k) mod N] * h[i]
    # which is the same sum, just index re-labeled.  So ac[k] IS C[k].
    total = np.sum(h)
    o = total - ac  # overlap at each shift
    k_max = int(np.argmax(o))
    return float(o[k_max]), k_max


def _project_box_simplex(h, target_sum):
    """Project h onto {0 <= x_i <= 1, sum(x) = target_sum}."""
    # First clamp
    h = np.clip(h, 0.0, 1.0)
    # Then adjust to meet the sum constraint via water-filling / bisection
    # We want the closest point in the hypercube with the right sum.
    # Standard: sort, use a lambda (Lagrange multiplier).
    s = np.sum(h)
    diff = target_sum - s
    if abs(diff) < 1e-14:
        return h
    # We need to add/subtract 'diff' across coordinates while staying in [0,1].
    # Use bisection on lambda: x_i = clip(h_i - lambda, 0, 1), choose lambda
    # so that sum = target_sum.
    def sum_at(lam):
        return np.sum(np.clip(h - lam, 0.0, 1.0))

    # Find range for lambda
    # If diff > 0 we need to increase sum -> lambda negative
    # If diff < 0 we need to decrease sum -> lambda positive
    lo, hi = -1.0, 1.0
    # Expand if needed
    for _ in range(60):
        if sum_at(lo) > target_sum and sum_at(hi) > target_sum:
            lo *= 2
        elif sum_at(lo) < target_sum and sum_at(hi) < target_sum:
            hi *= 2
        else:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        sm = sum_at(mid)
        if sm > target_sum:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    return np.clip(h - lam, 0.0, 1.0)


def _analytic_lb(N, total):
    """Lower bound on min overlap from autocorrelation energy.

    sum_k C(k)^2 / N >= (sum h)^2 / N  (Plancherel / Parseval for circular ac)
    Actually: sum_k C(k)^2 = sum_k |FFT(h)_k|^2 * ...
    Simpler: mean of C(k) over k=0..N-1 = (sum h)^2 / N.
    So min C(k) <= mean C(k) = total^2 / N.
    max O(k) = total - min C(k) >= total - total^2/N.
    """
    return total - total ** 2 / N


# ---- search -------------------------------------------------------------

def _search(deadline):
    best_h = np.full(N, 0.5)
    best_val, _ = _overlap_max(best_h)

    rng = np.random.default_rng(42)
    target = N / 2.0
    lb = _analytic_lb(N, target)

    # Try several random restarts, each a short projected-gradient run
    n_restarts = 200
    iters_per_restart = 3000
    lr = 0.5

    for rs in range(n_restarts):
        if time.time() > deadline:
            break
        # Random feasible start
        h = rng.random(N) * 0.5  # in [0, 0.5]
        h = _project_box_simplex(h, target)

        for it in range(iters_per_restart):
            if time.time() > deadline:
                return best_h, best_val
            val, k = _overlap_max(h)
            if val < best_val - 1e-12:
                best_val = val
                best_h = h.copy()
            if best_val <= lb + 1e-9:
                return best_h, best_val

            # Subgradient of O(k*) w.r.t. h_i:
            # O(k) = sum_i h_i * (1 - h_{(i+k) mod N})
            # dO/dh_i = (1 - h_{(i+k) mod N}) + sum_j delta_{i, j+k} * (1 - h_j)
            #         = (1 - h_{(i+k) mod N}) + (1 - h_{i-k mod N})
            # Wait, let's re-derive carefully.
            # O(k) = sum_i h_i (1 - h_{i+k})
            # = sum_i h_i - sum_i h_i h_{i+k}
            # dO/dh_m = 1 - d/dh_m [sum_i h_i h_{i+k}]
            # The sum term: term is nonzero when i=m (h_m h_{m+k}) or i+k=m i.e. i=m-k (h_{m-k} h_m)
            # d/dh_m sum_i h_i h_{i+k} = h_{m+k} + h_{m-k mod N}
            # So dO/dh_m = 1 - h_{m+k} - h_{m-k mod N}
            idx_p = (np.arange(N) + k) % N
            idx_m = (np.arange(N) - k) % N
            grad = 1.0 - h[idx_p] - h[idx_m]
            # Negative gradient step (we want to decrease O(k*))
            h_new = h - lr * grad
            # But this step might not reduce O(k*) if lr is too large;
            # projected gradient with a simple line search would be better,
            # but for speed we use a fixed small lr and projection.
            h = _project_box_simplex(h_new, target)

        if time.time() > deadline:
            break

    return best_h, best_val


def construct():
    """Return a sequence of N step heights in [0,1] with sum exactly N/2,
    minimizing the maximum overlap with its complement shifted."""
    start = time.time()
    deadline = start + 120.0 - 10.0  # 10s safety margin

    best_h, best_val = _search(deadline)

    # Final safety: ensure exact sum and bounds
    best_h = np.clip(best_h, 0.0, 1.0)
    s = np.sum(best_h)
    diff = N / 2.0 - s
    # Spread the residual evenly, then clamp and re-project
    if abs(diff) > 1e-12:
        best_h = best_h + diff / N
        best_h = _project_box_simplex(best_h, N / 2.0)

    # Round to avoid floating point noise that might make sum slightly off
    # but only if it doesn't push values out of [0,1]
    best_h = np.round(best_h, 10)
    s = np.sum(best_h)
    if abs(s - N / 2.0) > 1e-9:
        # Re-project without rounding
        best_h = _project_box_simplex(np.clip(best_h, 0.0, 1.0), N / 2.0)

    return best_h


# EVOLVE-BLOCK-END
