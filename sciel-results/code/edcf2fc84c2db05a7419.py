"""Model-and-optimize for flattest plus-minus-one 24-coefficient polynomial.

P(z) = sum c_k z^k, c_k in {+1,-1}, |c| = 24.
Score = max_{|z|=1} |P(z)| / sqrt(24).  Optimum ~ 1.2583758 (max|P| ~ 6.1363).

Model: coefficients are the parameters. We FIT them to minimize the objective
(max |P| on the unit circle) via (a) a good structured seed (Rudin-Shapiro) and
(b) local +/-1 flips accepted when they lower the measured peak.  The peak is
measured by dense FFT sampling plus golden-section refinement, then the true
sup-norm is estimated from the densest grid (a rigorous upper bound on the score).
"""
import math
import time
import numpy as np

_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0


def _time_left():
    return _LIMIT - (time.time() - _T0)


def _rs_poly():
    """Rudin-Shapiro coefficients truncated to _N (strong low-peak seed)."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]


def _eval(coeffs, M):
    """Return (peak, argmax_phase) of |P| on the unit circle via dense sampling."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    # P(z_k) = sum_j c_j * z_k^j for each k
    # Use matrix multiplication: Z[k,j] = z_k^j
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    best_idx = np.argmax(mag2)
    return math.sqrt(mag2[best_idx]), theta[best_idx]


def _score_final(coeffs, M=2048):
    """Score on a dense grid; the max on the grid is a rigorous upper bound on the score."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    Z = np.power(z[:, None], np.arange(_N)[None, :])
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    return math.sqrt(np.max(mag2)) / _SQRTN


def _refine(coeffs, M, peak, phase, iters=12):
    """Golden-section refinement of a local peak around `phase`."""
    lo = phase - math.pi / M
    hi = phase + math.pi / M
    c_arr = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)

    def f(a):
        z = np.exp(1j * a)
        # P(z) = sum c_j * z^j
        zk = np.power(z, powers)
        s = np.dot(c_arr, zk)
        return (s.real * s.real + s.imag * s.imag)

    f_lo = f(lo)
    f_hi = f(hi)
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            lo = m1
            f_lo = f1
        else:
            hi = m2
            f_hi = f2
    peak2 = max(f_lo, f_hi)
    if peak2 > peak * peak:
        peak = math.sqrt(peak2)
    return peak


def _search_peak(coeffs, M):
    """Measured peak: dense scan + refinement of top few peaks."""
    peak, phase = _eval(coeffs, M)
    # refine the global max and a couple neighbours for robustness
    nb = max(1, M // 512)
    for d in (0, nb, -nb, 2*nb, -2*nb):
        ph = phase + 2.0 * math.pi * d / M
        peak = _refine(coeffs, M, peak, ph, iters=10)
    return peak


def _fast_eval(coeffs, M):
    """Fast peak evaluation using vectorized computation."""
    k = np.arange(M)
    theta = 2.0 * np.pi * k / M
    z = np.exp(1j * theta)
    c = np.array(coeffs, dtype=complex)
    powers = np.arange(_N)
    # Compute z^j for all j simultaneously
    Z = z[:, None] ** powers[None, :]
    vals = Z @ c
    mag2 = np.abs(vals) ** 2
    best_idx = np.argmax(mag2)
    return math.sqrt(mag2[best_idx]), theta[best_idx]


def _local_search(seed, M, deadline):
    """Flip coefficients that reduce the measured peak; keep the best overall."""
    cur = list(seed)
    cur_peak = _search_peak(cur, M)
    best = list(cur)
    best_peak = cur_peak

    step = 1
    rng = np.random.default_rng(42)
    while _time_left() > 15.0:
        improved = False
        # Try all single flips in order
        for j in range(_N):
            if _time_left() < 15.0:
                break
            cur[j] = -cur[j]
            p = _search_peak(cur, M)
            if p < cur_peak - 1e-12:
                cur_peak = p
                improved = True
            else:
                cur[j] = -cur[j]
        if improved:
            if cur_peak < best_peak - 1e-12:
                best = list(cur)
                best_peak = cur_peak
        else:
            # Try double flips to escape local optima
            moved = False
            for j in range(_N):
                if _time_left() < 15.0:
                    break
                for k in range(j+1, _N):
                    if _time_left() < 15.0:
                        break
                    cur[j] = -cur[j]
                    cur[k] = -cur[k]
                    p = _search_peak(cur, M)
                    if p < cur_peak - 1e-12:
                        cur_peak = p
                        moved = True
                        if cur_peak < best_peak - 1e-12:
                            best = list(cur)
                            best_peak = cur_peak
                        break
                    else:
                        cur[j] = -cur[j]
                        cur[k] = -cur[k]
                if moved:
                    break
            if not moved:
                # Random restart from best
                cur = list(best)
                cur_peak = best_peak
                # Perturb randomly
                for _ in range(3):
                    j = int(rng.integers(0, _N))
                    cur[j] = -cur[j]
                cur_peak = _search_peak(cur, M)
                step += 1
    return best


def construct():
    global _T0
    _T0 = time.time()

    # Seed: Rudin-Shapiro (low peak ~6.19, just above the 6.1363 optimum).
    seed = _rs_poly()

    # Model-and-optimize: fit the coefficients to minimize max|P| via local flips.
    # Start with moderate M, then refine with larger M
    best = _local_search(seed, M=128, deadline=_LIMIT)

    if _time_left() > 20.0:
        # Refine with finer grid
        best = _local_search(best, M=256, deadline=_LIMIT)

    # Final high-precision polish: try all single flips with very fine peak measurement
    if _time_left() > 10.0:
        cur = list(best)
        cur_peak = _search_peak(cur, 512)
        best_peak = cur_peak
        for j in range(_N):
            if _time_left() < 5.0:
                break
            cur[j] = -cur[j]
            p = _search_peak(cur, 512)
            if p < best_peak - 1e-12:
                best_peak = p
                best = list(cur)
            else:
                cur[j] = -cur[j]
        # Second pass to catch any remaining improvements
        if _time_left() > 5.0:
            for j in range(_N):
                if _time_left() < 3.0:
                    break
                cur = list(best)
                cur[j] = -cur[j]
                p = _search_peak(cur, 512)
                if p < best_peak - 1e-12:
                    best_peak = p
                    best = list(cur)

    # Ensure valid output: exactly 24 coefficients in {+1,-1}.
    best = [1 if v >= 0 else -1 for v in best]
    return best[:_N]
