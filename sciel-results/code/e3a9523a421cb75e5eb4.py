"""Optimize step heights to minimize max overlap with shifted 1-h.

Strategy: parameterize h by its DFT coefficients. The overlap at shift s is
N/2 - C(s) where C(s) is the cyclic autocorrelation. Minimizing the maximum
overlap equals maximizing the minimum autocorrelation C_min. Using Parseval,
C(s) = (1/N) sum_k |a_k|^2 e^{2 pi i k s / N}, which is smooth in the real/imag
parts of a_k with simple bound constraints (a_0 = N/2 fixed, a_{N/2} in [0,1],
others free). This enables fast multi-start L-BFGS-B optimization over a smooth
soft-min surrogate (log-sum-exp based) to avoid nonsmoothness of min(C(s)).
"""
import time
import numpy as np

N = 100
DEADLINE = 150.0
START = time.time()


def _time_left():
    return DEADLINE - (time.time() - START)


def _autocorr_from_a(a):
    # a is the full DFT (length N). C(s) = (1/N) sum_k |a_k|^2 e^{2 pi i k s / N}
    power = np.abs(a) ** 2
    # C(s) = ifft(power) (numpy ifft includes 1/N)
    return np.real(np.fft.ifft(power))


def _soft_min(C, beta=10.0):
    """Smooth approximation to min(C) using log-sum-exp."""
    m = np.min(C)
    return m + np.log(np.exp(beta * (C - m)).sum()) / beta


def _objective_flat(flat, beta=10.0):
    # flat = [a1_r, a1_i, ..., a_{N/2-1}_r, a_{N/2-1}_i, a_{N/2}]
    a = np.empty(N, dtype=complex)
    a[0] = N / 2.0
    a[N // 2] = flat[-1]
    if N > 2:
        a[1:N // 2] = flat[0:-1].reshape(-1, 2)
        a[N // 2 + 1:] = np.conj(a[1:N // 2][::-1])
    C = _autocorr_from_a(a)
    # maximize a smooth surrogate of the minimum C(s)
    return _soft_min(C, beta)


def _flat_to_h(flat):
    a = np.empty(N, dtype=complex)
    a[0] = N / 2.0
    a[N // 2] = flat[-1]
    if N > 2:
        a[1:N // 2] = flat[0:-1].reshape(-1, 2)
        a[N // 2 + 1:] = np.conj(a[1:N // 2][::-1])
    h = np.real(np.fft.ifft(a))
    return h


def _h_to_flat(h):
    a = np.fft.fft(h)
    flat = np.empty(N // 2 - 1 + 1)
    if N > 2:
        flat[0:-1] = a[1:N // 2].real, a[1:N // 2].imag
        flat[0:-1] = np.column_stack([a[1:N // 2].real, a[1:N // 2].imag]).ravel()
    flat[-1] = a[N // 2].real
    return flat


def _score_h(h):
    # max overlap = N/2 - min_s C(s)
    C = np.real(np.fft.ifft(np.abs(np.fft.fft(h)) ** 2))
    return N / 2.0 - np.min(C)


def _optimize_from_flat(flat0, maxiter=400, beta=10.0):
    from scipy.optimize import minimize
    nfree = N // 2 - 1
    bounds = [(None, None)] * (2 * nfree) + [(0.0, 1.0)]
    res = minimize(
        lambda f: -_objective_flat(f, beta),
        flat0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8},
    )
    return res


def construct():
    from scipy.optimize import minimize

    nfree = N // 2 - 1  # number of free complex coeffs (1..N/2-1)
    dim = 2 * nfree + 1

    # Uniform 0.5 baseline
    best_h = np.full(N, 0.5)
    best_score = _score_h(best_h)  # should be 0.5

    # A good deterministic start: spread energy across many Fourier modes
    # to flatten the autocorrelation (all C(s) equal -> min is high).
    rng = np.random.default_rng(12345)

    starts = []
    # Start 1: energy concentrated in first few modes (low frequency)
    for nmode in [1, 2, 3, 5, 10, 20, 40]:
        f = np.zeros(dim)
        # put equal magnitude on modes 1..min(nmode, N/2-1)
        nm = min(nmode, N // 2 - 1)
        if nm >= 1:
            mag = 0.3
            for k in range(1, nm + 1):
                f[2 * (k - 1)] = mag * np.cos(2 * np.pi * k / N)
                f[2 * (k - 1) + 1] = mag * np.sin(2 * np.pi * k / N)
        f[-1] = 0.5
        starts.append(f)

    # Start 2: random phases with fixed magnitudes (better exploration)
    for trial in range(5):
        f = np.zeros(dim)
        nm = N // 2 - 1
        mags = rng.uniform(0.1, 0.5, size=nm)
        phases = rng.uniform(0, 2 * np.pi, size=nm)
        for k in range(nm):
            f[2 * k] = mags[k] * np.cos(phases[k])
            f[2 * k + 1] = mags[k] * np.sin(phases[k])
        f[-1] = rng.uniform(0.0, 1.0)
        starts.append(f)

    # Random starts
    for _ in range(30):
        f = np.zeros(dim)
        f[0:-1] = rng.normal(0.0, 0.2, size=2 * nfree)
        f[-1] = rng.uniform(0.0, 1.0)
        starts.append(f)

    # Shuffle a bit for variety
    order = rng.permutation(len(starts))

    for idx in order:
        if _time_left() < 5.0:
            break
        f0 = starts[idx].copy()
        try:
            # Use a moderate beta for the main optimization
            res = _optimize_from_flat(f0, maxiter=300, beta=10.0)
            if res.success or res.fun is not None:
                h = _flat_to_h(res.x)
                # clip tiny numerical violations
                h = np.clip(h, 0.0, 1.0)
                # renormalize sum to N/2
                s = h.sum()
                if s > 0:
                    h = h * (N / 2.0) / s
                    h = np.clip(h, 0.0, 1.0)
                sc = _score_h(h)
                if sc < best_score:
                    best_score = sc
                    best_h = h.copy()
        except Exception:
            continue

    # Final polish on the best with tighter tolerance if time remains
    # Use a larger beta for the polish to better approximate the true min
    if _time_left() > 15.0:
        try:
            f0 = _h_to_flat(best_h)
            # First pass with high beta for precision
            res = _optimize_from_flat(f0, maxiter=1000, beta=50.0)
            h = _flat_to_h(res.x)
            h = np.clip(h, 0.0, 1.0)
            s = h.sum()
            if s > 0:
                h = h * (N / 2.0) / s
                h = np.clip(h, 0.0, 1.0)
            sc = _score_h(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
        except Exception:
            pass

    # Additional polish with even higher beta if time permits
    if _time_left() > 25.0:
        try:
            f0 = _h_to_flat(best_h)
            res = _optimize_from_flat(f0, maxiter=2000, beta=100.0)
            h = _flat_to_h(res.x)
            h = np.clip(h, 0.0, 1.0)
            s = h.sum()
            if s > 0:
                h = h * (N / 2.0) / s
                h = np.clip(h, 0.0, 1.0)
            sc = _score_h(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
        except Exception:
            pass

    # Safety: ensure valid output
    best_h = np.clip(best_h, 0.0, 1.0)
    s = best_h.sum()
    if s > 0:
        best_h = best_h * (N / 2.0) / s
        best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
