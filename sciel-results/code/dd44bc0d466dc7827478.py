"""Rewritten program: ADMM spectral-peak minimization for signed autocorrelation C3'.

Objective: minimize  max_k | (x * x)_k | / (sum x)^2  over signed step heights x in R^N.
We minimize the peak autoconvolution under a unit-energy constraint. The seed ones(N)
yields peak = N, energy = N, ratio = 1. We seek a flatter autocorrelation profile
(Barker / CAZAC-like) via ADMM with a soft-thresholding prox for sparsity.
"""
import time
import numpy as np

N = 400
DEADLINE_SECONDS = 145.0   # safety margin before the 150 s wall-clock kill

_START = time.time()


def _remaining():
    return DEADLINE_SECONDS - (time.time() - _START)


def _fft_len(n):
    # next power of two >= 2n-1 for linear autoconvolution via FFT
    m = 1
    while m < (2 * n - 1):
        m <<= 1
    return m


def _autoconv_peak(x):
    """Return max |linear autoconvolution| of x (real, symmetric)."""
    n = x.size
    m = _fft_len(n)
    X = np.fft.rfft(x, m)
    ac = np.fft.irfft(X * X, m)
    ac = ac[:2 * n - 1]
    return float(np.max(np.abs(ac)))


def _score_of(x):
    """Raw evaluator value: max|autoconv| / (sum x)^2. Lower is better."""
    s = float(np.sum(x))
    if s == 0.0:
        return float('inf')
    return _autoconv_peak(x) / (s * s)


def _soft_threshold(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


def construct():
    """Return a sequence of signed step heights minimizing the C3' peak ratio."""
    rng = np.random.default_rng(12345)

    # Initialize with a Barker-like / random-sign candidate, then refine.
    best_x = None
    best_score = float('inf')

    def consider(x):
        nonlocal best_x, best_score
        sc = _score_of(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Candidate 0: Barker code (length 13) tiled, plus random signs, plus ones.
    barker = np.array([1, 1, 1, 1, 1, -1, -1])
    base = np.zeros(N)
    base[:N % len(barker)] = barker[:N % len(barker)]
    for i in range(0, N, len(barker)):
        base[i:i + len(barker)] = barker
    consider(base)
    consider(np.ones(N))
    consider(rng.choice([-1.0, 1.0], size=N))

    # ADMM: minimize  (1/2)||x - z||^2 + peak_penalty  s.t. z = x,  ||x||_2^2 = 1
    # We minimize the spectral peak via a dual mask on |X|^2 and a prox for sparsity.
    x = best_x.copy()
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm

    rho = 1.0
    lam = 0.05   # sparsity threshold (soft-threshold magnitude)
    mu = 1.0     # dual step for the spectral mask

    # Dual spectral mask: penalize large |X|^2 (which corresponds to autoconv peak)
    m = _fft_len(N)
    mask = np.zeros(m // 2 + 1)

    it = 0
    while _remaining() > 2.0:
        it += 1
        # --- x-update: minimize  (1/2)||x - z||^2 + (rho/2)||x||^2  s.t. prox sparsity
        #     combined with spectral penalty via mask on |X|^2
        X = np.fft.rfft(x, m)
        # gradient of spectral peak penalty: mu * mask * conj(X)
        g = mu * mask * np.conj(X)
        x_new = np.fft.irfft((X - g) / (1.0 + rho), m)[:N]
        # proximal soft-threshold for sparsity / large dynamic range
        x_new = _soft_threshold(x_new, lam / (1.0 + rho))
        # re-normalize to unit energy
        nx = np.linalg.norm(x_new)
        if nx > 1e-12:
            x_new = x_new / nx
        x = x_new

        # --- spectral-mask (dual) update: push mask toward |X|^2 to suppress peak
        X = np.fft.rfft(x, m)
        power = np.abs(X) ** 2
        mask = np.maximum(mask + mu * (power - np.median(power)), 0.0)
        # decay mask to avoid over-suppression
        mask *= 0.99

        consider(x)

        # adaptive rho
        if it % 25 == 0:
            rho *= 1.1

    # Final polish: a few local sign-flip hill-climb steps to reduce peak
    if _remaining() > 3.0:
        x = best_x.copy()
        improved = True
        while improved and _remaining() > 3.0:
            improved = False
            for _ in range(40):
                i = int(rng.integers(0, N))
                trial = x.copy()
                trial[i] = -trial[i]
                if _score_of(trial) < _score_of(x):
                    x = trial
                    improved = True
        consider(x)

    if best_x is None:
        best_x = np.ones(N)
    return best_x


if __name__ == "__main__":
    res = construct()
    print("N =", len(res))
    print("score =", _score_of(res))
