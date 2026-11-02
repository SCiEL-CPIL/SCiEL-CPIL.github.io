"""Signed autocorrelation maximum absolute-value variant (C3-prime).

Minimize  max_k |A(k)| / (sum_i h_i)^2  over signed step heights h,
where A is the (unnormalized) autoconvolution.  The constant seed h=1
gives 1/N; signed functions can drive the ratio far below that because
the denominator (sum h_i)^2 can be shrunk while sum h_i^2 (>= A(0)) is fixed.

Strategy: build several signed families (random signs, m-sequences/LFSR,
random real heights), pick the best by exact FFT autoconv, then polish with
smooth gradient descent on a smooth surrogate of the max-|A|/S^2 objective.
A hard wall-clock deadline guard always returns a valid signed signal.
"""
import time

import numpy as np

N = 400

# ---- deadline guard --------------------------------------------------------
_T0 = time.monotonic()
_LIMIT = 110.0  # wall-clock safety margin (evaluator kills at 150 s)

def _deadline():
    return (time.monotonic() - _T0) > _LIMIT


def _autoconv(h):
    """Exact linear autoconvolution A(k)=sum_i h_i h_{i+k}, k=-(N-1)..(N-1)."""
    h = np.asarray(h, dtype=np.float64)
    n = h.shape[0]
    m = 1
    while m < 2 * n:
        m <<= 1
    H = np.fft.rfft(h, n=m)
    Ac = np.fft.irfft(H * np.conj(H), n=m)[:2 * n - 1]
    # Ac[j] = sum_i h_i h_{i-j}; reverse to get the positive-lag convention.
    return Ac[::-1]


def _score(h):
    """C3' = max_k |A(k)| / (sum h_i)^2.  Returns inf for degenerate (sum~0)."""
    h = np.asarray(h, dtype=np.float64)
    S = h.sum()
    if abs(S) < 1e-9:
        return np.inf
    A = _autoconv(h)
    return float(np.max(np.abs(A)) / (S * S))


# ---- candidate families ----------------------------------------------------
def _rand_signs(rng):
    return np.where(rng.random(N) < 0.5, -1.0, 1.0)


def _lfsr(rng, bits=8, taps=None):
    """m-sequence-like signed sequence from an LFSR (period 2^bits-1), tiled."""
    if taps is None:
        taps = int(rng.integers(1, (1 << bits) - 1))
    state = int(rng.integers(1, (1 << bits) - 1))
    out = []
    for _ in range(N):
        out.append(1.0 if (state & 1) else -1.0)
        fb = (state ^ ((state >> 1) ^ (state >> (bits - 1)))) & 1
        state = (state >> 1) | (fb << (bits - 1))
        state &= (1 << bits) - 1
    return np.array(out, dtype=np.float64)


def _rand_real(rng, scale=1.0):
    x = rng.standard_normal(N) * scale
    return x


def _rand_scaled(rng, scale=1.0):
    x = rng.standard_normal(N) * scale
    S = x.sum()
    if abs(S) < 1e-12:
        return x
    # keep it bounded while preserving the (signed) direction
    return x / np.linalg.norm(x) * np.linalg.norm(x)


# ---- smooth gradient polish ------------------------------------------------
def _grad_step(h, t=0.05, lr=1e-2, iters=30):
    """Smooth gradient descent on  f = max_t(soft_max|A|/S^2 - t)."""
    h = np.array(h, dtype=np.float64)
    n = h.shape[0]
    S = h.sum()
    if abs(S) < 1e-6:
        return h
    A = _autoconv(h)
    signA = np.sign(A)
    for _ in range(iters):
        if _deadline():
            break
        S = h.sum()
        if abs(S) < 1e-6:
            break
        S2 = S * S
        z = np.abs(A) / S2 - t
        w = np.exp(z / t)
        w = w / w.sum()
        # d/dh_j softmax(|A|/S^2) = (1/S2) sum_k w_k signA_k (dA_k/dh_j)
        # dA_k/dh_j = h_{j+k}+h_{j-k}  (over valid indices)
        dA = np.zeros_like(h)
        for k in range(n):
            if k == 0:
                dA += h
            else:
                dA[:-k] += h[k:]
                dA[k:] += h[:-k]
        g = signA @ dA / S2
        # nudge away from S -> 0 (reward larger |S|)
        g -= 0.02 * np.sign(S)
        h = h - lr * g
    return h


def _best_candidate():
    """Return the lowest-scoring signed candidate among a small family set."""
    rng = np.random.default_rng(20240517)
    cands = []
    for _ in range(40):
        cands.append(_rand_signs(rng))
    for bits in (7, 8, 9, 10, 11, 12):
        for _ in range(8):
            cands.append(_lfsr(rng, bits=bits))
    for _ in range(30):
        cands.append(_rand_real(rng, scale=1.0))
    for _ in range(30):
        cands.append(_rand_scaled(rng, scale=1.0))

    best_h = np.where(rng.random(N) < 0.5, -1.0, 1.0)  # safe fallback
    best_s = _score(best_h)
    for h in cands:
        if _deadline():
            break
        s = _score(h)
        if s < best_s:
            best_s = s
            best_h = h
    return best_h


def construct():
    best_h = _best_candidate()

    # polish the best candidate a few times (cheap, deadline-guarded)
    for t in (0.05, 0.02, 0.01):
        if _deadline():
            break
        cand = _grad_step(best_h, t=t, lr=1e-2, iters=25)
        s = _score(cand)
        if np.isfinite(s) and s < _score(best_h):
            best_h = cand

    # final safety: guarantee a finite, non-degenerate signed signal
    if not np.all(np.isfinite(best_h)) or abs(best_h.sum()) < 1e-9:
        rng = np.random.default_rng(7)
        best_h = _rand_signs(rng)

    return best_h
