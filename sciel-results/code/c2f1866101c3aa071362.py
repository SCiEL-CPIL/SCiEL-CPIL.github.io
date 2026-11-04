"""Second autocorrelation inequality: maximize (||h||_2^2)/(||h||_1 * ||autoconv(h)||_inf).

Strategy:
  - Represent heights h (length 50, nonnegative) via the squared modulus of the
    inverse DFT of a spectral vector z:  h = |IDFT(z)|^2.  This enforces
    nonnegativity automatically and exposes a smooth manifold for L-BFGS-B.
  - Objective: maximize  f = sum(h^2) / (sum(h) * max(autoconv(h)))
    where autoconv is the circular autocorrelation of h (the natural discrete
    analogue on equal intervals).  Equivalently minimize g = log(denom) - log(num).
  - Combine a direct h-space local polish with the spectral manifold search.
  - Explicit wall-clock deadline guard: track elapsed time and return the best
    result found with a safety margin before the 150 s limit.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50

# ---------------------------------------------------------------------------
# Time budget guard
# ---------------------------------------------------------------------------
_DEADLINE = 150.0
_START = time.time()
_MARGIN = 3.0  # safety margin (seconds) before the hard kill

def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def _autoconv(h):
    """Circular autocorrelation of h (period N)."""
    H = np.fft.fft(h)
    return np.fft.ifft(H * np.conj(H)).real


def _score(h):
    """Value of the objective: ||h||_2^2 / (||h||_1 * ||autoconv||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = float(np.sum(h))
    if l1 <= 0:
        return -1e-12
    l2sq = float(np.sum(h * h))
    ac = _autoconv(h)
    peak = float(np.max(ac))
    if peak <= 0:
        return 1e12
    return l2sq / (l1 * peak)


def _neg_log_score(h):
    """Smooth surrogate to minimize: -log(score) = log(denom) - log(num)."""
    h = np.asarray(h, dtype=float)
    l1 = float(np.sum(h))
    l2sq = float(np.sum(h * h))
    ac = _autoconv(h)
    peak = float(np.max(ac))
    if l1 <= 1e-12 or peak <= 1e-12 or l2sq <= 1e-12:
        return 1e12
    return math.log(l1 * peak) - math.log(l2sq)


def _neg_log_score_grad(h):
    """Analytic gradient of -log(score)."""
    h = np.asarray(h, dtype=float)
    n = h.size
    l1 = float(np.sum(h))
    l2sq = float(np.sum(h * h))
    ac = _autoconv(h)
    peak = float(np.max(ac))
    # index of the peak (for subgradient of max)
    kmax = int(np.argmax(ac))
    # d ac_k / d h_j  (circular autocorrelation)
    # ac_k = sum_j h_j h_{j-k}  (circular).  d ac_k/d h_j = h_{j-k} + h_{j+k} (circular)
    # Use FFT derivative for the whole vector; but we only need the max component.
    # d ac_k / d h = circ(h shifted) + circ(h shifted other way)
    idx = np.arange(n)
    dmax_dh = np.zeros(n)
    # ac_k = sum_j h_j * h_{(j-k) mod n}
    # d ac_k/d h_j = h_{(j-k) mod n} + h_{(j+k) mod n}
    for j in range(n):
        dmax_dh[j] = h[(j - kmax) % n] + h[(j + kmax) % n]
    # gradient of log(l1) - log(l2sq) + log(peak)
    g = np.zeros(n)
    g += (1.0 / l1) * np.ones(n)          # d log l1 / dh
    g += (2.0 * h / l2sq) * (-1.0)        # d(-log l2sq)/dh
    g += (dmax_dh / peak)                 # d log peak / dh
    return g


# ---------------------------------------------------------------------------
# Spectral parameterization: h = |IDFT(z)|^2, z in C^N
# ---------------------------------------------------------------------------
def _h_from_z(z):
    """z: complex vector length N.  h = |IDFT(z)|^2 (real, nonnegative)."""
    x = np.fft.ifft(z)
    h = np.abs(x) ** 2
    return h


def _spectral_neg_log_score(z):
    h = _h_from_z(z)
    return _neg_log_score(h)


def _spectral_neg_log_score_grad(z):
    """Gradient of -log(score(h(z))) w.r.t. z (real/imag packed)."""
    # Use finite differences in the spectral domain for robustness, then it's
    # smooth enough for L-BFGS-B.  Cost is acceptable given the small N.
    z0 = z
    f0 = _spectral_neg_log_score(z0)
    g = np.zeros_like(z0)
    step = 1e-6
    for i in range(z0.size):
        zp = z0.copy(); zp[i] += step
        zm = z0.copy(); zm[i] -= step
        g[i] = (_spectral_neg_log_score(zp) - _spectral_neg_log_score(zm)) / (2 * step)
    return g


# ---------------------------------------------------------------------------
# Initializations (diverse profiles)
# ---------------------------------------------------------------------------
def _initializations():
    inits = []
    # 1. uniform
    inits.append(np.ones(N))
    # 2. triangular / tent
    t = np.linspace(-1, 1, N)
    inits.append(np.maximum(0.0, 1.0 - np.abs(t)))
    # 3. Gaussian bump
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-6.0 * x * x))
    # 4. cosine bell
    inits.append(0.5 * (1.0 + np.cos(np.pi * x)))
    # 5. two-lobe
    inits.append(np.exp(-30.0 * (x - 0.4) ** 2) + np.exp(-30.0 * (x + 0.4) ** 2))
    # 6. low-frequency comb (sparse spectral)
    z = np.zeros(N, dtype=complex)
    z[0] = 1.0
    z[N // 2] = 0.5
    inits.append(_h_from_z(z))
    # 7. random (deterministic seed)
    rng = np.random.default_rng(0)
    inits.append(np.abs(rng.normal(size=N)) + 0.1)
    # 8. sparse spike
    sp = np.zeros(N); sp[N // 2] = 1.0
    inits.append(sp)
    # normalize each to unit L1 to remove scale degeneracy
    out = []
    for h in inits:
        s = float(np.sum(h))
        if s > 0:
            out.append(h / s)
        else:
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Direct h-space local polish (projection to nonnegativity via soft clip)
# ---------------------------------------------------------------------------
def _polish_direct(h0, max_iter=400, timeout=20.0):
    h0 = np.asarray(h0, dtype=float).copy()
    s = float(np.sum(h0))
    if s > 0:
        h0 = h0 / s

    def obj(x):
        # penalize negatives softly
        h = np.maximum(x, 0.0)
        return _neg_log_score(h)

    def jac(x):
        h = np.maximum(x, 0.0)
        g = _neg_log_score_grad(h)
        # zero gradient where clipped
        g[x < 0] = 0.0
        return g

    best_h = h0.copy()
    best_val = _score(h0)
    t0 = time.time()
    try:
        res = minimize(obj, h0, jac=jac, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10})
        h1 = np.maximum(res.x, 0.0)
        if _score(h1) > best_val:
            best_h, best_val = h1, _score(h1)
    except Exception:
        pass
    # a few more local restarts if time permits
    while _time_left() > max(timeout, 10.0):
        if time.time() - t0 > timeout:
            break
        rng = np.random.default_rng(abs(hash(tuple(np.round(best_h, 6)) % (10**9))))
        pert = best_h * (1.0 + 0.05 * rng.normal(size=N))
        pert = np.maximum(pert, 0.0)
        sp = float(np.sum(pert))
        if sp > 0:
            pert = pert / sp
        try:
            res = minimize(obj, pert, jac=jac, method="L-BFGS-B",
                           options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-10})
            h1 = np.maximum(res.x, 0.0)
            v = _score(h1)
            if v > best_val:
                best_h, best_val = h1, v
        except Exception:
            pass
        break  # keep it simple and fast
    return best_h, best_val


# ---------------------------------------------------------------------------
# Spectral manifold optimization
# ---------------------------------------------------------------------------
def _polish_spectral(h0, max_iter=300, timeout=25.0):
    # Initialize z from h0 via FFT
    z0 = np.fft.fft(h0)
    best_h = h0.copy()
    best_val = _score(h0)
    t0 = time.time()
    try:
        res = minimize(_spectral_neg_log_score, z0,
                       jac=_spectral_neg_log_score_grad,
                       method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10})
        h1 = _h_from_z(res.x)
        v = _score(h1)
        if v > best_val:
            best_h, best_val = h1, v
    except Exception:
        pass
    return best_h, best_val


# ---------------------------------------------------------------------------
# Sparse-spectral mask: restrict to K active bins, optimize magnitudes+phases
# ---------------------------------------------------------------------------
def _sparse_spectral_opt(K, h0, max_iter=200, timeout=20.0):
    z0 = np.fft.fft(h0)
    # pick the K bins with largest magnitude
    mags = np.abs(z0)
    active = np.argsort(mags)[::-1][:K]
    best_h = h0.copy()
    best_val = _score(h0)

    def pack(params):
        # params: 2K real -> magnitudes (log) + phases
        m = np.exp(params[:K])
        ph = params[K:]
        z = np.zeros(N, dtype=complex)
        for i, b in enumerate(active):
            z[b] = m[i] * np.exp(1j * ph[i])
        # enforce conjugate symmetry for real h? Not required; h=|IDFT|^2 real anyway.
        return z

    def obj(params):
        z = pack(params)
        return _spectral_neg_log_score(z)

    def jac(params):
        z = pack(params)
        gz = _spectral_neg_log_score_grad(z)
        g = np.zeros(2 * K)
        for i, b in enumerate(active):
            m = np.exp(params[:K][i])
            ph = params[K:][i]
            # d z_b / d m_i = exp(1j ph)
            # d z_b / d ph_i = z_b * 1j
            dz_dm = np.exp(1j * ph)
            dz_dph = z[b] * 1j
            g[i] = 2.0 * np.real(np.conj(dz_dm) * gz[b])
            g[K + i] = 2.0 * np.real(np.conj(dz_dph) * gz[b])
        return g

    # initial params
    p0 = np.zeros(2 * K)
    for i, b in enumerate(active):
        p0[i] = math.log(max(np.abs(z0[b]), 1e-8))
        p0[K + i] = np.angle(z0[b])

    t0 = time.time()
    try:
        res = minimize(obj, p0, jac=jac, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-10})
        h1 = _h_from_z(pack(res.x))
        v = _score(h1)
        if v > best_val:
            best_h, best_val = h1, v
    except Exception:
        pass
    return best_h, best_val


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    best_h = np.ones(N)
    best_val = _score(best_h)

    inits = _initializations()
    for h0 in inits:
        if _time_left() < 8.0:
            break
        # Direct h-space polish
        h1, v1 = _polish_direct(h0, max_iter=300, timeout=15.0)
        if v1 > best_val:
            best_h, best_val = h1, v1
        # Spectral manifold polish from the improved point
        if _time_left() < 8.0:
            break
        h2, v2 = _polish_spectral(best_h, max_iter=250, timeout=20.0)
        if v2 > best_val:
            best_h, best_val = h2, v2
        # Sparse spectral mask (a few K values)
        for K in (2, 3, 4, 5):
            if _time_left() < 8.0:
                break
            h3, v3 = _sparse_spectral_opt(K, best_h, max_iter=150, timeout=15.0)
            if v3 > best_val:
                best_h, best_val = h3, v3

    # Final high-precision polish from the best found
    if _time_left() > 10.0:
        h4, v4 = _polish_direct(best_h, max_iter=600, timeout=25.0)
        if v4 > best_val:
            best_h, best_val = h4, v4
    if _time_left() > 10.0:
        h5, v5 = _polish_spectral(best_h, max_iter=500, timeout=25.0)
        if v5 > best_val:
            best_h, best_val = h5, v5

    # Return a plain list of floats (nonnegative)
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]
