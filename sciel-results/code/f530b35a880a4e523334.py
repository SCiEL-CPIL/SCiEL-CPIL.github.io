"""
Signed step heights minimizing max|autoconv| / (sum h)^2.

Strategy:
  1. Represent h as a superposition of orthogonal Walsh-Hadamard basis functions.
  2. Use a soft-max (log-sum-exp) surrogate for the max|ac| term, annealed with
     increasing beta for precision.
  3. Optimize continuous coefficients via L-BFGS, then quantize to +/-1 signs.
  4. A final single-flip local search with incremental autoconv update polishes.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# EVOLVE-BLOCK-START


def _autoconv(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2 * N - 1]


def _raw_value(h):
    """max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    return float(np.max(np.abs(ac))) / (s * s)


def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max|ac| via log-sum-exp."""
    a = np.abs(ac)
    m = np.max(a)
    if m < 1e-15:
        return 0.0, None
    scaled = beta * (a - m)
    scaled = np.clip(scaled, -60, 60)
    e = np.exp(scaled)
    log_sum = np.log(np.sum(e))
    weights = e / np.sum(e)
    return m + log_sum / beta, weights


def _smooth_objective(h, beta=50.0):
    """Smooth objective: soft_max_abs(ac) / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = _autoconv(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)


def _smooth_gradient(h, beta=50.0):
    """
    Gradient of soft_max_abs(ac) / (sum h)^2 w.r.t. h via FFT cross-correlation.
    """
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(h)

    ac = _autoconv(h)
    sm, weights = _soft_max_abs(ac, beta)
    if weights is None:
        return np.zeros_like(h)

    g_ac = weights * np.sign(ac)

    fsize_corr = 1
    while fsize_corr < 3 * N:
        fsize_corr *= 2

    V = np.fft.rfft(g_ac, fsize_corr)
    H = np.fft.rfft(h, fsize_corr)
    C = np.fft.irfft(V * np.conj(H), fsize_corr)

    grad_S = 2.0 * C[:N]
    df = (grad_S * s - 2.0 * sm) / (s ** 3)
    return df


def _lbfgs_smooth(h, beta=50.0, max_iter=300, deadline=None):
    """L-BFGS on the smooth soft-max surrogate."""
    h0 = h.copy()

    def fun(x):
        if deadline is not None and time.time() > deadline:
            return 1e10
        return _smooth_objective(x, beta)

    def jac(x):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x)
        return _smooth_gradient(x, beta)

    try:
        res = minimize(fun, h0, method='L-BFGS-B', jac=jac,
                       options={'maxiter': max_iter, 'gtol': 1e-7, 'maxfun': 1000})
        return res.x, res.fun
    except Exception:
        return h0, _smooth_objective(h0, beta)


def _build_walsh_hadamard_basis(n, num_funcs):
    """
    Build an orthonormal Walsh-Hadamard basis by:
      1. Generating a Hadamard matrix of size 2^k >= n.
      2. Taking the first num_funcs rows, truncated to n columns.
      3. Orthonormalizing via QR.
    """
    size = 1
    while size < n:
        size *= 2

    H = np.ones((1, 1))
    while H.shape[0] < size:
        H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])

    H_sub = H[:, :n]  # size x n
    if num_funcs > size:
        num_funcs = size
    rows = H_sub[:num_funcs, :]  # num_funcs x n
    Q, _ = np.linalg.qr(rows.T)  # n x num_funcs
    return Q.T  # num_funcs x n


def _quantize(h):
    """Convert to +/-1 signs, breaking zero-mass ties randomly."""
    s = np.sum(h)
    signs = np.where(h >= 0, 1.0, -1.0)
    if abs(s) < 1e-6:
        k = int(N // 2)
        idx = np.argsort(np.abs(h))
        idx_rev = idx[::-1]
        signs[idx_rev[:k]] = 1.0
        signs[idx_rev[k:]] = -1.0
    return signs


def _sign_flip_local_search(h, max_passes=15, deadline=None):
    """
    Local search over +/-1 signs using incremental autoconv update.
    O(N) per flip candidate (no FFT).
    """
    h = h.copy()
    s = np.sum(h)
    ac = _autoconv(h)
    best_val = float(np.max(np.abs(ac))) / (s * s)
    M = float(np.max(np.abs(ac)))

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            hi = h[i]
            s_new = s - 2.0 * hi
            if abs(s_new) < 1e-12:
                continue

            # ac_new = ac - 2*hi*(h shifted)
            ac_new = ac.copy()
            # term 1: ac_new[k] -= 2*hi * h[k-i] for k in [i, i+N-1]
            k1s, k1e = i, min(i + N, 2 * N - 1)
            if k1s < k1e:
                ac_new[k1s:k1e] -= 2.0 * hi * h[:k1e - k1s]
            # term 2: ac_new[k] -= 2*hi * h[i-k] for k in [max(0,i-N+1), i]
            k2s, k2e = max(0, i - N + 1), i + 1
            if k2s < k2e:
                idx = i - np.arange(k2s, k2e)
                ac_new[k2s:k2e] -= 2.0 * hi * h[idx]

            M_new = float(np.max(np.abs(ac_new)))
            new_val = M_new / (s_new * s_new)
            if new_val < best_val:
                h[i] = -hi
                s = s_new
                ac = ac_new
                M = M_new
                best_val = new_val
                improved = True
        if not improved:
            break

    return h, best_val


def _global_sign_search(basis, coeffs_init, deadline=None):
    """
    Global search over signs of a low-dimensional basis representation.
    h = sum_k c_k * b_k; c_k in {+1, -1}.
    Exhaustively searches 2^dim sign combinations, using FFT for each.
    """
    dim = basis.shape[0]
    if dim > 20:
        return coeffs_init.copy()

    best_c = coeffs_init.copy()
    best_val = 1e10
    n_total = 1 << dim

    for mask in range(n_total):
        if deadline is not None and time.time() > deadline:
            break
        c = np.ones(dim)
        for k in range(dim):
            if (mask >> k) & 1:
                c[k] = -1.0
        h = c @ basis
        val = _raw_value(h)
        if val < best_val:
            best_val = val
            best_c = c

    return best_c


def _refine_in_subspace(basis, h, dim, beta=50.0, max_iter=300, deadline=None):
    """
    Refine h within the span of 'basis' (dim orthonormal vectors).
    Projects h onto the subspace, optimizes continuous coefficients, then
    quantizes back to signs.
    """
    if deadline is not None and time.time() > deadline:
        return h, _raw_value(h)

    proj = basis @ h  # dim
    # Ensure nonzero coefficients for a valid starting point
    if np.all(np.abs(proj) < 1e-6):
        proj = np.ones(dim) / np.sqrt(dim)

    def obj(c):
        if deadline is not None and time.time() > deadline:
            return 1e10
        h_t = c @ basis
        s = np.sum(h_t)
        if abs(s) < 1e-12:
            return 1e10
        ac = _autoconv(h_t)
        return float(np.max(np.abs(ac))) / (s * s)

    def grad_c(c):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(c)
        h_t = c @ basis
        g = _smooth_gradient(h_t, beta)
        return basis @ g

    try:
        res = minimize(obj, proj, method='L-BFGS-B', jac=grad_c,
                       options={'maxiter': max_iter, 'gtol': 1e-6, 'maxfun': 800})
        best_c = res.x
    except Exception:
        best_c = proj

    # Quantize the optimized coefficients to signs
    c_q = np.where(best_c >= 0, 1.0, -1.0)
    h_q = c_q @ basis

    # Compare with the original
    val_q = _raw_value(h_q)
    val_orig = _raw_value(h)
    if val_q < val_orig:
        return h_q, val_q
    return h, val_orig


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin

    best_score = np.inf
    best_h = np.ones(N)

    def consider(h):
        nonlocal best_score, best_h
        v = _raw_value(h)
        if v < best_score:
            best_score = v
            best_h = h.copy()
        return v

    # Build Walsh-Hadamard basis (orthonormal, +/-1 entries, bounded autoconv peaks)
    wh_basis = _build_walsh_hadamard_basis(N, 16)

    # Initializations: constant, chirp, random, windowed, multi-frequency
    inits = [np.ones(N)]
    i = np.arange(N)
    inits.append(_quantize(np.cos(np.pi * i**2 / N)))
    for seed in [1, 7, 42]:
        rng = np.random.default_rng(seed)
        inits.append(_quantize(rng.standard_normal(N)))
    inits.append(_quantize(np.exp(-((i - N / 2)**2) / (2 * (N / 4)**2))))
    for f1, f2 in [(1, 2), (1, 3), (2, 5)]:
        inits.append(_quantize(np.cos(2 * np.pi * f1 * i / N) + 0.5 * np.cos(2 * np.pi * f2 * i / N)))
    inits.append(_quantize(np.sin(2 * np.pi * 3 * i / N) * np.cos(2 * np.pi * 1 * i / N)))

    # Optimization pipeline
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        h_work = h_init.copy()

        # 1. L-BFGS on soft-max surrogate (annealed beta)
        for beta in [10.0, 30.0, 80.0]:
            rem = deadline - time.time()
            if rem < 5:
                break
            h_opt, _ = _lbfgs_smooth(h_work, beta=beta, max_iter=150,
                                     deadline=deadline - 3)
            h_work = h_opt

        # 2. Quantize to signs
        h_work = _quantize(h_work)

        # 3. Refine in Walsh-Hadamard subspace (structural sign search)
        rem = deadline - time.time()
        if rem > 10:
            h_work, _ = _refine_in_subspace(wh_basis, h_work, 16,
                                            beta=50.0, max_iter=200,
                                            deadline=deadline - 5)

        # 4. Global sign search in low-dim subspace (exhaustive 2^dim)
        rem = deadline - time.time()
        if rem > 8:
            proj = wh_basis @ h_work
            c_best = _global_sign_search(wh_basis, proj, deadline=deadline - 3)
            h_g = c_best @ wh_basis
            if _raw_value(h_g) < _raw_value(h_work):
                h_work = h_g

        # 5. Single-flip local search (incremental, fast)
        rem = deadline - time.time()
        if rem > 8:
            h_work, _ = _sign_flip_local_search(h_work, max_passes=10,
                                                deadline=deadline - 3)

        consider(h_work)

    # Final refinement on the best
    rem = deadline - time.time()
    if rem > 15:
        best_h, _ = _lbfgs_smooth(best_h, beta=100.0, max_iter=300,
                                  deadline=deadline - 8)
        best_h = _quantize(best_h)

    rem = deadline - time.time()
    if rem > 10:
        best_h, _ = _refine_in_subspace(wh_basis, best_h, 16,
                                        beta=80.0, max_iter=300,
                                        deadline=deadline - 5)

    rem = deadline - time.time()
    if rem > 10:
        best_h, _ = _sign_flip_local_search(best_h, max_passes=20,
                                            deadline=deadline - 3)

    consider(best_h)

    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
