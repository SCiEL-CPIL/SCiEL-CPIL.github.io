"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Discrete formulation on 50 equal intervals of [-1/4, 1/4], spacing dh = 1/20.
Uses DCT parameterization h_k = |idct(c)|^2 to enforce non-negativity and
leverage spectral structure of the autoconvolution.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _ratio_neg(h):
    return -_ratio(h)


def _h_to_dct(h):
    """Convert h to DCT coefficients such that h = |idct(c)|^2."""
    # We want h_k = (idct(c))_k^2, so (idct(c))_k = sqrt(h_k) (nonneg)
    # c = dct(sqrt(h))
    h = np.maximum(h, 0.0)
    sqrt_h = np.sqrt(h)
    c = dct(sqrt_h, type=2, norm='ortho')
    return c


def _dct_to_h(c):
    """Convert DCT coefficients c back to h = |idct(c)|^2."""
    x = idct(c, type=2, norm='ortho')
    return x ** 2


def _soft_logR_and_grad_dct(c, beta=50.0):
    """
    Smooth surrogate in DCT space: log R_smooth = log(l2sq) - log(l1) - logsumexp(ac, beta)
    where h = |idct(c)|^2.
    """
    h = _dct_to_h(c)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 1e12, np.zeros_like(c)

    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    ac_shifted = ac - ac_max
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return 1e12, np.zeros_like(c)
    logsumexp = ac_max + (1.0 / beta) * math.log(sum_exp)

    if logsumexp <= 0.0:
        return 1e12, np.zeros_like(c)

    # Softmax weights
    w = exp_ac / sum_exp

    # Gradient of logsumexp wrt h:
    # d(logsumexp)/d(h_j) = sum_i w_i * d(ac_i)/d(h_j)
    # d(ac_i)/d(h_j) = 2*DH*h[i + OFFSET - j] if 0 <= i + OFFSET - j < N else 0
    d_logsumexp_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(N):
            k = i + OFFSET - j
            if 0 <= k < N:
                s += w[i] * h[k]
        d_logsumexp_dh[j] = 2.0 * DH * s

    # Gradient wrt h of log R:
    # d(log R)/d(h_j) = 2*h_j/l2sq - 1/l1 - d_logsumexp_dh[j]/logsumexp
    g_h = np.zeros(N)
    g_h += 2.0 * h / l2sq
    g_h -= 1.0 / l1
    g_h -= d_logsumexp_dh / logsumexp

    # Chain rule: h = x^2 where x = idct(c)
    # d(h_j)/d(c_m) = 2*x_j * (idct basis)
    # Using the property that idct is orthogonal (with norm='ortho'):
    # x = idct(c), so dx/dc is the idct matrix
    # d(h_j)/d(c_m) = 2*x_j * (idct matrix)_{j,m}
    # We can compute this as: g_c = 2 * x * idct(g_h) ... wait, need to be careful
    # Actually: h = x^2, so dh/dc = 2*x * (dx/dc)
    # And dx/dc = idct_matrix (since x = idct_matrix @ c)
    # So g_c = idct_matrix^T @ (2*x * g_h) = dct(2*x * g_h)
    # Because dct is the inverse/transpose of idct (with ortho norm)
    x = idct(c, type=2, norm='ortho')
    g_c = dct(2.0 * x * g_h, type=2, norm='ortho')

    logR = math.log(l2sq) - math.log(l1) - logsumexp
    return logR, g_c


def _local_optimize_dct(c0, time_budget):
    """Run L-BFGS-B on the DCT parameterization; return best h found."""
    c0 = np.asarray(c0, dtype=float)
    best_h = _dct_to_h(c0)
    best_v = _ratio(best_h)
    t0 = time.time()

    def neg_logR(c):
        c = np.asarray(c, dtype=float)
        h = _dct_to_h(c)
        v = _ratio(h)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(c):
        c = np.asarray(c, dtype=float)
        logR, g = _soft_logR_and_grad_dct(c, beta=50.0)
        if logR > 1e11:
            return 1e12, np.zeros_like(c)
        return -logR, -g

    try:
        res = minimize(
            neg_logR, c0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            rx = _dct_to_h(res.x)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    # Nelder-Mead polish in h-space
    try:
        for _ in range(3):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def _local_optimize_direct(h0, time_budget):
    """Run L-BFGS-B on the direct h parameterization; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    s = float(np.linalg.norm(h0))
    if s > 0:
        h0 = h0 / s
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    def neg_logR(x):
        x = np.maximum(x, 0.0)
        v = _ratio(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        l2sq = float(np.dot(x, x))
        l1 = float(np.sum(x))
        if l1 <= 0.0 or l2sq <= 0.0:
            return 1e12, np.zeros_like(x)
        ac = _autoconv(x)
        ac_max = float(np.max(ac))
        ac_shifted = ac - ac_max
        exp_ac = np.exp(50.0 * ac_shifted)
        sum_exp = float(np.sum(exp_ac))
        if sum_exp <= 0.0:
            return 1e12, np.zeros_like(x)
        logsumexp = ac_max + (1.0 / 50.0) * math.log(sum_exp)
        if logsumexp <= 0.0:
            return 1e12, np.zeros_like(x)
        w = exp_ac / sum_exp
        d_logsumexp_dh = np.zeros(N)
        for j in range(N):
            ss = 0.0
            for i in range(N):
                k = i + OFFSET - j
                if 0 <= k < N:
                    ss += w[i] * x[k]
            d_logsumexp_dh[j] = 2.0 * DH * ss
        g = np.zeros(N)
        g += 2.0 * x / l2sq
        g -= 1.0 / l1
        g -= d_logsumexp_dh / logsumexp
        logR = math.log(l2sq) - math.log(l1) - logsumexp
        return -logR, -g

    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    try:
        for _ in range(3):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def _initializations():
    """Yield a set of promising starting points."""
    inits = []
    rng = np.random.default_rng(12345)
    # Single spike at center
    for pos in [0, N // 2, N - 1, N // 4, 3 * N // 4]:
        s = np.zeros(N); s[pos] = 1.0
        inits.append(s)
    # Two spikes
    for a in range(0, N, 10):
        for b in range(a + 10, N, 10):
            s = np.zeros(N); s[a] = 1.0; s[b] = 1.0
            inits.append(s)
    # Three spikes
    for a in range(0, N, 15):
        for b in range(a + 15, N, 15):
            for c in range(b + 15, N, 15):
                s = np.zeros(N); s[a] = 1.0; s[b] = 1.0; s[c] = 1.0
                inits.append(s)
    # Random sparse profiles
    for _ in range(40):
        k = rng.integers(1, 12)
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)
    # Random dense profiles
    for _ in range(40):
        s = rng.random(N) ** 3.0
        inits.append(s)
    # Gaussian-ish bumps
    for center in range(0, N, 7):
        x = np.arange(N)
        s = np.exp(-0.5 * ((x - center) / 3.0) ** 2)
        inits.append(s)
    # Lorentzian profiles
    for center in range(0, N, 7):
        for width in [1.0, 2.0, 3.0]:
            x = np.arange(N)
            s = 1.0 / (1.0 + ((x - center) / width) ** 2)
            inits.append(s)
    return inits


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    inits = _initializations()
    scored = []
    for s in inits:
        v = _ratio(s)
        scored.append((v, s))
        if v > best_v:
            best_v = v
            best_h = s.copy()
    scored.sort(key=lambda t: -t[0])

    budget_per = 3.0
    for v0, s in scored:
        if _time_left() <= budget_per + 2.0:
            break
        if v0 < best_v * 0.5 and best_v > 0 and best_v > 15.0:
            continue

        # Try DCT optimization
        c0 = _h_to_dct(s)
        h_dct, v_dct = _local_optimize_dct(c0, time_budget=budget_per * 0.6)
        if v_dct > best_v:
            best_v = v_dct
            best_h = h_dct.copy()

        if _time_left() <= budget_per + 2.0:
            break

        # Try direct optimization
        h_dir, v_dir = _local_optimize_direct(s, time_budget=budget_per * 0.4)
        if v_dir > best_v:
            best_v = v_dir
            best_h = h_dir.copy()

        if _time_left() <= 20.0:
            break

    # Final polish
    if best_h is not None and _time_left() > 15.0:
        # Polish in DCT space
        c_best = _h_to_dct(best_h)
        h_pol, v_pol = _local_optimize_dct(c_best, time_budget=_time_left() * 0.5)
        if v_pol > best_v:
            best_v = v_pol
            best_h = h_pol.copy()

        if _time_left() > 10.0:
            h_pol2, v_pol2 = _local_optimize_direct(best_h, time_budget=_time_left() * 0.4)
            if v_pol2 > best_v:
                best_v = v_pol2
                best_h = h_pol2.copy()

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
