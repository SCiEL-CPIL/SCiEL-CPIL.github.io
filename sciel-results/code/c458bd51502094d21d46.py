"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Structural change: Parameterize h = |g|^2 where g is complex-valued.
This removes non-negativity constraints and leverages spectral structure.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Wall-clock deadline with safety margin (evaluator kills at 150 s).
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


def _soft_logR_and_grad(h, beta=50.0):
    """
    Smooth surrogate: log R_smooth = log(l2sq) - log(l1) - logsumexp(ac, beta)
    """
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 1e12, np.zeros_like(h)

    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    ac_shifted = ac - ac_max
    exp_ac = np.exp(beta * ac_shifted)
    sum_exp = float(np.sum(exp_ac))
    if sum_exp <= 0.0:
        return 1e12, np.zeros_like(h)
    logsumexp = ac_max + (1.0 / beta) * math.log(sum_exp)

    if logsumexp <= 0.0:
        return 1e12, np.zeros_like(h)

    w = exp_ac / sum_exp

    d_logsumexp_dh = np.zeros(N)
    for j in range(N):
        s = 0.0
        for i in range(N):
            k = i + OFFSET - j
            if 0 <= k < N:
                s += w[i] * h[k]
        d_logsumexp_dh[j] = 2.0 * DH * s

    g = np.zeros(N)
    g += 2.0 * h / l2sq
    g -= 1.0 / l1
    g -= d_logsumexp_dh / logsumexp

    logR = math.log(l2sq) - math.log(l1) - logsumexp
    return logR, g


def _normalize(h):
    s = float(np.linalg.norm(h))
    if s <= 0:
        return h
    return h / s


def _local_optimize_trust(h0, time_budget):
    """Run trust-region optimization on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
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
        logR, g = _soft_logR_and_grad(x, beta=50.0)
        if logR > 1e11:
            return 1e12, np.zeros_like(x)
        return -logR, -g

    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='trust-constr',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 200, 'gtol': 1e-10, 'disp': False,
                     'verbose': 0, 'barrier_tol': 1e-10},
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


def _local_optimize_lbfgsb(h0, time_budget):
    """Run L-BFGS-B on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
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
        logR, g = _soft_logR_and_grad(x, beta=50.0)
        if logR > 1e11:
            return 1e12, np.zeros_like(x)
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
    for pos in [0, N // 2, N - 1, N // 4, 3 * N // 4]:
        s = np.zeros(N); s[pos] = 1.0
        inits.append(s)
    for a in range(0, N, 10):
        for b in range(a + 10, N, 10):
            s = np.zeros(N); s[a] = 1.0; s[b] = 1.0
            inits.append(s)
    for a in range(0, N, 15):
        for b in range(a + 15, N, 15):
            for c in range(b + 15, N, 15):
                s = np.zeros(N); s[a] = 1.0; s[b] = 1.0; s[c] = 1.0
                inits.append(s)
    rng = np.random.default_rng(12345)
    for _ in range(40):
        k = rng.integers(1, 12)
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)
    for _ in range(40):
        s = rng.random(N) ** 3.0
        inits.append(s)
    for center in range(0, N, 7):
        x = np.arange(N)
        s = np.exp(-0.5 * ((x - center) / 3.0) ** 2)
        inits.append(s)
    return inits


def _complex_objective(z_flat):
    """
    Objective for complex parameterization z.
    h = |z|^2.
    We minimize -log(R) to maximize R.
    R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)
    """
    z = z_flat.reshape(N, 2)
    z_complex = z[:, 0] + 1j * z[:, 1]
    h = np.abs(z_complex) ** 2
    
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    
    if l1 <= 1e-12 or l2sq <= 1e-12:
        return 1e12
    
    ac = _autoconv(h)
    ac_max = float(np.max(ac))
    
    if ac_max <= 1e-12:
        return 1e12
        
    R = l2sq / (l1 * ac_max)
    
    if R <= 0:
        return 1e12
        
    return -math.log(R)


def _complex_gradient(z_flat):
    """
    Gradient of the complex objective using finite differences or analytic?
    For simplicity and robustness in this structural change, we use finite differences
    or rely on L-BFGS-B's internal approximation if jac is not provided?
    No, we should provide a gradient for speed.
    
    Let h_j = z_j \bar{z}_j = x_j^2 + y_j^2.
    d h_j / d x_j = 2 x_j
    d h_j / d y_j = 2 y_j
    
    We can reuse the gradient logic from _soft_logR_and_grad by treating h as the variable,
    then chain rule.
    """
    z = z_flat.reshape(N, 2)
    z_complex = z[:, 0] + 1j * z[:, 1]
    h = np.abs(z_complex) ** 2
    
    # Get gradient of log R wrt h
    logR, grad_h = _soft_logR_and_grad(h, beta=50.0)
    
    if logR > 1e11:
        return np.zeros_like(z_flat)
        
    grad_z = np.zeros_like(z_flat)
    # d logR / d x_j = grad_h[j] * 2 x_j
    # d logR / d y_j = grad_h[j] * 2 y_j
    grad_z[:, 0] = 2.0 * z[:, 0] * grad_h
    grad_z[:, 1] = 2.0 * z[:, 1] * grad_h
    
    # We are minimizing -logR, so negate
    return -grad_z


def _local_optimize_complex(z0, time_budget):
    """
    Optimize the complex parameterization z.
    Returns best h and value.
    """
    z0 = np.asarray(z0, dtype=float).flatten()
    
    # Initial h and value
    z_complex = z0[:N] + 1j * z0[N:]
    h_init = np.abs(z_complex) ** 2
    best_h = h_init.copy()
    best_v = _ratio(best_h)
    
    t0 = time.time()
    
    try:
        res = minimize(
            _complex_objective,
            z0,
            jac=_complex_gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        
        if res.x is not None:
            z_opt = res.x
            z_opt_complex = z_opt[:N] + 1j * z_opt[N:]
            h_opt = np.abs(z_opt_complex) ** 2
            v_opt = _ratio(h_opt)
            
            if v_opt > best_v:
                best_v = v_opt
                best_h = h_opt.copy()
                
    except Exception:
        pass
    
    # Polish with direct h optimization
    if _time_left() > 5.0:
        try:
            h_pol, v_pol = _local_optimize_lbfgsb(best_h, time_budget=_time_left() - 6.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass
            
    return best_h, best_v


def _initializations_complex():
    """Yield complex initializations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Random complex phases with random magnitudes
    for _ in range(50):
        mags = rng.random(N) ** 0.5
        phases = rng.random(N) * 2 * math.pi
        z = np.zeros(2 * N)
        z[0::2] = mags * np.cos(phases)
        z[1::2] = mags * np.sin(phases)
        inits.append(z)
        
    # Structured: Constant magnitude, random phases
    for _ in range(20):
        phases = rng.random(N) * 2 * math.pi
        z = np.zeros(2 * N)
        z[0::2] = np.cos(phases)
        z[1::2] = np.sin(phases)
        inits.append(z)
        
    # Structured: Linear phase (chirp)
    for freq in np.linspace(0.1, 2.0, 10):
        phases = 2 * math.pi * freq * np.arange(N) / N
        z = np.zeros(2 * N)
        z[0::2] = np.cos(phases)
        z[1::2] = np.sin(phases)
        inits.append(z)
        
    return inits


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    # 1. Try complex parameterizations
    complex_inits = _initializations_complex()
    
    # Evaluate and sort
    scored_complex = []
    for z in complex_inits:
        z_complex = z[:N] + 1j * z[N:]
        h = np.abs(z_complex) ** 2
        v = _ratio(h)
        scored_complex.append((v, z))
        if v > best_v:
            best_v = v
            best_h = h.copy()
            
    scored_complex.sort(key=lambda t: -t[0])
    
    budget_per = 3.0
    for idx, (v0, z) in enumerate(scored_complex):
        if _time_left() <= budget_per + 2.0:
            break
            
        # Skip if not competitive
        if v0 < best_v * 0.5 and best_v > 0:
            if best_v > 10.0:
                continue
                
        h_opt, v_opt = _local_optimize_complex(z, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
            
        if _time_left() <= 20.0:
            break
            
    # 2. Also try direct h initializations if time permits
    if _time_left() > 30.0:
        direct_inits = _initializations()
        scored_direct = []
        for s in direct_inits:
            v = _ratio(s)
            scored_direct.append((v, s))
            if v > best_v:
                best_v = v
                best_h = s.copy()
                
        scored_direct.sort(key=lambda t: -t[0])
        
        for idx, (v0, s) in enumerate(scored_direct[:10]):
            if _time_left() <= 10.0:
                break
            if v0 < best_v * 0.5:
                continue
                
            h_opt, v_opt = _local_optimize_lbfgsb(s, time_budget=5.0)
            if v_opt > best_v:
                best_v = v_opt
                best_h = h_opt.copy()

    # Final polish
    if best_h is not None and _time_left() > 15.0:
        try:
            h_pol, v_pol = _local_optimize_trust(best_h, time_budget=_time_left() - 18.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

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
