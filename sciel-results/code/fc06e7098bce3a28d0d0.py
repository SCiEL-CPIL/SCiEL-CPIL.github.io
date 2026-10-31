"""Uncertainty inequality optimization with analytic gradient and multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # 15s safety margin before 210s kill


def _time_left():
    return _DEADLINE - time.time()


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..30."""
    I = np.zeros(31)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 31):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()


def _coeffs_from_params(a0, a4, a8):
    """
    Compute polynomial coefficients p_k of P(x) = sum p_k x^(2k)
    where P = a0*H0 + a4*H4 + a8*H8 + a12*H12
    and a12 = -(a0 + 3*a4 + 105*a8)/945
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p = np.zeros(7)
    # p[k] = coefficient of x^(2k)
    # H0: [1, 0, 0, 0, 0, 0, 0]
    # H4: [3, -6, 1, 0, 0, 0, 0]
    # H8: [105, -420, 210, -28, 1, 0, 0]
    # H12: [945, -7560, 13860, -6930, 1155, -66, 1]
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p


def _compute_bound(a0, a4, a8):
    """Compute the uncertainty product sqrt(vf * vF)."""
    p = _coeffs_from_params(a0, a4, a8)
    # Q_k = (-1)^k * p_k for Fourier side
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Compute m_k = integral x^(2k) |P|^2 e^(-x^2) dx for k=0,1,2
    # m_k = sum_{i,j} p[i]*p[j] * I[k+i+j]
    m = np.zeros(3)
    n = np.zeros(3)
    
    for i in range(7):
        for j in range(7):
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 31:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18
    
    return bound


def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product and its analytic gradient
    with respect to (a0, a4, a8).
    
    vf = m2/m0 where m_k = integral x^(2k) |P|^2 e^(-x^2) dx
    vF = n2/n0 where n_k = 2*pi * integral x^(2k) |Q|^2 e^(-x^2) dx, Q_k = (-1)^k p_k
    
    d(bound)/da_i = (1/(2*bound)) * (d(vf)/da_i * vF + vf * d(vF)/da_i)
    d(vf)/da_i = (dm2/da_i * m0 - m2 * dm0/da_i) / m0^2
    d(vF)/da_i = (dn2/da_i * n0 - n2 * dn0/da_i) / n0^2
    """
    p = _coeffs_from_params(a0, a4, a8)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    # Compute m_k and n_k for k=0,1,2
    m = np.zeros(3)
    n = np.zeros(3)
    
    for i in range(7):
        for j in range(7):
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            if pij == 0.0 and qij == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 31:
                    m[k] += pij * _I[idx]
                    n[k] += qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    if m2 <= 0 or n2 <= 0:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Compute dp/da for each parameter
    # a12 = -(a0 + 3*a4 + 105*a8)/945
    # da12/da0 = -1/945, da12/da4 = -3/945 = -1/315, da12/da8 = -105/945 = -1/9
    
    da12_da = np.array([-1.0/945.0, -1.0/315.0, -1.0/9.0])
    
    # dp[k]/da_i = coeff_H4[k]*delta_{i,1} + coeff_H8[k]*delta_{i,2} + coeff_H12[k]*da12/da_i
    # where coeff_H4 = [3, -6, 1, 0, 0, 0, 0]
    # coeff_H8 = [105, -420, 210, -28, 1, 0, 0]
    # coeff_H12 = [945, -7560, 13860, -6930, 1155, -66, 1]
    
    coeff_H4 = np.array([3.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    coeff_H8 = np.array([105.0, -420.0, 210.0, -28.0, 1.0, 0.0, 0.0])
    coeff_H12 = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])
    
    # dp_da[i][k] = dp[k]/da_i
    dp_da = np.zeros((7, 3))
    for i in range(3):
        dp_da[:, i] = (coeff_H4 if i == 1 else (coeff_H8 if i == 2 else np.zeros(7))) + coeff_H12 * da12_da[i]
    
    # dm_k/da_i = 2 * sum_j p[j] * dp[k]/da_i * I[k+i+j] + 2 * sum_j p[j] * dp[j]/da_i * I[k+i+j]
    # More precisely: m_k = sum_{i,j} p[i]*p[j]*I[k+i+j]
    # dm_k/da = 2 * sum_{i,j} (dp[i]/da * p[j] + p[i] * dp[j]/da) * I[k+i+j] / 2
    # = 2 * sum_{i,j} dp[i]/da * p[j] * I[k+i+j]  (by symmetry)
    
    dm_da = np.zeros((3, 3))  # dm_da[k, i] = dm[k]/da[i]
    dn_da = np.zeros((3, 3))  # dn_da[k, i] = dn[k]/da[i]
    
    for i in range(3):
        for k in range(3):
            dm_val = 0.0
            dn_val = 0.0
            for ii in range(7):
                for jj in range(7):
                    idx = k + ii + jj
                    if idx < 31:
                        dm_val += 2.0 * (dp_da[ii, i] * p[jj] + p[ii] * dp_da[jj, i]) * 0.5 * _I[idx]
                        dn_val += 2.0 * (dp_da[ii, i] * q[jj] + q[ii] * dp_da[jj, i]) * 0.5 * _I[idx]
            dm_da[k, i] = dm_val
            dn_da[k, i] = dn_val * 2.0 * np.pi
    
    # d(vf)/da_i = (dm2/da_i * m0 - m2 * dm0/da_i) / m0^2
    # d(vF)/da_i = (dn2/da_i * n0 - n2 * dn0/da_i) / n0^2
    # d(bound)/da_i = (1/(2*bound)) * (d(vf)/da_i * vF + vf * d(vF)/da_i)
    
    grad = np.zeros(3)
    for i in range(3):
        dvf_da = (dm_da[1, i] * m0 - m2 * dm_da[0, i]) / (m0 * m0)
        dvF_da = (dn_da[1, i] * n0 - n2 * dn_da[0, i]) / (n0 * n0)
        grad[i] = (dvf_da * vF + vf * dvF_da) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    return _compute_bound(params[0], params[1], params[2])


def _objective_grad(params):
    bound, grad = _compute_bound_and_grad(params[0], params[1], params[2])
    return bound, grad


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Broad random sampling
    starts = []
    # Around the seed
    for _ in range(30):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, 0.15, 3)
        starts.append(s)
    # Wider exploration
    for _ in range(80):
        s = np.array([rng.uniform(-4.0, 3.0), rng.uniform(-0.8, 0.8), rng.uniform(-0.02, 0.02)])
        starts.append(s)
    # Structured grid
    for a0 in np.linspace(-3.5, 2.0, 15):
        for a4 in np.linspace(-0.5, 0.5, 10):
            a8 = 0.001 * rng.normal()
            starts.append(np.array([a0, a4, a8]))
    
    # Evaluate all starts
    for s in starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            val = _compute_bound(s[0], s[1], s[2])
            if val < best_val:
                best_val = val
                best_params = s.copy()
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B from best starts
    scored_starts = []
    for s in starts:
        try:
            val = _compute_bound(s[0], s[1], s[2])
            scored_starts.append((val, s))
        except Exception:
            pass
    scored_starts.sort(key=lambda x: x[0])
    
    top_starts = [s for _, s in scored_starts[:20]]
    top_starts.append(best_params.copy())
    
    for s in top_starts:
        if time.time() > _DEADLINE - 50:
            break
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global search
    if time.time() < _DEADLINE - 45:
        try:
            from scipy.optimize import differential_evolution
            bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.03, 0.03)]
            de_res = differential_evolution(_objective, bounds, maxiter=300,
                                           popsize=30, tol=1e-12, seed=42,
                                           polish=False, init='sobol')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish
    if time.time() < _DEADLINE - 15:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 300000, 'xatol': 1e-16, 'fatol': 1e-18})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS-B polish
    if time.time() < _DEADLINE - 5:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 30000, 'ftol': 1e-16, 'gtol': 1e-15})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Perturbation and re-optimization
    if time.time() < _DEADLINE - 3:
        try:
            for _ in range(12):
                perturbed = best_params + rng.normal(0, 1e-4, 3)
                res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                             jac=True, options={'maxiter': 15000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
