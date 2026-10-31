"""Uncertainty inequality optimization with analytic gradient and multi-start local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 195  # 15s safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = np.array([math.sqrt(math.pi) * math.prod([(2*k-1)/2.0 for k in range(1, n+1)]) for n in range(25)])

# Precompute the Jacobian dp/da analytically
# p[k] as function of (a0, a4, a8) with a12 = -(a0 + 3*a4 + 105*a8)/945
# p[0] = 0 (constraint satisfied exactly)
# p[1] = -6*a4 - 420*a8 - 7560*a12
# p[2] = a4 + 210*a8 + 13860*a12
# p[3] = -28*a8 - 6930*a12
# p[4] = a8 + 1155*a12
# p[5] = -66*a12
# p[6] = a12

# da12/da = [-1/945, -3/945, -105/945]
_dp_da = np.array([
    [0.0, 0.0, 0.0],       # p[0] = 0 always
    [8.0, 18.0, 420.0],    # p[1]
    [-44.0/3.0, -43.0, -1330.0],  # p[2]
    [22.0/3.0, 22.0, 742.0],      # p[3]
    [-11.0/9.0, -11.0/3.0, -1154.0/9.0],  # p[4]
    [2.0/105.0, 2.0/15.0, 22.0/3.0],     # p[5]
    [-1.0/945.0, -1.0/315.0, -1.0/9.0]   # p[6]
])

# Signs for Fourier transform: q[k] = (-1)^k * p[k]
_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


def _compute_bound_and_grad(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(vf * vF) and its analytic gradient.
    vf = m2/m0, vF = n2/n0
    m_k = integral x^(2k) |P|^2 e^(-x^2) dx
    n_k = 2*pi * integral x^(2k) |Q|^2 e^(-x^2) dx
    """
    a = np.array([a0, a4, a8])
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Compute p coefficients
    p = np.zeros(7)
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    q = p * _signs
    
    # Compute m_k and n_k for k=0,1,2
    m = np.zeros(3)
    n = np.zeros(3)
    
    for i in range(7):
        for j in range(i, 7):  # Use symmetry
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            sym = 2.0 if i != j else 1.0
            for k in range(3):
                idx = k + i + j
                if idx < 25:
                    m[k] += sym * pij * _I[idx]
                    n[k] += sym * qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.zeros(3)
    if m2 <= 0 or n2 <= 0:
        return 1e18, np.zeros(3)
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    
    if not np.isfinite(bound):
        return 1e18, np.zeros(3)
    
    # Analytic gradient using chain rule
    # d(bound)/da_c = (1/(2*bound)) * (vF * d(vf)/da_c + vf * d(vF)/da_c)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/da_c = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    # dm_k/da_c = 2 * sum_{i,j} (dp_i/da_c * p_j + p_i * dp_j/da_c) * I[k+i+j]
    # By symmetry: dm_k/da_c = 2 * sum_i (dp_i/da_c) * sum_j p_j * I[k+i+j]
    
    grad = np.zeros(3)
    for c in range(3):
        dm_da = np.zeros(3)
        dn_da = np.zeros(3)
        for k in range(3):
            dm_k = 0.0
            dn_k = 0.0
            for i in range(7):
                dpc = _dp_da[i, c]
                if dpc == 0.0:
                    continue
                for j in range(7):
                    idx = k + i + j
                    if idx < 25:
                        dm_k += dpc * p[j] * _I[idx]
                        dn_k += dpc * q[j] * _I[idx]
            # Factor of 2 from symmetry in |P|^2
            dm_da[k] = 2.0 * dm_k
            dn_da[k] = 2.0 * dn_k
        
        # d(vf)/da_c
        dvmf = (dm_da[1] * m0 - m2 * dm_da[0]) / (m0 * m0)
        # d(vF)/da_c
        dvmF = (dn_da[1] * n0 - n2 * dn_da[0]) / (n0 * n0)
        
        grad[c] = (vF * dvmf + vf * dvmF) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    """Compute just the bound value."""
    a12 = -(params[0] + 3.0 * params[1] + 105.0 * params[2]) / 945.0
    p = np.zeros(7)
    p[1] = -6.0 * params[1] - 420.0 * params[2] - 7560.0 * a12
    p[2] = params[1] + 210.0 * params[2] + 13860.0 * a12
    p[3] = -28.0 * params[2] - 6930.0 * a12
    p[4] = params[2] + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    q = p * _signs
    
    m = np.zeros(3)
    n = np.zeros(3)
    for i in range(7):
        for j in range(i, 7):
            pij = p[i] * p[j]
            qij = q[i] * q[j]
            sym = 2.0 if i != j else 1.0
            for k in range(3):
                idx = k + i + j
                if idx < 25:
                    m[k] += sym * pij * _I[idx]
                    n[k] += sym * qij * _I[idx]
    n *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 1e-15 or n0 <= 1e-15 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 <= 0 or n2 <= 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    bound = np.sqrt(vf * vF)
    return bound if np.isfinite(bound) else 1e18


def _objective_grad(params):
    return _compute_bound_and_grad(params[0], params[1], params[2])


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 0.000308642])
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Focused multi-start with diverse scales
    starts = []
    # Seed neighborhood
    for _ in range(15):
        s = np.array([-1.13, 0.04, 0.000308642]) + rng.normal(0, [0.2, 0.1, 0.005], 3)
        starts.append(s)
    # Moderate exploration
    for _ in range(30):
        s = np.array([rng.uniform(-3.0, 2.0), rng.uniform(-0.5, 0.5), rng.uniform(-0.01, 0.01)])
        starts.append(s)
    # Coarse grid in a0, a4 with small a8
    for a0 in np.linspace(-2.5, 1.5, 8):
        for a4 in np.linspace(-0.3, 0.3, 6):
            starts.append(np.array([a0, a4, 0.0]))
    
    # Evaluate all starts
    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 80:
            break
        try:
            val = _objective(s)
            if np.isfinite(val) and val < 1e17:
                scored.append((val, s.copy()))
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    for val, s in scored[:5]:
        if val < best_val:
            best_val = val
            best_params = s
    
    # Phase 2: L-BFGS-B from top starts with tight tolerances
    top_n = min(12, len(scored))
    for i in range(top_n):
        if time.time() > _DEADLINE - 50:
            break
        s = scored[i][1]
        try:
            res = minimize(_objective_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 8000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Differential evolution (global search)
    if time.time() < _DEADLINE - 45:
        try:
            bounds = [(-4.0, 3.0), (-1.0, 1.0), (-0.02, 0.02)]
            de_res = differential_evolution(_objective, bounds, maxiter=150,
                                           popsize=20, tol=1e-14, seed=42,
                                           polish=False, init='sobol')
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
                # Polish DE result
                res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                              jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead polish (very tight)
    if time.time() < _DEADLINE - 15:
        try:
            nm_res = minimize(_objective, best_params, method='Nelder-Mead',
                            options={'maxiter': 100000, 'xatol': 1e-15, 'fatol': 1e-17,
                                     'adaptive': True})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS-B with extreme tightness
    if time.time() < _DEADLINE - 5:
        try:
            fin_res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                             jac=True, options={'maxiter': 30000, 'ftol': 1e-18, 'gtol': 1e-17})
            if fin_res.fun < best_val and np.isfinite(fin_res.fun):
                best_val = fin_res.fun
                best_params = fin_res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Multi-scale perturbation restart
    if time.time() < _DEADLINE - 2:
        try:
            for scale in [1e-3, 1e-4, 1e-5]:
                if time.time() > _DEADLINE - 2:
                    break
                for _ in range(5):
                    perturbed = best_params + rng.normal(0, scale, 3)
                    res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                                 jac=True, options={'maxiter': 8000, 'ftol': 1e-17, 'gtol': 1e-16})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
