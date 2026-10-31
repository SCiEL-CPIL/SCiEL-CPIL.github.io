"""Fourier-invariant Hermite-Gaussian test function: minimize the uncertainty
product via multi-start L-BFGS-B with analytic objective and finite-diff gradient."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 190.0  # safety margin before 210s kill


def _precompute_I():
    """Precompute I[n] = integral x^(2n) e^(-x^2) dx for n=0..24."""
    I = np.zeros(25)
    I[0] = np.sqrt(np.pi)
    for n in range(1, 25):
        I[n] = (2.0 * n - 1.0) / 2.0 * I[n - 1]
    return I


_I = _precompute_I()

# Hermite polynomial coefficients (physicists' convention):
# H_0(x) = 1
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945

# p_k coefficients for P(x) = sum_{k=0}^{6} p_k x^{2k}
# P = a0*H0 + a4*H4 + a8*H8 + a12*H12, a12 = -(a0+3*a4+105*a8)/945
#
# p_0 = a0 + 3a4 + 105a8 + 945*a12 = 0 (by constraint)
# p_1 = -6a4 - 420a8 - 7560*a12
# p_2 = a4 + 210a8 + 13860*a12
# p_3 = -28a8 - 6930*a12
# p_4 = a8 + 1155*a12
# p_5 = -66*a12
# p_6 = a12

# dp/da matrix: rows are p_k (k=0..6), columns are (a0, a4, a8)
# a12 = -(a0 + 3a4 + 105a8)/945
# da12/da0 = -1/945, da12/da4 = -3/945 = -1/315, da12/da8 = -105/945 = -1/9

_dp_da = np.zeros((7, 3))
_dp_da[0] = [0.0, 0.0, 0.0]  # p_0 = 0 always
_dp_da[1] = [-7560.0 * (-1.0/945.0), -6.0 - 7560.0*(-1.0/315.0), -420.0 - 7560.0*(-1.0/9.0)]
_dp_da[2] = [13860.0 * (-1.0/945.0), 1.0 + 13860.0*(-1.0/315.0), 210.0 + 13860.0*(-1.0/9.0)]
_dp_da[3] = [-6930.0 * (-1.0/945.0), -6930.0*(-1.0/315.0), -28.0 - 6930.0*(-1.0/9.0)]
_dp_da[4] = [1155.0 * (-1.0/945.0), 1155.0*(-1.0/315.0), 1.0 + 1155.0*(-1.0/9.0)]
_dp_da[5] = [-66.0 * (-1.0/945.0), -66.0*(-1.0/315.0), -66.0*(-1.0/9.0)]
_dp_da[6] = [-1.0/945.0, -1.0/315.0, -1.0/9.0]

# Precompute I[k+i+j] tables for k=0,1,2 and i,j in 0..6
# For m_k = sum_{i,j} p_i p_j I[k+i+j]
# We need I[k+i+j] for k in {0,1}, i,j in 0..6, so max index = 1+6+6 = 13 < 25. Good.

# Build matrices: M_k[i,j] = I[k+i+j] for k=0,1
_M0 = np.array([[_I[i+j] for j in range(7)] for i in range(7)])
_M1 = np.array([[_I[1+i+j] for j in range(7)] for i in range(7)])

# For the Fourier side, Q_k = (-1)^k p_k, so q_i = (-1)^i p_i
# n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j] = 2*pi * sum_{i,j} (-1)^{i+j} p_i p_j I[k+i+j]
# N_k[i,j] = (-1)^{i+j} I[k+i+j]
_N0 = np.array([[((-1.0)**(i+j)) * _I[i+j] for j in range(7)] for i in range(7)])
_N1 = np.array([[((-1.0)**(i+j)) * _I[1+i+j] for j in range(7)] for i in range(7)])

_TWO_PI = 2.0 * np.pi


def _compute_bound(params):
    """Compute sqrt(vf * vF) where vf = m2/m0, vF = n2/n0."""
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    
    # Compute p vector
    p = np.zeros(7)
    p[0] = 0.0  # by constraint
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    
    # m0 = p^T M0 p, m2 = p^T M1 p
    m0 = p @ _M0 @ p
    m2 = p @ _M1 @ p
    
    # n0 = 2*pi * p^T N0 p, n2 = 2*pi * p^T N1 p
    n0 = _TWO_PI * (p @ _N0 @ p)
    n2 = _TWO_PI * (p @ _N1 @ p)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    if m2 < 0 or n2 < 0:
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf < 0 or vF < 0:
        return 1e18
    
    bound = np.sqrt(vf * vF)
    return bound if np.isfinite(bound) else 1e18


def _compute_grad(params):
    """Compute gradient of bound using finite differences (robust)."""
    h = 1e-8
    grad = np.zeros(3)
    f0 = _compute_bound(params)
    for c in range(3):
        params_p = params.copy()
        params_m = params.copy()
        params_p[c] += h
        params_m[c] -= h
        fp = _compute_bound(params_p)
        fm = _compute_bound(params_m)
        grad[c] = (fp - fm) / (2.0 * h)
    return grad


def _objective_and_grad(params):
    f = _compute_bound(params)
    g = _compute_grad(params)
    return f, g


def construct():
    best_val = np.inf
    best_params = np.array([-1.13, 0.04, 1.0/3240.0])
    
    # Evaluate the seed value first
    seed_val = _compute_bound(best_params)
    if seed_val < best_val and np.isfinite(seed_val):
        best_val = seed_val
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    # Around seed with small perturbations
    for _ in range(30):
        s = best_params + rng.normal(0, 0.05, 3)
        starts.append(s)
    # Medium-scale exploration
    for _ in range(40):
        s = np.array([
            rng.uniform(-2.5, 1.5),
            rng.uniform(-0.3, 0.3),
            rng.uniform(-0.01, 0.01)
        ])
        starts.append(s)
    # Coarse grid
    for a0 in np.linspace(-2.0, 1.0, 8):
        for a4 in np.linspace(-0.2, 0.2, 6):
            for a8 in np.linspace(-0.005, 0.005, 4):
                starts.append(np.array([a0, a4, a8]))
    
    # Phase 1: Evaluate all starts, keep top candidates
    scored = []
    for s in starts:
        if time.time() > _DEADLINE - 60.0:
            break
        try:
            val = _compute_bound(s)
            if np.isfinite(val) and val < 1e17:
                scored.append((val, s))
        except Exception:
            pass
    
    scored.sort(key=lambda x: x[0])
    
    # Update best from scoring
    for val, s in scored[:5]:
        if val < best_val:
            best_val = val
            best_params = s.copy()
    
    # Phase 2: L-BFGS-B from top starts
    top_n = min(20, len(scored))
    for i in range(top_n):
        if time.time() > _DEADLINE - 20.0:
            break
        s = scored[i][1]
        try:
            res = minimize(_objective_and_grad, s, method='L-BFGS-B',
                          jac=True, options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead polish from best
    if time.time() < _DEADLINE - 10.0:
        try:
            nm_res = minimize(_compute_bound, best_params, method='Nelder-Mead',
                            options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20,
                                     'adaptive': True})
            if nm_res.fun < best_val and np.isfinite(nm_res.fun):
                best_val = nm_res.fun
                best_params = nm_res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision L-BFGS-B polish
    if time.time() < _DEADLINE - 5.0:
        try:
            final_res = minimize(_objective_and_grad, best_params, method='L-BFGS-B',
                                jac=True, options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16})
            if final_res.fun < best_val and np.isfinite(final_res.fun):
                best_val = final_res.fun
                best_params = final_res.x.copy()
        except Exception:
            pass
    
    # Ensure we return valid finite values
    if not np.isfinite(best_val) or best_val >= 1e17:
        return np.array([-1.13, 0.04, 1.0/3240.0])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
