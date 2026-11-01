"""Analytic uncertainty bound with exact gradient + L-BFGS-B optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 200

# Precompute I[n] = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
# Need n up to 12 (since max index is k+i+j where k<=2, i,j<=6 => max 14, but we only need up to 12 for the actual sums)
def _precompute_I():
    I = np.zeros(15)
    sp = np.sqrt(np.pi)
    for n in range(15):
        if n == 0:
            I[n] = sp
        else:
            # (2n)! / (4^n * n!) * sqrt(pi)
            # Use log to avoid overflow
            log_val = 0.0
            for m in range(1, 2*n+1):
                log_val += np.log(m)
            log_val -= n * np.log(4.0)
            for m in range(1, n+1):
                log_val -= np.log(m)
            log_val += 0.5 * np.log(np.pi)
            I[n] = np.exp(log_val)
    return I

_I = _precompute_I()

# Precompute the derivative coefficient vectors
# da12/da_c for c = a0, a4, a8
_DA12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# dp_k/da_c for each k=0..6 and each c=0..2
# p0 = a0 + 3a4 + 105a8 + 945a12
# p1 = -6a4 - 420a8 - 7560a12
# p2 = a4 + 210a8 + 13860a12
# p3 = -28a8 - 6930a12
# p4 = a8 + 1155a12
# p5 = -66a12
# p6 = a12

# Direct coefficients (without a12 contribution)
_DIRECT = np.array([
    [1.0, 3.0, 105.0],   # p0
    [0.0, -6.0, -420.0],  # p1
    [0.0, 1.0, 210.0],    # p2
    [0.0, 0.0, -28.0],    # p3
    [0.0, 0.0, 1.0],      # p4
    [0.0, 0.0, 0.0],      # p5
    [0.0, 0.0, 0.0],      # p6
])

# Coefficient of a12 in each p_k
_A12_COEFF = np.array([945.0, -7560.0, 13860.0, -6930.0, 1155.0, -66.0, 1.0])

# dp_k/da_c = _DIRECT[k, c] + _A12_COEFF[k] * _DA12[c]
_DP = np.zeros((7, 3))
for k in range(7):
    for c in range(3):
        _DP[k, c] = _DIRECT[k, c] + _A12_COEFF[k] * _DA12[c]

# q_k = p_k * (-1)^k, so dq_k/da_c = dp_k/da_c * (-1)^k
_SGN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_DQ = _DP * _SGN[:, np.newaxis]


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    
    A(f)^2 = <x^2>/<1> for |g|^2
    A(F)^2 = <xi^2>/<1> for |G|^2
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    # Coefficients of P(x) = sum_{k=0}^{6} p_k x^{2k}
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    
    I = _I  # precomputed
    
    # m_k = sum_{i,j=0}^{6} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for i in range(7):
        for j in range(7):
            pp = p[i] * p[j]
            if pp == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    m[k] += pp * I[idx]
    
    # For Fourier transform: Q(xi) = sum q_k xi^{2k} where q_k = p_k * (-1)^k
    q = p * _SGN
    
    # n_k = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        for j in range(7):
            qq = q[i] * q[j]
            if qq == 0.0:
                continue
            for k in range(3):
                idx = k + i + j
                if idx < 15:
                    n[k] += 2.0*np.pi * qq * I[idx]
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Compute gradient analytically
    # d(bound)/d(a_c) = (1/(2*bound)) * (vF * d(vf)/d(a_c) + vf * d(vF)/d(a_c))
    # d(vf)/d(a_c) = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # d(vF)/d(a_c) = (dn2/da_c * n0 - n2 * dn0/da_c) / n0^2
    
    grad = np.zeros(3)
    for c in range(3):
        dp = _DP[:, c]
        
        # dm_k/da_c = 2*sum_i dp_i * sum_j p_j I[k+i+j]
        dm = np.zeros(3)
        for i in range(7):
            dpi = dp[i]
            if dpi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += p[j] * I[idx]
                dm[k] += 2.0 * dpi * s
        
        dq = _DQ[:, c]
        
        dn = np.zeros(3)
        for i in range(7):
            dqi = dq[i]
            if dqi == 0.0:
                continue
            for k in range(3):
                s = 0.0
                for j in range(7):
                    idx = k + i + j
                    if idx < 15:
                        s += q[j] * I[idx]
                dn[k] += 2.0 * 2.0*np.pi * dqi * s
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _objective(params):
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


def construct():
    # Multiple starting points
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),  # seed
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 0.1, 0.001]),
        np.array([-2.0, 0.05, 0.0005]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([1.0, -0.05, 0.0005]),
        np.array([-3.0, 0.15, 0.002]),
    ]
    
    rng = np.random.default_rng(123)
    for i in range(20):
        starts.append(rng.normal(0, 0.5, 3))
    
    best_val = np.inf
    best_params = None
    
    # Phase 1: L-BFGS-B from multiple starts
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-13})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish from best
    if best_params is not None and time.time() < _DEADLINE - 20:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision L-BFGS-B polish
    if best_params is not None and time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Additional local search with smaller steps
    if best_params is not None and time.time() < _DEADLINE - 5:
        # Try small perturbations around the best point
        rng2 = np.random.default_rng(456)
        step_sizes = [1e-4, 1e-5, 1e-6, 1e-7]
        for ss in step_sizes:
            if time.time() > _DEADLINE - 5:
                break
            for trial in range(50):
                if time.time() > _DEADLINE - 5:
                    break
                perturbed = best_params + rng2.normal(0, ss, 3)
                try:
                    res = minimize(_objective_grad, perturbed, method='L-BFGS-B',
                                  jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
                    if res.fun < best_val and np.isfinite(res.fun):
                        best_val = res.fun
                        best_params = res.x.copy()
                except Exception:
                    pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
