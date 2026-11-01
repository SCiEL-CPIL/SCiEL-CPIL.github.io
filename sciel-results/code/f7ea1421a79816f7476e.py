"""Analytic uncertainty bound with vectorized gradient + Differential Evolution + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_DEADLINE = time.time() + 200

# Precompute I_n = integral x^{2n} e^{-x^2} dx for n=0..14
# I_n = (2n)! / (4^n n!) * sqrt(pi)
_I_vals = np.zeros(15)
for n in range(15):
    # Use log to avoid overflow, though 28! is manageable
    # (2n)! / (4^n n!) = product_{k=1}^n (2n - 2k + 1) / 2^k ? No.
    # (2n)! / n! = 2 * 4 * ... * (2n)
    # (2n)! / (4^n n!) = (2n)! / (2^{2n} n!)
    # Let's just compute directly for n <= 14
    num = 1.0
    for i in range(1, 2*n + 1):
        num *= i
    den = (4.0 ** n)
    for i in range(1, n + 1):
        den *= i
    _I_vals[n] = (num / den) * np.sqrt(np.pi)

# Build matrices Q_k for k=0,1,2 (corresponding to moments m_0, m_2, m_4)
# (Q_k)_{ij} = I_{k+i+j} for i,j in 0..6
_Q_mats = []
for k in range(3):
    Q = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                Q[i, j] = _I_vals[idx]
    _Q_mats.append(Q)

# For Fourier transform, q_k = p_k * (-1)^k
# n_k = 2*pi * sum q_i q_j I_{k+i+j}
# Let D = diag((-1)^k). q = D p.
# n_k = 2*pi * p^T D Q_k D p
_Q_mats_F = []
D = np.diag([(-1.0)**k for k in range(7)])
for k in range(3):
    QF = 2.0 * np.pi * (D @ _Q_mats[k] @ D)
    _Q_mats_F.append(QF)

def _compute_p(a0, a4, a8):
    """Compute polynomial coefficients p[0..6] given a0, a4, a8."""
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    p = np.zeros(7)
    p[0] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    p[1] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    p[2] = a4 + 210.0*a8 + 13860.0*a12
    p[3] = -28.0*a8 - 6930.0*a12
    p[4] = a8 + 1155.0*a12
    p[5] = -66.0*a12
    p[6] = a12
    return p

def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    params = [a0, a4, a8]
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    # Compute p
    p = _compute_p(a0, a4, a8)
    
    # Compute moments m_k = p^T Q_k p
    m0 = p @ _Q_mats[0] @ p
    m2 = p @ _Q_mats[1] @ p
    m4 = p @ _Q_mats[2] @ p
    
    # Compute moments n_k for Fourier
    n0 = p @ _Q_mats_F[0] @ p
    n2 = p @ _Q_mats_F[1] @ p
    n4 = p @ _Q_mats_F[2] @ p
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
        
    bound = np.sqrt(vf * vF)
    
    # Gradient
    # d(bound)/da_c = (1/(2*bound)) * (vF * d(vf)/da_c + vf * d(vF)/da_c)
    # d(vf)/da_c = (dm2/da_c * m0 - m2 * dm0/da_c) / m0^2
    # dm_k/da_c = 2 * (Q_k p) dot dp/da_c
    
    # dp/da_c matrix (7x3)
    # p0 = a0 + 3a4 + 105a8 + 945a12
    # p1 = -6a4 - 420a8 - 7560a12
    # ...
    # Let's compute dp/da_c explicitly
    dp = np.zeros((7, 3))
    
    # da12/da = [-1/945, -3/945, -105/945]
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    # p0
    dp[0, 0] = 1.0 + 945.0*da12[0]
    dp[0, 1] = 3.0 + 945.0*da12[1]
    dp[0, 2] = 105.0 + 945.0*da12[2]
    
    # p1
    dp[1, 0] = 0.0 - 7560.0*da12[0]
    dp[1, 1] = -6.0 - 7560.0*da12[1]
    dp[1, 2] = -420.0 - 7560.0*da12[2]
    
    # p2
    dp[2, 0] = 0.0 + 13860.0*da12[0]
    dp[2, 1] = 1.0 + 13860.0*da12[1]
    dp[2, 2] = 210.0 + 13860.0*da12[2]
    
    # p3
    dp[3, 0] = 0.0 - 6930.0*da12[0]
    dp[3, 1] = 0.0 - 6930.0*da12[1]
    dp[3, 2] = -28.0 - 6930.0*da12[2]
    
    # p4
    dp[4, 0] = 0.0 + 1155.0*da12[0]
    dp[4, 1] = 0.0 + 1155.0*da12[1]
    dp[4, 2] = 1.0 + 1155.0*da12[2]
    
    # p5
    dp[5, 0] = 0.0 - 66.0*da12[0]
    dp[5, 1] = 0.0 - 66.0*da12[1]
    dp[5, 2] = 0.0 - 66.0*da12[2]
    
    # p6
    dp[6, 0] = 0.0 + 1.0*da12[0]
    dp[6, 1] = 0.0 + 1.0*da12[1]
    dp[6, 2] = 0.0 + 1.0*da12[2]
    
    grad = np.zeros(3)
    
    # Precompute Q_k p and QF_k p
    Qp_m = [Q @ p for Q in _Q_mats]
    Qp_F = [Q @ p for Q in _Q_mats_F]
    
    for c in range(3):
        # dm_k/da_c = 2 * (Q_k p) dot dp_c
        dm0 = 2.0 * np.dot(Qp_m[0], dp[:, c])
        dm2 = 2.0 * np.dot(Qp_m[1], dp[:, c])
        
        # dn_k/da_c = 2 * (QF_k p) dot dp_c  (Note: QF already includes 2*pi)
        # Wait, n_k = p^T QF_k p. d(n_k) = 2 p^T QF_k dp = 2 (QF_k p) dot dp.
        dn0 = 2.0 * np.dot(Qp_F[0], dp[:, c])
        dn2 = 2.0 * np.dot(Qp_F[1], dp[:, c])
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
        
    return bound, grad

def _objective(params):
    val, _ = _bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad

def construct():
    # Define bounds for differential evolution
    # Based on seed and typical scales
    bounds = [(-10.0, 10.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    best_val = np.inf
    best_params = None
    
    # 1. Differential Evolution for global search
    if time.time() < _DEADLINE - 60:
        try:
            de_res = differential_evolution(_objective, bounds, seed=42, maxiter=1000, tol=1e-12, polish=False)
            if de_res.fun < best_val and np.isfinite(de_res.fun):
                best_val = de_res.fun
                best_params = de_res.x.copy()
        except Exception:
            pass

    # 2. Local polish with L-BFGS-B from DE result and seed
    starts = []
    if best_params is not None:
        starts.append(best_params)
    starts.append(np.array([-1.13, 0.04, 0.000308642])) # Seed
    
    # Add some random starts if time permits
    rng = np.random.default_rng(123)
    for i in range(10):
        starts.append(rng.uniform(bounds[0][0], bounds[0][1], size=3))
    
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
        try:
            res = minimize(_objective_grad, start, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    # 3. Final high-precision polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return np.array(best_params)
# EVOLVE-BLOCK-END
