"""
Log-barrier interior-point minimization of uncertainty bound in reduced 2D space.
Uses the P(0)=0 constraint to eliminate a12, then optimizes over (a0, a4, a8)
with a log-barrier on the variances to enforce positivity and guide search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from math import lgamma, pi

_START_TIME = time.time()
_DEADLINE = _START_TIME + 190.0  # Safety margin before 210s kill

# Hermite polynomial values at 0
_H0_0 = 1.0
_H4_0 = 3.0
_H8_0 = 105.0
_H12_0 = 945.0

# Precompute the K matrices analytically
# K[i,j,0] = integral H_n H_m e^{-x^2} dx = sqrt(pi) 2^n n! delta_{nm}
# K[i,j,1] = integral x^2 H_n H_m e^{-x^2} dx

_IDX = [0, 4, 8, 12]
_N_IDX = len(_IDX)

def _hermite_norm(a):
    if a < 0:
        return 0.0
    return np.sqrt(pi) * (2.0 ** a) * np.exp(lgamma(a + 1))

# Build K0 (identity-like) and K1 (x^2 coupling)
_K0 = np.zeros((_N_IDX, _N_IDX))
_K1 = np.zeros((_N_IDX, _N_IDX))

for i, n in enumerate(_IDX):
    for j, m in enumerate(_IDX):
        _K0[i, j] = _hermite_norm(n) if n == m else 0.0
        # x^2 H_n = 0.25 H_{n+2} + (n+0.5) H_n + n(n-1) H_{n-2}
        # integral x^2 H_n H_m = 0.25 <H_{n+2},H_m> + (n+0.5)<H_n,H_m> + n(n-1)<H_{n-2},H_m>
        val = 0.0
        if n + 2 == m:
            val += 0.25 * _hermite_norm(n + 2)
        if n == m:
            val += (n + 0.5) * _hermite_norm(n)
        if n - 2 == m:
            val += n * (n - 1) * _hermite_norm(n - 2)
        _K1[i, j] = val

# Phase matrix for Fourier side
# phase[i,j] = (-1)^{(_IDX[i]+_IDX[j])/2}
_PHASE = np.zeros((_N_IDX, _N_IDX))
for i, n in enumerate(_IDX):
    for j, m in enumerate(_IDX):
        _PHASE[i, j] = (-1.0) ** ((n + m) // 2)

# Constraint: a12 = -(a0 + 3*a4 + 105*a8) / 945
# So c = [a0, a4, a8, a12] where a12 is determined by the first three.
# We'll work in the 3D space (a0, a4, a8) and compute c on the fly.

def _get_c(params):
    """Compute the 4 coefficients from the 3 free parameters."""
    a0, a4, a8 = params
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    return np.array([a0, a4, a8, a12])

def _compute_variances(params):
    """
    Compute Vx and Vxi from the 3 free parameters.
    Returns (Vx, Vxi) or (None, None) if invalid.
    """
    c = _get_c(params)
    
    # M0 = c^T K0 c, M2 = c^T K1 c
    M0 = c @ _K0 @ c
    M2 = c @ _K1 @ c
    
    # N0 = c^T (PHASE * K0) c, N2 = c^T (PHASE * K1) c
    N0 = c @ (_PHASE * _K0) @ c
    N2 = c @ (_PHASE * _K1) @ c
    
    if M0 <= 1e-15 or N0 <= 1e-15:
        return None, None
    
    Vx = M2 / M0
    Vxi = N2 / N0
    
    if Vx <= 0 or Vxi <= 0:
        return None, None
    
    return Vx, Vxi

def _log_barrier_objective(params, mu=0.01):
    """
    Log-barrier objective: minimize 0.5*(log(Vx) + log(Vxi)) + barrier terms.
    The barrier enforces Vx > 0 and Vxi > 0 by adding -mu*(log(Vx) + log(Vxi))
    as a penalty when they approach zero. Actually, to guide away from singularities,
    we add a barrier that goes to +infinity as Vx or Vxi -> 0+.
    
    We minimize: 0.5*(log(Vx) + log(Vxi)) - mu*(log(Vx) + log(Vxi))
    = (0.5 - mu)*(log(Vx) + log(Vxi))
    
    But this doesn't really help. Let me think differently.
    
    Better approach: minimize log(sqrt(Vx*Vxi)) = 0.5*(log(Vx)+log(Vxi))
    with a barrier that penalizes getting close to Vx=0 or Vxi=0.
    
    Actually, the cleanest is: minimize 0.5*(log(Vx)+log(Vxi)) + (1/Vx + 1/Vxi)*mu
    This adds a repulsive force as Vx or Vxi -> 0.
    
    Let me use: f = 0.5*(log(Vx) + log(Vxi)) + mu*(1/Vx + 1/Vxi)
    """
    Vx, Vxi = _compute_variances(params)
    if Vx is None or Vxi is None:
        return 1e18
    
    # Log objective
    f_log = 0.5 * (np.log(Vx) + np.log(Vxi))
    
    # Barrier term to keep away from singularities
    f_barrier = mu * (1.0 / Vx + 1.0 / Vxi)
    
    return f_log + f_barrier

def _log_barrier_grad(params, mu=0.01):
    """
    Gradient of the log-barrier objective.
    """
    Vx, Vxi = _compute_variances(params)
    if Vx is None or Vxi is None:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    c = _get_c(params)
    
    # Compute gradients of Vx and Vxi with respect to (a0, a4, a8)
    # dc/da_k for k in {0,1,2}
    # c = [a0, a4, a8, a12], a12 = -(a0+3a4+105a8)/945
    # dc/da0 = [1, 0, 0, -1/945]
    # dc/da4 = [0, 1, 0, -3/945]
    # dc/da8 = [0, 0, 1, -105/945]
    
    dc_da = np.array([
        [1.0, 0.0, 0.0, -1.0/945.0],
        [0.0, 1.0, 0.0, -3.0/945.0],
        [0.0, 0.0, 1.0, -105.0/945.0]
    ])
    
    # M0 = c^T K0 c, dM0/da_k = 2 * dc_k^T K0 c
    # M2 = c^T K1 c, dM2/da_k = 2 * dc_k^T K1 c
    # Vx = M2/M0, dVx/da_k = (dM2*M0 - M2*dM0)/M0^2
    
    M0 = c @ _K0 @ c
    M2 = c @ _K1 @ c
    N0 = c @ (_PHASE * _K0) @ c
    N2 = c @ (_PHASE * _K1) @ c
    
    dVx = np.zeros(3)
    dVxi = np.zeros(3)
    
    for k in range(3):
        dc_k = dc_da[k]
        
        dM0_k = 2.0 * (dc_k @ _K0 @ c)
        dM2_k = 2.0 * (dc_k @ _K1 @ c)
        
        dN0_k = 2.0 * (dc_k @ (_PHASE * _K0) @ c)
        dN2_k = 2.0 * (dc_k @ (_PHASE * _K1) @ c)
        
        dVx[k] = (dM2_k * M0 - M2 * dM0_k) / (M0 * M0)
        dVxi[k] = (dN2_k * N0 - N2 * dN0_k) / (N0 * N0)
    
    # f = 0.5*(log(Vx) + log(Vxi)) + mu*(1/Vx + 1/Vxi)
    # df/da_k = 0.5*(dVx[k]/Vx + dVxi[k]/Vxi) - mu*(dVx[k]/Vx^2 + dVxi[k]/Vxi^2)
    
    grad = 0.5 * (dVx / Vx + dVxi / Vxi) - mu * (dVx / (Vx * Vx) + dVxi / (Vxi * Vxi))
    
    f = 0.5 * (np.log(Vx) + np.log(Vxi)) + mu * (1.0 / Vx + 1.0 / Vxi)
    
    return f, grad

def construct():
    """
    Returns the three coefficients [a0, a4, a8] that minimize the uncertainty bound.
    """
    if time.time() > _DEADLINE:
        return np.array([-1.13, 0.04, 0.000308642])
    
    # Multi-start with log-barrier interior-point method
    # We'll use a sequence of decreasing mu values (path-following)
    
    best_val = np.inf
    best_params = None
    
    # Initial guesses
    starts = [
        np.array([-1.13, 0.04, 0.000308642]),
        np.array([-1.0, 0.05, 0.0005]),
        np.array([-1.5, 0.08, 0.0008]),
        np.array([-0.5, 0.02, 0.0002]),
        np.array([-2.0, 0.1, 0.001]),
        np.array([0.0, 0.0, 0.0]),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(8):
        starts.append(rng.normal(0, 0.5, 3))
    
    # Path-following: start with large mu, decrease
    mu_values = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    
    current_params = None
    
    for start in starts:
        if time.time() > _DEADLINE - 15:
            break
        
        # Try to find a valid starting point
        Vx, Vxi = _compute_variances(start)
        if Vx is None or Vxi is None:
            continue
        
        # Path-following from this start
        p = start.copy()
        for mu in mu_values:
            if time.time() > _DEADLINE - 10:
                break
            
            try:
                res = minimize(_log_barrier_grad, p, args=(mu,), method='L-BFGS-B',
                              jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < 1e17 and np.isfinite(res.fun):
                    p = res.x.copy()
                    # Check if this is better
                    Vx_f, Vxi_f = _compute_variances(p)
                    if Vx_f is not None and Vxi_f is not None:
                        val = np.sqrt(Vx_f * Vxi_f)
                        if val < best_val:
                            best_val = val
                            best_params = p.copy()
            except Exception:
                pass
        
        # Also try direct optimization without barrier for the final candidate
        if time.time() > _DEADLINE - 10:
            break
        
        try:
            def _direct_obj(params):
                Vx, Vxi = _compute_variances(params)
                if Vx is None or Vxi is None:
                    return 1e18
                return np.sqrt(Vx * Vxi)
            
            def _direct_grad(params):
                Vx, Vxi = _compute_variances(params)
                if Vx is None or Vxi is None:
                    return 1e18, np.array([1e18, 1e18, 1e18])
                
                c = _get_c(params)
                dc_da = np.array([
                    [1.0, 0.0, 0.0, -1.0/945.0],
                    [0.0, 1.0, 0.0, -3.0/945.0],
                    [0.0, 0.0, 1.0, -105.0/945.0]
                ])
                
                M0 = c @ _K0 @ c
                M2 = c @ _K1 @ c
                N0 = c @ (_PHASE * _K0) @ c
                N2 = c @ (_PHASE * _K1) @ c
                
                dVx = np.zeros(3)
                dVxi = np.zeros(3)
                
                for k in range(3):
                    dc_k = dc_da[k]
                    dM0_k = 2.0 * (dc_k @ _K0 @ c)
                    dM2_k = 2.0 * (dc_k @ _K1 @ c)
                    dN0_k = 2.0 * (dc_k @ (_PHASE * _K0) @ c)
                    dN2_k = 2.0 * (dc_k @ (_PHASE * _K1) @ c)
                    
                    dVx[k] = (dM2_k * M0 - M2 * dM0_k) / (M0 * M0)
                    dVxi[k] = (dN2_k * N0 - N2 * dN0_k) / (N0 * N0)
                
                bound = np.sqrt(Vx * Vxi)
                grad = (Vxi * dVx + Vx * dVxi) / (2.0 * bound)
                
                return bound, grad
            
            res = minimize(_direct_grad, p, method='L-BFGS-B',
                          jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return best_params
# EVOLVE-BLOCK-END
