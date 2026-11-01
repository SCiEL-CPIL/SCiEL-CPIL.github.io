"""Analytic uncertainty bound with reparameterization + differential evolution + gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = _T0 + 195

# Precompute I_n = integral x^{2n} e^{-x^2} dx = (2n)! / (4^n n!) * sqrt(pi)
def _precompute_I():
    sp = math.sqrt(math.pi)
    I = np.zeros(14)
    for n in range(14):
        num = 1.0
        for m in range(1, 2*n + 1):
            num *= m
        den = (4.0 ** n)
        for m in range(1, n + 1):
            den *= m
        I[n] = num / den * sp
    return I

_I_TABLE = _precompute_I()

# Coefficient matrix for p in terms of (a0, a4, a8, a12)
# p_k = sum_j P_COEFFS[k,j] * a_j
_P_COEFFS = np.array([
    [1.0, 3.0, 105.0, 945.0],
    [0.0, -6.0, -420.0, -7560.0],
    [0.0, 1.0, 210.0, 13860.0],
    [0.0, 0.0, -28.0, -6930.0],
    [0.0, 0.0, 1.0, 1155.0],
    [0.0, 0.0, 0.0, -66.0],
    [0.0, 0.0, 0.0, 1.0],
])

# a12 = -(a0 + 3*a4 + 105*a8) / 945
_A12_COEFFS = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])

# Precompute M matrices for efficient moment computation
_M_MATRICES = []
for k in range(3):
    Mk = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 14:
                Mk[i, j] = _I_TABLE[idx]
    _M_MATRICES.append(Mk)

# Q coefficients: q_k = p_k * (-1)^k
_SIGNS = np.array([(-1.0)**k for k in range(7)])

# Precompute DP for gradient: dp/da_c = P_COEFFS[:, c] + P_COEFFS[:, 3] * A12_COEFFS[c]
_DP = np.zeros((7, 3))
for c in range(3):
    _DP[:, c] = _P_COEFFS[:, c] + _P_COEFFS[:, 3] * _A12_COEFFS[c]


def _bound_and_grad(params):
    """
    Compute uncertainty product and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 = -(a0 + 3*a4 + 105*a8)/945  (enforces g(0)=0)
    """
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = _A12_COEFFS[0]*a0 + _A12_COEFFS[1]*a4 + _A12_COEFFS[2]*a8
    
    # Compute p vector
    p = _P_COEFFS[0]*a0 + _P_COEFFS[1]*a4 + _P_COEFFS[2]*a8 + _P_COEFFS[3]*a12
    
    # Compute q vector (Fourier domain)
    q = p * _SIGNS
    
    # Compute moments using precomputed matrices
    m0 = p @ _M_MATRICES[0] @ p
    m2 = p @ _M_MATRICES[1] @ p
    n0 = 2.0*math.pi * (q @ _M_MATRICES[0] @ q)
    n2 = 2.0*math.pi * (q @ _M_MATRICES[1] @ q)
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0 or not np.isfinite(vf) or not np.isfinite(vF):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = math.sqrt(vf * vF)
    
    # Compute gradient analytically
    grad = np.zeros(3)
    for c in range(3):
        dp = _DP[:, c]
        dq = dp * _SIGNS
        
        # dm_k/da_c = 2 * dp^T M_k p
        dm0 = 2.0 * (dp @ _M_MATRICES[0] @ p)
        dm2 = 2.0 * (dp @ _M_MATRICES[1] @ p)
        
        # dn_k/da_c = 2 * 2*pi * dq^T M_k q
        dn0 = 2.0 * 2.0*math.pi * (dq @ _M_MATRICES[0] @ q)
        dn2 = 2.0 * 2.0*math.pi * (dq @ _M_MATRICES[1] @ q)
        
        dvm2 = (dm2*m0 - m2*dm0) / (m0*m0)
        dvn2 = (dn2*n0 - n2*dn0) / (n0*n0)
        
        grad[c] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def _bound_only(params):
    """Compute just the bound value (for optimizers that don't use gradients)."""
    val, _ = _bound_and_grad(params)
    return val


def _objective_grad(params):
    val, grad = _bound_and_grad(params)
    return val, grad


# Reparameterization: u = a4/a0, v = a8/a0, so a0 is free scale
# Since the problem is scale-invariant, we can fix a0 = 1 and optimize (u, v)
# But a0 can be negative, so we use a0 = sign * |a0| or just let a0 be free
# Actually, let's keep 3D but use differential evolution as global search

def _reparam_to_uv(uv):
    """Convert ratio variables (u, v) to (a0, a4, a8) with a0 = -1 (since a0 < 0 typically)."""
    u, v = uv[0], uv[1]
    a0 = -1.0
    a4 = a0 * u
    a8 = a0 * v
    return np.array([a0, a4, a8])


def _uv_objective(uv):
    """Objective in (u, v) space with a0 fixed to -1."""
    params = _reparam_to_uv(uv)
    return _bound_only(params)


def construct():
    best_val = np.inf
    best_params = None
    
    # Phase 1: Dense 2D grid search over (a0, a4) for several a8 values
    a8_candidates = [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005]
    
    for a8_val in a8_candidates:
        if time.time() > _DEADLINE - 70:
            break
        a0_grid = np.linspace(-3.0, 1.0, 60)
        a4_grid = np.linspace(-0.2, 0.3, 60)
        
        # Vectorized evaluation
        A0, A4 = np.meshgrid(a0_grid, a4_grid, indexing='ij')
        A8 = np.full_like(A0, a8_val)
        params_flat = np.column_stack([A0.ravel(), A4.ravel(), A8.ravel()])
        
        # Compute a12 for all points
        a12_all = params_flat @ _A12_COEFFS
        
        # Compute p vectors
        P_all = params_flat @ _P_COEFFS.T + np.outer(a12_all, _P_COEFFS[3])
        
        # Compute q vectors
        Q_all = P_all * _SIGNS
        
        M0 = _M_MATRICES[0]
        M1 = _M_MATRICES[1]
        
        m0_all = np.einsum('ij,ij,jk,ik->i', P_all, P_all, M0, P_all)
        m2_all = np.einsum('ij,ij,jk,ik->i', P_all, P_all, M1, P_all)
        n0_all = 2.0*math.pi * np.einsum('ij,ij,jk,ik->i', Q_all, Q_all, M0, Q_all)
        n2_all = 2.0*math.pi * np.einsum('ij,ij,jk,ik->i', Q_all, Q_all, M1, Q_all)
        
        valid = (m0_all > 0) & (n0_all > 0) & np.isfinite(m0_all) & np.isfinite(n0_all)
        vf = np.where(valid, m2_all / m0_all, 1e18)
        vF = np.where(valid, n2_all / n0_all, 1e18)
        
        valid2 = valid & (vf > 0) & (vF > 0) & np.isfinite(vf) & np.isfinite(vF)
        bounds = np.where(valid2, np.sqrt(vf * vF), 1e18)
        
        idx_min = np.unravel_index(np.argmin(bounds), bounds.shape)
        if bounds[idx_min] < best_val and np.isfinite(bounds[idx_min]):
            best_val = bounds[idx_min]
            best_params = np.array([a0_grid[idx_min[0]], a4_grid[idx_min[1]], a8_val])
    
    # Phase 2: Differential evolution in (u, v) space for global search
    # u = a4/a0, v = a8/a0
    # Based on seed: a0 ~ -1.13, a4 ~ 0.04, a8 ~ 0.0003
    # u ~ 0.04/-1.13 ~ -0.035, v ~ 0.0003/-1.13 ~ -0.00027
    if time.time() < _DEADLINE - 60:
        # Bounds for u and v
        # a0 is typically around -1 to -3, a4 around -0.2 to 0.3, a8 around 0 to 0.01
        # So u = a4/a0 ranges from about -0.3 to 0.5
        # v = a8/a0 ranges from about 0 to 0.01 (since a8 >= 0 and a0 < 0, v <= 0)
        # Let's use wider bounds: u in [-1, 1], v in [-0.1, 0.1]
        bounds_uv = [(-1.0, 1.0), (-0.1, 0.1)]
        
        try:
            de_result = differential_evolution(
                _uv_objective,
                bounds_uv,
                maxiter=200,
                tol=1e-10,
                seed=42,
                popsize=30,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                best_val = de_result.fun
                best_params = _reparam_to_uv(de_result.x)
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B from best and multiple random starts
    if best_params is not None and time.time() < _DEADLINE - 30:
        starts = [best_params.copy()]
        rng = np.random.default_rng(123)
        for i in range(20):
            starts.append(best_params + rng.normal(0, 0.05, 3))
        
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
    
    # Phase 4: SLSQP from best and nearby points
    if best_params is not None and time.time() < _DEADLINE - 15:
        slsqp_starts = [best_params.copy()]
        rng2 = np.random.default_rng(456)
        for i in range(10):
            slsqp_starts.append(best_params + rng2.normal(0, 0.03, 3))
        
        for start in slsqp_starts:
            if time.time() > _DEADLINE - 10:
                break
            try:
                res = minimize(_objective_grad, start, method='SLSQP',
                              jac=True, options={'maxiter': 3000, 'ftol': 1e-15})
                if res.fun < best_val and np.isfinite(res.fun):
                    best_val = res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 5: Nelder-Mead polish
    if best_params is not None and time.time() < _DEADLINE - 5:
        try:
            res = minimize(_bound_only, best_params, method='Nelder-Mead',
                          options={'maxiter': 100000, 'xatol': 1e-18, 'fatol': 1e-20})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 6: Final L-BFGS-B polish with tight tolerances
    if best_params is not None and time.time() < _DEADLINE - 2:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 20000, 'ftol': 1e-18, 'gtol': 1e-17})
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
