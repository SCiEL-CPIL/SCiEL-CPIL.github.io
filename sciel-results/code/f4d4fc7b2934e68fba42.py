import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_N = 10
_START = time.time()
_DEADLINE = 600.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

# Precompute grid and projection indices
_x = np.arange(_N)
_y = np.arange(_N)
_X, _Y = np.meshgrid(_x, _y, indexing='xy')

# Projections
_val_d = _X - _Y
_val_m0 = _X
_val_m1 = _X + _Y
_val_m2 = _X + 2 * _Y

# Map to non-negative integers for bincount
_min_d = _val_d.min()
_inv_d = (_val_d - _min_d).astype(int)
_size_d = int(_val_d.max() - _min_d) + 1

_inv_m0 = _val_m0.astype(int)
_size_m0 = int(_val_m0.max()) + 1

_inv_m1 = _val_m1.astype(int)
_size_m1 = int(_val_m1.max()) + 1

_inv_m2 = _val_m2.astype(int)
_size_m2 = int(_val_m2.max()) + 1

# Flatten indices for fast access
_inv_d_flat = _inv_d.ravel()
_inv_m0_flat = _inv_m0.ravel()
_inv_m1_flat = _inv_m1.ravel()
_inv_m2_flat = _inv_m2.ravel()

# Precompute cell-to-bin mappings for gradient computation
# For each cell, we need to know which bins it contributes to
_cells = np.arange(_N * _N)
_cell_d_bin = _inv_d_flat
_cell_m0_bin = _inv_m0_flat
_cell_m1_bin = _inv_m1_flat
_cell_m2_bin = _inv_m2_flat

def _shannon_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-18]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def _compute_entropies(P):
    """
    P is a 10x10 array.
    Returns D, H0, H1, H2.
    """
    p_flat = P.ravel()
    
    p_d = np.bincount(_cell_d_bin, weights=p_flat, minlength=_size_d)
    D = _shannon_bits(p_d)
    
    p_m0 = np.bincount(_cell_m0_bin, weights=p_flat, minlength=_size_m0)
    H0 = _shannon_bits(p_m0)
    
    p_m1 = np.bincount(_cell_m1_bin, weights=p_flat, minlength=_size_m1)
    H1 = _shannon_bits(p_m1)
    
    p_m2 = np.bincount(_cell_m2_bin, weights=p_flat, minlength=_size_m2)
    H2 = _shannon_bits(p_m2)
    
    return D, H0, H1, H2, p_d, p_m0, p_m1, p_m2

def _ratio_from_P(P):
    D, H0, H1, H2, _, _, _, _ = _compute_entropies(P)
    M = max(H0, H1, H2)
    if M < 1e-9:
        return 0.0
    return D / M

def _param_to_P(params):
    """
    params: [A, B, C, D, E, F, G, H, I, J]
    log_w = A*X + B*Y + C*X*Y + D*X^2 + E*Y^2 + F*X^2*Y^2 + G*X^3 + H*Y^3 + I*X*Y^2 + J*X^2*Y
    """
    A, B, C, D, E, F, G, H, I, J = params
    log_w = (A * _X + B * _Y + C * _X * _Y + D * _X**2 + E * _Y**2 + 
             F * (_X**2) * (_Y**2) + G * _X**3 + H * _Y**3 + I * _X * _Y**2 + J * _X**2 * _Y)
    # Shift for stability
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        return np.ones((_N, _N)) / (_N * _N)
    return w / total

def _objective(params):
    P = _param_to_P(params)
    return _ratio_from_P(P)

def _objective_neg(params):
    val = _objective(params)
    if not np.isfinite(val):
        return 0.0
    return -val

def _objective_grad(params):
    """
    Numerical gradient using central differences.
    """
    val = _objective(params)
    if not np.isfinite(val):
        return 0.0, np.zeros(len(params))
    
    eps = 1e-6
    grad = np.zeros(len(params))
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        val_plus = _objective(params_plus)
        
        params_minus = params.copy()
        params_minus[i] -= eps
        val_minus = _objective(params_minus)
        
        if not np.isfinite(val_plus) or not np.isfinite(val_minus):
            grad[i] = 0.0
        else:
            grad[i] = (val_plus - val_minus) / (2 * eps)
            
    return -val, -grad

def _direct_optimize_P(P_init, max_iters=500, step_size=0.01):
    """
    Direct coordinate descent on the 100 weights.
    Uses binary search for optimal step size.
    """
    P = P_init.copy()
    best_val = _ratio_from_P(P)
    
    if _remaining() < 10:
        return P, best_val
    
    # Precompute active cells
    active = np.where(P > 1e-8)[0]
    if len(active) == 0:
        return P, best_val
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        if _remaining() < 15:
            break
        
        improved = False
        
        # Try random pairs of cells
        for _ in range(50):
            i = rng.integers(len(active))
            j = rng.integers(len(active))
            if i == j:
                continue
            
            ci = active[i]
            cj = active[j]
            
            # Try moving mass from cj to ci
            # Binary search for best delta
            lo = -P[cj]
            hi = P[ci]
            
            if hi <= lo:
                continue
            
            # Try a few delta values
            for _ in range(10):
                mid = (lo + hi) / 2
                if mid <= lo or mid >= hi:
                    break
                
                P_new = P.copy()
                P_new[ci] += mid
                P_new[cj] -= mid
                
                val = _ratio_from_P(P_new)
                
                # Also try the other direction
                P_new2 = P.copy()
                P_new2[ci] -= mid
                P_new2[cj] += mid
                val2 = _ratio_from_P(P_new2)
                
                if val > best_val + 1e-9:
                    P = P_new
                    best_val = val
                    improved = True
                    lo = mid
                elif val2 > best_val + 1e-9:
                    P = P_new2
                    best_val = val2
                    improved = True
                    hi = mid
                else:
                    # Neither direction improved, try smaller step
                    hi = mid if val > val2 else lo
                    lo = mid if val2 > val else lo
                    break
        
        if not improved:
            # Try smaller step sizes
            step_size *= 0.5
            if step_size < 1e-6:
                break
        else:
            step_size = 0.01
    
    return P, best_val

def _two_phase_optimize(P_init):
    """
    Two-phase optimization: smooth phase then exact phase.
    """
    # Phase 1: Smooth optimization with larger steps
    P_smooth, val_smooth = _direct_optimize_P(P_init, max_iters=200, step_size=0.02)
    
    if _remaining() < 30:
        return P_smooth, val_smooth
    
    # Phase 2: Exact optimization with smaller steps
    P_exact, val_exact = _direct_optimize_P(P_smooth, max_iters=300, step_size=0.005)
    
    return P_exact, val_exact

def arithmetic_kakeya_distribution() -> np.ndarray:
    # Initial best
    best_params = np.zeros(10)
    best_val = _objective(best_params)
    best_P = _param_to_P(best_params)
    
    # 1. Differential Evolution with more parameters
    bounds = [(-5, 5)] * 10
    
    try:
        if _remaining() > 120:
            res_de = differential_evolution(_objective_neg, bounds, 
                                            maxiter=80, popsize=30, 
                                            seed=42, tol=1e-10, 
                                            polish=False,
                                            mutation=(0.3, 0.9),
                                            recombination=0.7)
            if -res_de.fun > best_val:
                best_val = -res_de.fun
                best_params = res_de.x
                best_P = _param_to_P(best_params)
    except Exception:
        pass
    
    # 2. Local Optimization (L-BFGS-B) from multiple starts
    starts = [best_params]
    rng = np.random.default_rng(42)
    for _ in range(30):
        starts.append(rng.uniform(-4, 4, 10))
    
    for s in starts:
        if _remaining() < 30:
            break
        try:
            res = minimize(_objective_neg, s, method='L-BFGS-B', jac=_objective_grad,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if -res.fun > best_val:
                best_val = -res.fun
                best_params = res.x
                best_P = _param_to_P(best_params)
        except Exception:
            pass
            
    # 3. Two-phase direct optimization
    if _remaining() > 60:
        best_P, best_val_direct = _two_phase_optimize(best_P)
    
    # 4. Additional polish with very small steps
    if _remaining() > 20:
        best_P, best_val_polish = _direct_optimize_P(best_P, max_iters=200, step_size=0.001)
        if best_val_polish > best_val:
            best_val = best_val_polish
            best_P = best_P

    # Final safety checks
    best_P = np.clip(best_P, 0.0, None)
    s = np.sum(best_P)
    if s < 1e-9:
        best_P = np.ones((_N, _N)) / (_N * _N)
    else:
        best_P = best_P / s
        
    return best_P
