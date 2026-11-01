import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195

# Precompute Hermite polynomial coefficients for H_0, H_4, H_8, H_12
# Using physicists' convention
_H_COEFFS = np.array([
    [1.0, 0, 0, 0, 0, 0, 0],           # H_0
    [12.0, -48.0, 16.0, 0, 0, 0, 0],   # H_4
    [1680.0, -13440.0, 13440.0, -3072.0, 256.0, 0, 0],  # H_8
    [665280.0, -6451200.0, 12902400.0, -6451200.0, 1290240.0, -98304.0, 4096.0]  # H_12
])

_H_AT_ZERO = np.array([1.0, 12.0, 1680.0, 665280.0])

# Precompute I_n = ∫ x^(2n) e^{-x^2} dx
def _compute_I_table():
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            log_num = sum(math.log(k) for k in range(1, 2*n+1))
            log_den = n * math.log(4) + sum(math.log(k) for k in range(1, n+1))
            I[n] = math.exp(log_num - log_den) * np.sqrt(np.pi)
    return I

_I_TABLE = _compute_I_table()

# Matrix M where M[k, i, j] = I_{k+i+j}
_M_SPATIAL = np.zeros((2, 7, 7))
_M_FREQ = np.zeros((2, 7, 7))
for k in range(2):
    for i in range(7):
        for j in range(7):
            idx = k + i + j
            if idx < 15:
                _M_SPATIAL[k, i, j] = _I_TABLE[idx]
                _M_FREQ[k, i, j] = 2.0 * np.pi * _I_TABLE[idx]

_SIGNS = np.array([(-1.0)**k for k in range(7)])

def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound sqrt(<x^2><xi^2>) and its analytic gradient.
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    a12 is determined by g(0)=0: a12 = -(a0 + 12*a4 + 1680*a8) / 665280
    """
    a0, a4, a8 = params[0], params[1], params[2]
    
    a12 = -(a0 + 12.0*a4 + 1680.0*a8) / 665280.0
    coeffs = np.array([a0, a4, a8, a12])
    
    p = _H_COEFFS @ coeffs
    
    m = np.array([p @ _M_SPATIAL[k] @ p for k in range(2)])
    m0, m2 = m[0], m[1]
    
    q = p * _SIGNS
    n = np.array([q @ _M_FREQ[k] @ q for k in range(2)])
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    da12_vec = -_H_AT_ZERO[:3] / _H_AT_ZERO[3]
    
    grad = np.zeros(3)
    for c_idx in range(3):
        dp = _H_COEFFS[c_idx] + da12_vec[c_idx] * _H_COEFFS[3]
        
        dm = 2.0 * np.array([dp @ _M_SPATIAL[k] @ p for k in range(2)])
        
        dq = dp * _SIGNS
        
        dn = 2.0 * np.array([dq @ _M_FREQ[k] @ q for k in range(2)])
        
        dvm2 = (dm[1]*m0 - m2*dm[0]) / (m0*m0)
        dvn2 = (dn[1]*n0 - n2*dn[0]) / (n0*n0)
        
        grad[c_idx] = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad

def _objective(params):
    val, _ = _compute_bound_and_grad(params)
    return val

def _objective_grad(params):
    val, grad = _compute_bound_and_grad(params)
    return val, grad

def _ratio_objective(uv):
    u, v = uv[0], uv[1]
    a0 = 1.0
    a4 = u
    a8 = v
    val, _ = _compute_bound_and_grad(np.array([a0, a4, a8]))
    return val

def _ratio_grad(uv):
    u, v = uv[0], uv[1]
    a0 = 1.0
    a4 = u
    a8 = v
    val, grad = _compute_bound_and_grad(np.array([a0, a4, a8]))
    return val, np.array([grad[1], grad[2]])

def _sphere_params_to_coeffs(theta, phi):
    """Convert spherical coordinates to (a0, a4, a8) with sum c_i^2 = 1."""
    a0 = np.cos(theta)
    a4 = np.sin(theta) * np.cos(phi)
    a8 = np.sin(theta) * np.sin(phi)
    return np.array([a0, a4, a8])

def _sphere_objective(sph):
    """Objective in spherical coordinates."""
    theta, phi = sph[0], sph[1]
    coeffs = _sphere_params_to_coeffs(theta, phi)
    val, _ = _compute_bound_and_grad(coeffs)
    return val

def construct():
    """
    Global optimization combining differential evolution, sphere-based local search,
    and L-BFGS-B refinement.
    Returns three coefficients (a0, a4, a8) for H0, H4, H8.
    """
    best_val = np.inf
    best_params = None
    
    def _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14):
        nonlocal best_val, best_params
        try:
            if method == 'L-BFGS-B':
                res = minimize(_objective_grad, start, method='L-BFGS-B',
                              jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
            elif method == 'Nelder-Mead':
                res = minimize(_objective, start, method='Nelder-Mead',
                              options={'maxiter': maxiter, 'xatol': 1e-16, 'fatol': 1e-16})
            else:
                return
            if res.fun < best_val and np.isfinite(res.fun):
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 1: Global search using differential_evolution in ratio space
    bounds_ratio = [(-0.5, 0.5), (-0.01, 0.01)]
    
    if time.time() < _DEADLINE - 60:
        try:
            de_result = differential_evolution(
                _ratio_objective,
                bounds_ratio,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-12,
                polish=True,
                init='sobol',
                updating='immediate'
            )
            if de_result.fun < best_val and np.isfinite(de_result.fun):
                u_best, v_best = de_result.x
                a0_best = 1.0
                a4_best = u_best
                a8_best = v_best
                best_val = de_result.fun
                best_params = np.array([a0_best, a4_best, a8_best])
        except Exception:
            pass
    
    # Phase 2: Sphere-based local search with adaptive step sizes
    if time.time() < _DEADLINE - 40:
        rng = np.random.default_rng(999)
        
        # Initialize on sphere from multiple starting points
        sphere_starts = []
        # Known good point normalized
        kg = np.array([-1.13, 0.04, 0.000308642])
        kg_norm = kg / np.linalg.norm(kg)
        # Convert to spherical coords
        theta0 = np.arccos(np.clip(kg_norm[0], -1, 1))
        phi0 = np.arctan2(kg_norm[2], kg_norm[1])
        sphere_starts.append([theta0, phi0])
        
        # Add DE result if available
        if best_params is not None:
            bp_norm = best_params / np.linalg.norm(best_params)
            theta1 = np.arccos(np.clip(bp_norm[0], -1, 1))
            phi1 = np.arctan2(bp_norm[2], bp_norm[1])
            sphere_starts.append([theta1, phi1])
        
        # Random sphere starts
        for _ in range(30):
            theta = rng.uniform(0, np.pi)
            phi = rng.uniform(-np.pi, np.pi)
            sphere_starts.append([theta, phi])
        
        # Adaptive step size local search on sphere
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for start_sph in sphere_starts:
            if time.time() > _DEADLINE - 35:
                break
            
            current_sph = np.array(start_sph, dtype=float)
            current_val = _sphere_objective(current_sph)
            
            for step in step_sizes:
                if time.time() > _DEADLINE - 35:
                    break
                
                improved = False
                for _ in range(50):
                    # Random perturbation on sphere
                    dtheta = rng.normal(0, step)
                    dphi = rng.normal(0, step)
                    
                    new_theta = current_sph[0] + dtheta
                    new_phi = current_sph[1] + dphi
                    
                    # Wrap phi
                    new_phi = (new_phi + np.pi) % (2*np.pi) - np.pi
                    new_theta = np.clip(new_theta, 1e-6, np.pi - 1e-6)
                    
                    new_sph = np.array([new_theta, new_phi])
                    new_val = _sphere_objective(new_sph)
                    
                    if new_val < current_val:
                        current_sph = new_sph
                        current_val = new_val
                        improved = True
                
                if not improved:
                    break
            
            # Convert best sphere point back to coefficients
            if current_val < best_val:
                best_params = _sphere_params_to_coeffs(current_sph[0], current_sph[1])
                best_val = current_val
    
    # Phase 3: Local refinement from multiple starting points
    known_good = np.array([-1.13, 0.04, 0.000308642])
    
    starts = []
    if best_params is not None:
        starts.append(best_params.copy())
    starts.append(known_good)
    
    for da0 in np.linspace(-0.5, 0.5, 7):
        for da4 in np.linspace(-0.05, 0.05, 7):
            for da8 in np.linspace(-0.0005, 0.0005, 5):
                starts.append(known_good + np.array([da0, da4, da8]))
    
    rng = np.random.default_rng(12345)
    for scale in [0.5, 1.0, 2.0, 5.0]:
        for _ in range(10):
            starts.append(rng.normal(0, scale, 3))
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        _try_optimize(start, method='L-BFGS-B', maxiter=5000, ftol=1e-16, gtol=1e-14)
    
    # Phase 4: Polish with Nelder-Mead and tighter L-BFGS-B
    if best_params is not None and time.time() < _DEADLINE - 15:
        _try_optimize(best_params, method='Nelder-Mead', maxiter=20000, ftol=1e-16, gtol=1e-14)
        
        if time.time() < _DEADLINE - 10:
            _try_optimize(best_params, method='L-BFGS-B', maxiter=10000, ftol=1e-18, gtol=1e-16)
    
    # Phase 5: Fine grid around best point if time remains
    if best_params is not None and time.time() < _DEADLINE - 5:
        bp = best_params
        fine_starts = []
        for da0 in np.linspace(-0.1, 0.1, 7):
            for da4 in np.linspace(-0.01, 0.01, 7):
                for da8 in np.linspace(-0.0001, 0.0001, 5):
                    fine_starts.append(bp + np.array([da0, da4, da8]))
        
        for start in fine_starts:
            if time.time() > _DEADLINE - 3:
                break
            _try_optimize(start, method='L-BFGS-B', maxiter=2000, ftol=1e-16, gtol=1e-14)
    
    if best_params is None:
        return np.array([-1.13, 0.04, 0.000308642])
        
    return best_params
