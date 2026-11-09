<<<<<<< SEARCH
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model.
    a is a numpy array.
    """
    Om_de = 1.0 - Om_m - Om_r
    # Vectorized operations
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    # Prevent division by zero or negative values due to numerical issues
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

def _true_dL_fast(z, w0, wa, num_points=4096):
    """
    Fast reference calculation of (1+z) * integral.
    Uses a dense fixed grid to avoid adaptive overhead of quad.
    """
    if z <= 0:
        return 0.0
    # Grid in z-space from 0 to z
    zs = np.linspace(0, z, num_points)
    # Integrand is 1/E(a(z'))
    a_vals = 1.0 / (1.0 + zs)
    E_vals = _E_cpl(a_vals, w0, wa)
    integrand = 1.0 / E_vals
    
    # Trapezoidal rule with high resolution is very accurate
    integral = np.trapz(integrand, zs)
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    """
    Approximate dL using the 8-node quadrature rule.
    """
    if z <= 0:
        return 0.0
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _objective(params):
    """
    Objective function to minimize: worst relative error.
    params: 8 node params (t), 8 weight params (v).
    """
    if _time_left() < 1.0:
        return 1e10
        
    # Decode nodes
    # t_i -> x_i sequence
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        # Sigmoid mapping to (0, 1)
        s = 1.0 / (1.0 + np.exp(-params[i]))
        # Scale step by remaining space
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    # Ensure strictly increasing and in (0,1)
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    # Decode weights
    w_raw = np.exp(params[8:16])
    w = w_raw / np.sum(w_raw)
    
    # Test set
    # We need a representative set of cosmologies.
    # Include LCDM, wCDM, and some extreme CPL cases.
    # The evaluator uses "disjoint public validation cosmologies".
    # We should sample a broad range of w0, wa.
    test_cosmos = [
        (0.0, 0.0),    # LCDM
        (-1.0, 0.0),   # Lambda
        (0.5, -1.0),   # Quintessence
        (-1.5, 0.5),   # Phantom
        (-0.5, 0.5),   
        (0.3, -0.3),   
        (-2.0, 1.0),   # More extreme phantom
        (1.0, -2.0)    # More extreme quintessence
    ]
    
    # Redshifts
    # Cover a wide range
    # The evaluator likely tests a specific set. We should cover 0 < z < maybe 10 or 20.
    # Low z is critical because relative error can be amplified if dL is small?
    # dL ~ z for small z.
    test_zs = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    max_err = 0.0
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL_fast(z, w0, wa)
            quad_val = _quad_dL(z, x, w, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                if rel_err > max_err:
                    max_err = rel_err
                    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initial guess
    # Start with Gauss-Legendre-like distribution
    # t params for nodes:
    # We want x_i roughly uniform or GL distributed.
    # Let's just start with random or simple linear spacing in t-space.
    
    # Better initial guess: Map GL nodes to t-space?
    # x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    # x_gl is in [-1, 1], map to [0, 1]: x = 0.5*(x_gl+1)
    # We need to find t such that the sequential sigmoid produces x.
    # This is an inverse problem.
    # Simpler: Just use a reasonable spread in t.
    # If t is large, s is close to 1.
    
    # Let's try to initialize t such that x_i are roughly evenly spaced.
    # x_i = i/8
    # We can solve for t_i approximately.
    # Or just use random initialization for DE, it handles it well.
    
    # Bounds for DE
    # t_i: -5 to 5 is a good range for sigmoid.
    # v_i: -5 to 5 is a good range for log-weights.
    bounds = [(-5, 5)] * 16
    
    # Differential Evolution
    # popsize=15, maxiter=50
    # This is ~750 evaluations.
    # Each evaluation is fast.
    
    best_result = None
    best_score = 1e10
    
    try:
        if _time_left() > 20:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=10, 
                tol=1e-8,
                polish=False # We will polish manually
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
    except Exception as e:
        pass
    
    # Refinement
    if best_result is not None and _time_left() > 10:
        try:
            # L-BFGS-B or Nelder-Mead
            # Nelder-Mead is robust for non-smooth worst-case objectives
            res2 = minimize(
                _objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception as e:
        pass
            
    # If DE failed, try a simple grid or fallback
    if best_result is None:
        # Fallback to Gauss-Legendre
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = 0.5 * (x_gl + 1.0)
        weights = 0.5 * w_gl
        weights = weights / np.sum(weights)
        return {"nodes": nodes, "weights": weights}

    # Decode final result
    t = best_result[:8]
    v = best_result[8:16]
    
    x = np.zeros(8)
    prev = 0.0
    for i in range(8):
        s = 1.0 / (1.0 + np.exp(-t[i]))
        remaining = 1.0 - prev
        step = s * remaining
        prev += step
        x[i] = prev
    x = np.clip(x, 1e-12, 1.0 - 1e-12)
    
    w_raw = np.exp(v)
    w = w_raw / np.sum(w_raw)
    
    # Final validation check
    # Ensure sum of weights is 1
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}
=======
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, linprog
import sys

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    """
    Compute E(a) = H(a)/H0 for CPL model.
    a is a numpy array.
    """
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

# Pre-compute reference integrals for a dense set of cosmologies and redshifts
# This is the key speedup: we compute expensive reference values once, then
# the objective only does fast quadrature evaluations.

# Define the test cosmologies (broad range of w0, wa)
_TEST_COSMOS = [
    (0.0, 0.0),     # LCDM
    (-1.0, 0.0),    # Lambda
    (0.5, -1.0),    # Quintessence
    (-1.5, 0.5),    # Phantom
    (-0.5, 0.5),
    (0.3, -0.3),
    (-2.0, 1.0),    # Extreme phantom
    (1.0, -2.0),    # Extreme quintessence
    (-0.8, -0.2),   # Near-CDM
    (0.2, 0.8),     # Near-CDM
    (-1.2, 0.3),    # Moderate phantom
    (0.7, -0.5),    # Moderate quintessence
]

# Redshifts: cover 0.01 to 20, with denser sampling at low z
_TEST_ZS = np.array([
    0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5,
    2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0
])

# Pre-compute reference integrals using high-resolution quadrature
# We'll store: for each (cosmo_idx, z_idx), the true integral value
# and the true dL value
_N_COSMOS = len(_TEST_COSMOS)
_N_ZS = len(_TEST_ZS)

# Reference integral: I(z, w0, wa) = ∫_0^z dζ / E(1/(1+ζ))
# dL/(c/H0) = (1+z) * I(z, w0, wa)
# Quadrature: I ≈ z * Σ w_i / E(1/(1+z*x_i))
# So dL_quad = (1+z) * z * Σ w_i / E(1/(1+z*x_i))

# We'll pre-compute the reference dL values
_REF_DL = np.zeros((_N_COSMOS, _N_ZS))

def _compute_references():
    """Pre-compute reference dL values using high-resolution trapezoidal rule."""
    num_points = 2048
    for ci, (w0, wa) in enumerate(_TEST_COSMOS):
        for zi, z in enumerate(_TEST_ZS):
            if z <= 0:
                _REF_DL[ci, zi] = 0.0
                continue
            zs = np.linspace(0, z, num_points)
            a_vals = 1.0 / (1.0 + zs)
            E_vals = _E_cpl(a_vals, w0, wa)
            integrand = 1.0 / E_vals
            integral = np.trapz(integrand, zs)
            _REF_DL[ci, zi] = (1.0 + z) * integral

# Compute references at module load time (one-time cost)
_compute_references()

def _quad_dL_vectorized(z, nodes, weights, w0, wa):
    """
    Approximate dL using the 8-node quadrature rule.
    Vectorized over redshifts.
    """
    if z.ndim == 0:
        z = z.reshape(1)
    # a_vals shape: (n_z, 8)
    a_vals = 1.0 / (1.0 + z[:, None] * nodes[None, :])
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights[None, :] / E_vals, axis=1)
    result = (1.0 + z) * z * integral_approx
    if result.size == 1:
        return result[0]
    return result

def _objective(params):
    """
    Objective function to minimize: worst relative error.
    params: 16 values = 8 node offsets + 8 log-weights
    
    Node encoding: x_i = 0.5 + 0.5 * sin(pi * (i + 0.5) / 8 + offset_i)
    This ensures nodes are in (0,1) and roughly Chebyshev-distributed.
    Actually, let's use a simpler and more flexible encoding:
    x_i = (i + 1 + offset_i) / (N + 1) where offset_i is in (-0.5, 0.5)
    This keeps nodes ordered if offsets are small.
    
    Better: Use the "cumulative sigmoid" approach but with a different
    initial distribution. Or just use direct nodes with bounds enforcement.
    
    Let's use: x_i = 0.5 * (1 + sin(pi * (i - 0.5) / N + phase_i))
    where phase_i are the optimization variables. This gives Chebyshev-like
    distribution with adjustable phases.
    """
    if _time_left() < 0.5:
        return 1e10
    
    # Decode nodes using Chebyshev-based parameterization
    # x_i = 0.5 * (1 + sin(pi * (i - 0.5) / 8 + phase_i))
    # phase_i in (-pi/2, pi/2) to keep x_i in (0, 1)
    phases = params[:8]
    x = 0.5 * (1.0 + np.sin(np.pi * (np.arange(8) - 0.5) / 8.0 + phases))
    # Clip to ensure strictly in (0,1)
    x = np.clip(x, 1e-10, 1.0 - 1e-10)
    # Ensure strictly increasing (should be if phases are small, but enforce)
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-10
    
    # Decode weights
    w_raw = np.exp(params[8:16])
    w = w_raw / np.sum(w_raw)
    
    # Compute worst-case relative error across all test cosmologies and redshifts
    max_err = 0.0
    for ci in range(_N_COSMOS):
        w0, wa = _TEST_COSMOS[ci]
        true_vals = _REF_DL[ci, :]
        quad_vals = _quad_dL_vectorized(_TEST_ZS, x, w, w0, wa)
        # Relative error where true_val > 0
        mask = true_vals > 1e-10
        if np.any(mask):
            rel_err = np.abs(quad_vals[mask] - true_vals[mask]) / true_vals[mask]
            err = np.max(rel_err)
            if err > max_err:
                max_err = err
    
    return max_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 85.0
    
    # Initial guess: Chebyshev nodes with zero phase, equal weights
    # x_i = 0.5 * (1 + sin(pi * (i - 0.5) / 8))
    init_phases = np.zeros(8)
    init_log_w = np.zeros(8)  # Equal weights
    init_params = np.concatenate([init_phases, init_log_w])
    
    # Bounds
    # Phases: (-pi/2, pi/2) to keep nodes in (0,1)
    # Log-weights: (-5, 5) for reasonable range
    bounds = [(-np.pi/2, np.pi/2)] * 8 + [(-5, 5)] * 8
    
    best_params = init_params.copy()
    best_score = _objective(init_params)
    
    # Phase 1: Differential Evolution (global search)
    try:
        if _time_left() > 30:
            res = differential_evolution(
                _objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=8,
                tol=1e-6,
                polish=False,
                init='sobol'
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass
    
    # Phase 2: Nelder-Mead refinement
    if _time_left() > 15:
        try:
            res2 = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B if we have time
    if _time_left() > 8:
        try:
            res3 = minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x
        except Exception:
            pass
    
    # Final refinement with multiple Nelder-Mead restarts if time allows
    if _time_left() > 5:
        for restart in range(3):
            if _time_left() < 2:
                break
            try:
                # Add small perturbation
                np.random.seed(42 + restart)
                perturbed = best_params + np.random.normal(0, 0.01, size=16)
                perturbed = np.clip(perturbed, [b[0] for b in bounds], [b[1] for b in bounds])
                res4 = minimize(
                    _objective, 
                    perturbed, 
                    method='Nelder-Mead', 
                    options={'maxiter': 100, 'xatol': 1e-13, 'fatol': 1e-15}
                )
                if res4.fun < best_score:
                    best_score = res4.fun
                    best_params = res4.x
            except Exception:
                pass
    
    # Decode final result
    phases = best_params[:8]
    x = 0.5 * (1.0 + np.sin(np.pi * (np.arange(8) - 0.5) / 8.0 + phases))
    x = np.clip(x, 1e-10, 1.0 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-10
    
    w_raw = np.exp(best_params[8:16])
    w = w_raw / np.sum(w_raw)
    
    # Final validation
    w = w / np.sum(w)
    
    return {"nodes": x, "weights": w}
>>>>>>> REPLACE
