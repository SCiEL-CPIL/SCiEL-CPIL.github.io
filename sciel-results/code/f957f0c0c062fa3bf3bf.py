"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _E_squared(a, Om, w0, wa):
    """Compute E(a)^2 for CPL model."""
    Ode = 1.0 - Om
    if Ode < 0:
        Ode = 0.0
    term1 = Om / (a**3)
    term2 = Ode * (a**(-3.0*w0)) * np.exp(-3.0*wa*(1.0-a))
    return np.maximum(term1 + term2, 1e-30)

def _quadrature_value(nodes, weights, z, Om, w0, wa):
    """Compute quadrature approximation for luminosity distance."""
    a_vals = 1.0 / (1.0 + z * nodes)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    return (1.0 + z) * z * np.sum(weights / E)

def _reference_value_fast(z, Om, w0, wa, n_ref=3000):
    """High-accuracy reference using fine quadrature with vectorization."""
    z_grid = np.linspace(0, z, n_ref)
    a_vals = 1.0 / (1.0 + z_grid)
    E2 = _E_squared(a_vals, Om, w0, wa)
    E = np.sqrt(E2)
    f_vals = z_grid / (1.0 + z_grid) / E
    f_vals[0] = 0.0
    from scipy.integrate import trapezoid
    return trapezoid(f_vals, z_grid)

def _node_polynomial(z, nodes):
    """Compute prod_i (z - nodes[i])."""
    result = np.ones_like(z, dtype=float)
    for node in nodes:
        result *= (z - node)
    return result

def _rational_error_objective(x, portfolio, z_grid, ref_cache):
    """
    Zolotarev-type objective: minimize uniform norm of rational error.
    R(z) = (I_quad(z) - I_exact(z)) / prod(z - z_i)
    """
    nodes = x[:8]
    weights = x[8:]
    
    if np.any(nodes <= 1e-8) or np.any(nodes >= 1-1e-8):
        return 1e10
    if np.any(np.diff(nodes) <= 1e-10):
        return 1e10
    if np.any(weights <= 1e-10):
        return 1e10
    
    max_rational_err = 0.0
    for idx, (Om, w0, wa) in enumerate(portfolio):
        for j, z in enumerate(z_grid):
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            ref_val = ref_cache[idx, j]
            if abs(ref_val) < 1e-10:
                continue
            rel_err = (quad_val - ref_val) / ref_val * 100.0
            # Divide by node polynomial to get rational error
            poly_val = _node_polynomial(np.array([z]), nodes)[0]
            if abs(poly_val) < 1e-10:
                # z is near a node, skip or use large value
                rational_err = abs(rel_err) * 1000.0
            else:
                rational_err = abs(rel_err) / abs(poly_val)
            if rational_err > max_rational_err:
                max_rational_err = rational_err
    
    return max_rational_err

def _worst_relative_error(nodes, weights, portfolio, z_grid, ref_cache):
    """Compute worst relative error (actual metric)."""
    max_err = 0.0
    for idx, (Om, w0, wa) in enumerate(portfolio):
        for j, z in enumerate(z_grid):
            quad_val = _quadrature_value(nodes, weights, z, Om, w0, wa)
            ref_val = ref_cache[idx, j]
            if abs(ref_val) > 1e-10:
                rel_err = abs(quad_val - ref_val) / abs(ref_val) * 100.0
                if rel_err > max_err:
                    max_err = rel_err
    return max_err

def construct():
    t_start = time.time()
    deadline = 85.0
    
    # Portfolio of cosmologies
    portfolio = [
        (0.30, -1.0, 0.0),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.3),
        (0.27, -1.0, 0.0),
        (0.27, -1.2, 0.5),
        (0.27, -0.8, -0.5),
        (0.31, -1.1, 0.2),
        (0.31, -0.9, -0.2),
    ]
    
    # Redshift grid for evaluation
    z_grid = np.linspace(0.05, 3.0, 60)
    n_models = len(portfolio)
    n_z = len(z_grid)
    
    # Pre-compute reference values
    ref_cache = np.zeros((n_models, n_z))
    for i, (Om, w0, wa) in enumerate(portfolio):
        for j, z in enumerate(z_grid):
            ref_cache[i, j] = _reference_value_fast(z, Om, w0, wa, n_ref=2000)
    
    # Phase 1: Global search using differential evolution on rational error
    bounds_de = [(1e-4, 1-1e-4)]*8 + [(1e-6, 0.5)]*8
    
    def _de_objective(x):
        if time.time() - t_start > deadline - 20.0:
            return 1e10
        # Normalize weights to sum to 1
        w = x[8:]
        w = np.maximum(w, 1e-8)
        w = w / np.sum(w)
        nodes = np.sort(x[:8])
        # Ensure strict separation
        for i in range(1, 8):
            if nodes[i] - nodes[i-1] < 1e-8:
                nodes[i] = nodes[i-1] + 1e-8
        nodes = np.clip(nodes, 1e-4, 1-1e-4)
        return _rational_error_objective(np.concatenate([nodes, w]), portfolio, z_grid, ref_cache)
    
    x0_de = np.concatenate([(np.polynomial.legendre.leggauss(8)[0] + 1.0)/2.0, 
                             np.ones(8)/8.0])
    
    try:
        res_de = differential_evolution(_de_objective, bounds_de, 
                                        maxiter=50, popsize=15, tol=1e-6,
                                        seed=42, polish=False)
        if time.time() - t_start < deadline - 15.0:
            best_x = res_de.x.copy()
        else:
            best_x = x0_de.copy()
    except Exception:
        best_x = x0_de.copy()
    
    # Phase 2: Local refinement using SLSQP on rational error
    def _slsqp_objective(x):
        if time.time() - t_start > deadline - 5.0:
            return 1e10
        nodes = x[:8]
        weights = x[8:]
        
        if np.any(nodes <= 1e-6) or np.any(nodes >= 1-1e-6):
            return 1e10
        if np.any(np.diff(nodes) <= 1e-10):
            return 1e10
        if np.any(weights <= 1e-10):
            return 1e10
        
        return _rational_error_objective(x, portfolio, z_grid, ref_cache)
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[8:]) - 1.0},
    ]
    bounds_slsqp = [(1e-5, 1-1e-5)]*8 + [(1e-8, 0.5)]*8
    
    # Normalize initial weights
    w_init = best_x[8:]
    w_init = np.maximum(w_init, 1e-8)
    w_init = w_init / np.sum(w_init)
    best_x = np.concatenate([best_x[:8], w_init])
    
    best_obj = _slsqp_objective(best_x)
    
    while time.time() - t_start < deadline - 10.0:
        try:
            res = minimize(_slsqp_objective, best_x, method='SLSQP', 
                         bounds=bounds_slsqp, constraints=constraints,
                         options={'maxiter': 100, 'ftol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
            else:
                break
        except Exception:
            break
    
    # Phase 3: Final polish on actual worst relative error
    def _final_objective(x):
        if time.time() - t_start > deadline - 2.0:
            return 1e10
        nodes = x[:8]
        weights = x[8:]
        
        if np.any(nodes <= 1e-6) or np.any(nodes >= 1-1e-6):
            return 1e10
        if np.any(np.diff(nodes) <= 1e-10):
            return 1e10
        if np.any(weights <= 1e-10):
            return 1e10
        
        return _worst_relative_error(nodes, weights, portfolio, z_grid, ref_cache)
    
    try:
        res_final = minimize(_final_objective, best_x, method='SLSQP',
                           bounds=bounds_slsqp, constraints=constraints,
                           options={'maxiter': 50, 'ftol': 1e-14})
        if res.fun < best_obj:
            best_x = res_final.x
    except Exception:
        pass
    
    nodes = best_x[:8]
    weights = best_x[8:]
    
    # Final constraint enforcement
    weights = np.maximum(weights, 1e-10)
    weights = weights / np.sum(weights)
    nodes = np.sort(nodes)
    for i in range(1, 8):
        if nodes[i] - nodes[i-1] < 1e-10:
            nodes[i] = nodes[i-1] + 1e-10
    nodes = np.clip(nodes, 1e-8, 1-1e-8)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
