import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _cosmology_E(z, w0, wa, Om_m=0.3):
    """CPL dark energy: E(z)^2 = Om_m*(1+z)^3 + (1-Om_m)*(1+z)^(3*(1+w0)) * (1 - wa*(1-1/(1+z)))^(1.5*wa)"""
    z = np.atleast_1d(z)
    one_plus_z = 1.0 + z
    matter = Om_m * one_plus_z**3
    de_base = (1.0 - Om_m) * one_plus_z**(3.0 * (1.0 + w0))
    de_arg = 1.0 - wa * (1.0 - 1.0 / one_plus_z)
    de_arg = np.maximum(de_arg, 1e-30)
    de = de_base * de_arg**(1.5 * wa)
    return np.sqrt(matter + de)

def _integrand_z(z, w0, wa, Om_m):
    """f(z) = z / E(z) for the integral I(zmax) = ∫_0^zmax z'/E(z') dz'"""
    if z < 1e-15:
        return 0.0
    E = _cosmology_E(z, w0, wa, Om_m)
    return z / E

def _true_integral(zmax, w0, wa, Om_m):
    """Adaptive quadrature reference."""
    if zmax < 1e-15:
        return 0.0
    result, _ = quad(_integrand_z, 0, zmax, args=(w0, wa, Om_m), limit=200, epsabs=1e-13, epsrel=1e-13)
    return result

def _rel_error(zmax, nodes, weights, w0, wa, Om_m):
    """Relative error in percent for quadrature vs true integral."""
    z_i = zmax * nodes
    E_i = _cosmology_E(z_i, w0, wa, Om_m)
    f_i = z_i / E_i
    I_quad = zmax * np.sum(weights * f_i)
    I_true = _true_integral(zmax, w0, wa, Om_m)
    dL_quad = (1.0 + zmax) * I_quad
    dL_true = (1.0 + zmax) * I_true
    if abs(dL_true) < 1e-15:
        return 0.0
    return abs(dL_quad - dL_true) / abs(dL_true) * 100.0

def _precompute_true_integrals(cosmologies, z_values):
    """Precompute all true integrals for speed."""
    true_integrals = {}
    for w0, wa, Om in cosmologies:
        for z in z_values:
            true_integrals[(w0, wa, Om, z)] = _true_integral(z, w0, wa, Om)
    return true_integrals

def _objective_smooth(params, cosmologies, z_values, true_integrals, beta=50.0):
    """
    Smooth surrogate for worst-case relative error using log-sum-exp.
    This is differentiable and allows better optimization.
    """
    s = params[:8]
    w = params[8:16]
    
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 1e-15):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    
    errors = []
    for w0, wa, Om in cosmologies:
        for z in z_values:
            z_i = z * nodes
            E_i = _cosmology_E(z_i, w0, wa, Om)
            f_i = z_i / E_i
            I_quad = z * np.sum(weights * f_i)
            I_true = true_integrals[(w0, wa, Om, z)]
            dL_quad = (1.0 + z) * I_quad
            dL_true = (1.0 + z) * I_true
            if abs(dL_true) < 1e-15:
                continue
            err = abs(dL_quad - dL_true) / abs(dL_true) * 100.0
            errors.append(err)
    
    if len(errors) == 0:
        return 0.0
    
    errors = np.array(errors)
    # Log-sum-exp for smooth max
    max_err = np.max(errors)
    if max_err < 1e-15:
        return 0.0
    # Smooth approximation to max: (1/beta) * log(sum(exp(beta * err)))
    shifted = beta * (errors - max_err)
    log_sum_exp = max_err + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    return log_sum_exp

def _objective_max(params, cosmologies, z_values, true_integrals):
    """True max objective for final evaluation."""
    s = params[:8]
    w = params[8:16]
    
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 1e-15):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    
    max_err = 0.0
    for w0, wa, Om in cosmologies:
        for z in z_values:
            z_i = z * nodes
            E_i = _cosmology_E(z_i, w0, wa, Om)
            f_i = z_i / E_i
            I_quad = z * np.sum(weights * f_i)
            I_true = true_integrals[(w0, wa, Om, z)]
            dL_quad = (1.0 + z) * I_quad
            dL_true = (1.0 + z) * I_true
            if abs(dL_true) < 1e-15:
                continue
            err = abs(dL_quad - dL_true) / abs(dL_true) * 100.0
            if err > max_err:
                max_err = err
    return max_err

def construct():
    """
    Eight-node positive quadrature rule for dimensionless luminosity distance.
    Starts from Gauss-Legendre and refines via smooth-objective optimization.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Gauss-Legendre nodes and weights on [0,1]
    x, w = np.polynomial.legendre.leggauss(8)
    nodes_gl = 0.5 * (x + 1.0)
    weights_gl = 0.5 * w
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Define test cosmologies and z-values
    cosmologies = [
        (0.0, 0.0, 0.3),
        (-0.5, 0.5, 0.3),
        (0.3, -0.6, 0.3),
        (-1.0, 0.0, 0.3),
        (-0.8, 0.3, 0.3),
        (0.0, -0.5, 0.3),
        (-0.5, 0.0, 0.3),
        (-1.0, 0.5, 0.3),
        (0.0, 0.3, 0.3),
        (-0.3, -0.3, 0.3),
    ]
    z_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    
    # Precompute true integrals
    true_integrals = _precompute_true_integrals(cosmologies, z_values)
    
    # Encode GL as parameters
    spacings_gl = np.diff(np.concatenate([[0], nodes_gl]))
    s_init = np.log(spacings_gl)
    w_init = np.log(weights_gl)
    x0 = np.concatenate([s_init, w_init])
    
    # Evaluate GL baseline with max objective
    best_params = x0.copy()
    best_score = _objective_max(x0, cosmologies, z_values, true_integrals)
    
    elapsed = time.time() - start_time
    print(f"GL baseline: {best_score:.2e}% after {elapsed:.1f}s", flush=True)
    
    # Phase 1: Nelder-Mead with smooth objective (coarse z-values for speed)
    if elapsed < deadline - 40:
        z_coarse = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
        true_integrals_coarse = _precompute_true_integrals(cosmologies, z_coarse)
        
        try:
            result = minimize(
                _objective_smooth, x0, method='Nelder-Mead',
                args=(cosmologies, z_coarse, true_integrals_coarse, 50.0),
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
            )
            elapsed = time.time() - start_time
            if result.fun < best_score:
                # Verify with full max objective
                full_score = _objective_max(result.x, cosmologies, z_values, true_integrals)
                if full_score < best_score:
                    best_score = full_score
                    best_params = result.x.copy()
                    print(f"NM phase: {best_score:.2e}% after {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"NM phase failed: {e}", flush=True)
    
    # Phase 2: L-BFGS-B with smooth objective (full z-values)
    elapsed = time.time() - start_time
    if elapsed < deadline - 25:
        try:
            result = minimize(
                _objective_smooth, best_params, method='L-BFGS-B',
                args=(cosmologies, z_values, true_integrals, 100.0),
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            elapsed = time.time() - start_time
            if result.fun < best_score:
                full_score = _objective_max(result.x, cosmologies, z_values, true_integrals)
                if full_score < best_score:
                    best_score = full_score
                    best_params = result.x.copy()
                    print(f"L-BFGS phase: {best_score:.2e}% after {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"L-BFGS phase failed: {e}", flush=True)
    
    # Phase 3: Final polish with tighter smooth objective
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            result = minimize(
                _objective_smooth, best_params, method='L-BFGS-B',
                args=(cosmologies, z_values, true_integrals, 200.0),
                options={'maxiter': 30, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            elapsed = time.time() - start_time
            if result.fun < best_score:
                full_score = _objective_max(result.x, cosmologies, z_values, true_integrals)
                if full_score < best_score:
                    best_score = full_score
                    best_params = result.x.copy()
                    print(f"Polish phase: {best_score:.2e}% after {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"Polish phase failed: {e}", flush=True)
    
    # Decode best parameters
    s = best_params[:8]
    w = best_params[8:16]
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
    # Final safety: if anything went wrong, fall back to GL
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0) and np.all(weights > 0)):
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Verify constraints
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(nodes > 0) and np.all(nodes < 1), "Nodes must be in (0,1)"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    
    elapsed = time.time() - start_time
    print(f"Final: {best_score:.2e}% after {elapsed:.1f}s", flush=True)
    
    return {"nodes": nodes, "weights": weights}
