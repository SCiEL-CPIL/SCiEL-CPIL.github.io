"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _E_of_z(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model. Om0+Ol0=1, Ode0=0."""
    # For flat: Om0 + Ol0 = 1, dark energy is in Ol0
    # E^2 = Om0*(1+z)^3 + Ol0 * (1+z)^(3*(1+w0+w1*(1-1/(1+z)))) * exp(3*w1*(1 - 1/(1+z)))
    # Actually for CPL: w(a) = w0 + w1*(1-a), a = 1/(1+z)
    # rho_de(a) = rho_de0 * a^(-3(1+w0)) * exp(-3*w1*(1-a))
    # E^2 = Om0*a^-3 + Ol0*a^(-3(1+w0))*exp(-3*w1*(1-a))
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    # DE term
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(E2)

def _integrand(z, Om0, Ol0, w0, w1):
    """f(z) = z/(1+z) * 1/E(z)"""
    if z < 1e-15:
        return 0.0
    E = _E_of_z(z, Om0, Ol0, w0, w1)
    return z / (1.0 + z) / E

def _true_integral(z_max, Om0, Ol0, w0, w1, n_sub=200):
    """High-accuracy adaptive integration for reference."""
    from scipy.integrate import quad
    result, _ = quad(lambda z: _integrand(z, Om0, Ol0, w0, w1), 0, z_max, limit=200)
    return result

def _approx_integral(z_max, nodes, weights, Om0, Ol0, w0, w1):
    """Quadrature approximation."""
    z_eval = z_max * nodes
    total = 0.0
    for i in range(len(nodes)):
        total += weights[i] * _integrand(z_eval[i], Om0, Ol0, w0, w1)
    return total * z_max

def _objective(params, design_cosmos, z_max_list):
    """Worst relative error across design cosmologies and z values."""
    n = len(params) // 2
    nodes = params[:n]
    weights = params[n:]
    
    # Ensure nodes in (0,1) and increasing
    nodes = np.clip(nodes, 1e-10, 1.0 - 1e-10)
    # Sort and re-associate weights (this is a hack; better to parameterize differently)
    # Actually let's use a different parameterization
    
    max_err = 0.0
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            true_val = _true_integral(z_max, Om0, Ol0, w0, w1)
            if abs(true_val) < 1e-15:
                continue
            approx_val = _approx_integral(z_max, nodes, weights, Om0, Ol0, w0, w1)
            rel_err = abs(approx_val - true_val) / abs(true_val) * 100.0
            if rel_err > max_err:
                max_err = rel_err
    return max_err

def _objective_v2(params, design_cosmos, z_max_list):
    """Better parameterization: nodes via sigmoid, weights via exp."""
    n = 8
    # First n params: node positions (use tanh to map to (0,1))
    node_params = params[:n]
    # Next n params: log-weights
    log_w = params[n:]
    
    nodes = 0.5 + 0.5 * np.tanh(node_params)
    # Ensure strictly increasing: add small epsilon spacing
    # Actually tanh is monotonic so if node_params are increasing, nodes are increasing
    # But we need to enforce this. Let's use cumulative approach.
    
    weights = np.exp(log_w)
    w_sum = np.sum(weights)
    weights = weights / w_sum
    
    max_err = 0.0
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            true_val = _true_integral(z_max, Om0, Ol0, w0, w1)
            if abs(true_val) < 1e-15:
                continue
            approx_val = _approx_integral(z_max, nodes, weights, Om0, Ol0, w0, w1)
            rel_err = abs(approx_val - true_val) / abs(true_val) * 100.0
            if rel_err > max_err:
                max_err = rel_err
    return max_err

def construct():
    t_start = time.time()
    deadline = t_start + 75.0  # Safety margin before 90s
    
    # Start with Gauss-Legendre 8-point nodes and weights
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    # Map from [-1,1] to [0,1]
    gl_nodes_01 = 0.5 * (gl_nodes + 1.0)
    
    # Design cosmologies: span the expected parameter space
    # Om0 ~ [0.2, 0.35], Ol0 = 1-Om0, w0 ~ [-1.5, 0.5], w1 ~ [-1, 1]
    design_cosmos = [
        (0.3, 0.7, -1.0, 0.0),    # Lambda CDM
        (0.27, 0.73, -1.0, 0.0),  # Planck-like
        (0.3, 0.7, -1.0, 1.0),    # CPL positive w1
        (0.3, 0.7, -1.0, -1.0),   # CPL negative w1
        (0.2, 0.8, -1.2, 0.5),    # More DE, phantom-ish w0
        (0.35, 0.65, -0.8, 0.3),  # Less DE
        (0.25, 0.75, -1.5, 1.0),  # Phantom w0, positive w1
        (0.3, 0.7, -0.5, -0.5),   # Mildly varying
        (0.3, 0.7, -1.0, 0.5),    # Moderate CPL
        (0.3, 0.7, -1.5, -0.5),   # Phantom, negative w1
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    best_nodes = gl_nodes_01.copy()
    best_weights = gl_weights.copy()
    best_score = float('inf')
    
    # Precompute true integrals for design set
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            tv = _true_integral(z_max, Om0, Ol0, w0, w1)
            true_vals.append((Om0, Ol0, w0, w1, z_max, tv))
    
    def _fast_objective(params):
        n = 8
        node_params = params[:n]
        log_w = params[n:]
        
        nodes = 0.5 + 0.5 * np.tanh(node_params)
        # Enforce strictly increasing by adding cumulative small offsets
        # Sort nodes and keep corresponding weights? No, that breaks the mapping.
        # Instead, use the constraint that tanh is monotonic, so we need node_params sorted.
        # Alternative: just evaluate and if not sorted, return large penalty
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e6
        
        weights = np.exp(log_w)
        w_sum = np.sum(weights)
        weights = weights / w_sum
        
        max_err = 0.0
        for i, (Om0, Ol0, w0, w1, z_max, tv) in enumerate(true_vals):
            if abs(tv) < 1e-15:
                continue
            z_eval = z_max * nodes
            approx = z_max * np.sum(weights * np.array([_integrand(z_eval[j], Om0, Ol0, w0, w1) for j in range(8)]))
            rel_err = abs(approx - tv) / abs(tv) * 100.0
            if rel_err > max_err:
                max_err = rel_err
        return max_err
    
    # Initial params: map GL nodes to tanh^{-1} space
    init_node_params = np.arctanh(2.0 * best_nodes - 1.0)
    init_log_w = np.log(best_weights)
    x0 = np.concatenate([init_node_params, init_log_w])
    
    # Try Nelder-Mead (no gradient needed)
    try:
        result = minimize(_fast_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if result.fun < best_score:
            best_score = result.fun
            node_params = result.x[:8]
            log_w = result.x[8:]
            best_nodes = 0.5 + 0.5 * np.tanh(node_params)
            best_weights = np.exp(log_w)
            best_weights = best_weights / np.sum(best_weights)
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"nodes": best_nodes, "weights": best_weights}
    
    # Local refinement with L-BFGS-B
    try:
        result2 = minimize(_fast_objective, x0, method='L-BFGS-B',
                          options={'maxiter': 300, 'gtol': 1e-10})
        if result2.fun < best_score:
            best_score = result2.fun
            node_params = result2.x[:8]
            log_w = result2.x[8:]
            best_nodes = 0.5 + 0.5 * np.tanh(node_params)
            best_weights = np.exp(log_w)
            best_weights = best_weights / np.sum(best_weights)
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"nodes": best_nodes, "weights": best_weights}
    
    # Multi-start: try different initializations
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline:
            break
        # Random perturbation of GL nodes
        x0_r = x0.copy()
        x0_r[:8] += np.random.randn(8) * 0.3
        x0_r[8:] += np.random.randn(8) * 0.1
        
        try:
            result_r = minimize(_fast_objective, x0_r, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if result_r.fun < best_score:
                best_score = result_r.fun
                node_params = result_r.x[:8]
                log_w = result_r.x[8:]
                best_nodes = 0.5 + 0.5 * np.tanh(node_params)
                best_weights = np.exp(log_w)
                best_weights = best_weights / np.sum(best_weights)
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 5:
        try:
            node_params_final = np.arctanh(2.0 * best_nodes - 1.0)
            log_w_final = np.log(best_weights)
            x0_final = np.concatenate([node_params_final, log_w_final])
            result_f = minimize(_fast_objective, x0_final, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result_f.fun < best_score:
                best_score = result_f.fun
                node_params = result_f.x[:8]
                log_w = result_f.x[8:]
                best_nodes = 0.5 + 0.5 * np.tanh(node_params)
                best_weights = np.exp(log_w)
                best_weights = best_weights / np.sum(best_weights)
        except Exception:
            pass
    
    # Ensure nodes are strictly in (0,1) and increasing
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    # Re-sort if needed (shouldn't happen with our parameterization)
    order = np.argsort(best_nodes)
    best_nodes = best_nodes[order]
    best_weights = best_weights[order]
    
    # Normalize weights to sum to 1
    best_weights = best_weights / np.sum(best_weights)
    
    # Ensure positive
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
