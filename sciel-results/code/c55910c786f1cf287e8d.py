"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_DEADLINE = time.time() + 85.0  # 5s safety margin before 90s kill

def _E_of_z(z, Om0, Ol0, w0, w1):
    """E(z) = H(z)/H0 for flat CPL model.
    w(a) = w0 + w1*(1-a), a = 1/(1+z)
    rho_de(a) = rho_de0 * a^(-3(1+w0)) * exp(-3*w1*(1-a))
    E^2 = Om0*a^-3 + Ol0 * a^(-3(1+w0)) * exp(-3*w1*(1-a))
    """
    a = 1.0 / (1.0 + z)
    a3 = a ** 3
    exponent = -3.0 * w1 * (1.0 - a)
    de_factor = a ** (-3.0 * (1.0 + w0)) * np.exp(exponent)
    E2 = Om0 / a3 + Ol0 * de_factor
    return np.sqrt(np.maximum(E2, 1e-30))

def _integrand(z, Om0, Ol0, w0, w1):
    """f(z) = z/(1+z) * 1/E(z) for the luminosity distance integral."""
    if z < 1e-15:
        return 0.0
    E = _E_of_z(z, Om0, Ol0, w0, w1)
    return z / (1.0 + z) / E

def _true_integral(z_max, Om0, Ol0, w0, w1):
    """High-accuracy adaptive integration for reference."""
    result, _ = quad(lambda z: _integrand(z, Om0, Ol0, w0, w1), 0, z_max, limit=200)
    return result

def _approx_integral(z_max, nodes, weights, Om0, Ol0, w0, w1):
    """Quadrature approximation: (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i]), so z_i = z*nodes[i].
    dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    = z * sum_i weights[i] * (1+z*nodes[i]) / E(z*nodes[i])  ... wait.
    
    Actually: dL/(c/H0) = (1+z) * integral_0^z dz'/E(z')
    The integral is approximated by z * sum_i weights[i] * f(nodes[i])
    where f(t) = 1/E(z*t) ... no.
    
    Let me re-read the task:
    dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    where a_i = 1/(1+z*nodes[i])
    
    So the formula is: (1+z) * z * sum_i weights[i] / E(a_i)
    where E(a_i) = H(a_i)/H0 and a_i = 1/(1+z*nodes[i]).
    
    This means the integrand being quadrature'd is 1/E(a) with respect to some variable.
    Let u = z' / z, so z' = z*u, dz' = z*du, u in [0,1].
    integral_0^z dz'/E(z') = integral_0^1 z*du/E(z*u) = z * integral_0^1 du/E(z*u)
    
    So dL/(c/H0) = (1+z) * z * integral_0^1 du / E(z*u)
    
    The quadrature approximates integral_0^1 du/E(z*u) ≈ sum_i weights[i] / E(z*nodes[i])
    where E(z*nodes[i]) = E(a_i) with a_i = 1/(1+z*nodes[i]).
    
    So: dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(z*nodes[i])
    """
    z_eval = z_max * nodes
    E_vals = _E_of_z(z_eval, Om0, Ol0, w0, w1)
    return (1.0 + z_max) * z_max * np.sum(weights / E_vals)

def _relative_error(nodes, weights, z_max, Om0, Ol0, w0, w1, true_val):
    """Compute relative error in percent."""
    approx = _approx_integral(z_max, nodes, weights, Om0, Ol0, w0, w1)
    if abs(true_val) < 1e-15:
        return 0.0
    return abs(approx - true_val) / abs(true_val) * 100.0

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # safety margin
    
    # Design cosmologies spanning expected parameter space
    design_cosmos = [
        (0.3, 0.7, -1.0, 0.0),    # Lambda CDM
        (0.27, 0.73, -1.0, 0.0),  # Planck-like
        (0.3, 0.7, -1.0, 1.0),    # CPL positive w1
        (0.3, 0.7, -1.0, -1.0),   # CPL negative w1
        (0.2, 0.8, -1.2, 0.5),    # More DE
        (0.35, 0.65, -0.8, 0.3),  # Less DE
        (0.25, 0.75, -1.5, 1.0),  # Phantom w0
        (0.3, 0.7, -0.5, -0.5),   # Mildly varying
        (0.3, 0.7, -1.0, 0.5),    # Moderate CPL
        (0.3, 0.7, -1.5, -0.5),   # Phantom, negative w1
        (0.3, 0.7, 0.0, 0.0),     # w=0 (quintessence-like)
        (0.3, 0.7, -0.5, 0.5),    # Another CPL
    ]
    z_max_list = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    # Precompute true integrals
    true_vals = []
    for (Om0, Ol0, w0, w1) in design_cosmos:
        for z_max in z_max_list:
            tv = _true_integral(z_max, Om0, Ol0, w0, w1)
            true_vals.append((Om0, Ol0, w0, w1, z_max, tv))
    
    # Initialize with Gauss-Legendre 8-point
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    best_nodes = 0.5 * (gl_nodes + 1.0)
    best_weights = gl_weights.copy()
    
    # Fast objective using precomputed true values
    def _fast_objective(params):
        n = 8
        node_params = params[:n]
        log_w = params[n:]
        
        nodes = 0.5 + 0.5 * np.tanh(node_params)
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e6
        
        weights = np.exp(log_w)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return 1e6
        weights = weights / w_sum
        
        max_err = 0.0
        for (Om0, Ol0, w0, w1, z_max, tv) in true_vals:
            if abs(tv) < 1e-15:
                continue
            approx = _approx_integral(z_max, nodes, weights, Om0, Ol0, w0, w1)
            rel_err = abs(approx - tv) / abs(tv) * 100.0
            if rel_err > max_err:
                max_err = rel_err
                if max_err > 10.0:  # early exit if bad
                    break
        return max_err
    
    # Initial params from GL
    init_node_params = np.arctanh(np.clip(2.0 * best_nodes - 1.0, -0.999, 0.999))
    init_log_w = np.log(best_weights)
    x0 = np.concatenate([init_node_params, init_log_w])
    
    best_score = _fast_objective(x0)
    
    # Nelder-Mead optimization
    try:
        result = minimize(_fast_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
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
    
    # L-BFGS-B refinement
    try:
        result2 = minimize(_fast_objective, x0, method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-12})
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
    
    # Multi-start with different initializations
    np.random.seed(42)
    for restart in range(8):
        if time.time() > deadline - 5.0:
            break
        
        # Different starting points
        if restart == 0:
            # Equally spaced
            nodes_init = (np.arange(8) + 0.5) / 8.0
        elif restart == 1:
            # Chebyshev nodes
            k = np.arange(8)
            nodes_init = 0.5 * (1.0 - np.cos(np.pi * (2*k+1) / 16.0))
        elif restart == 2:
            # Slightly perturbed GL
            nodes_init = best_nodes + 0.02 * np.random.randn(8)
        elif restart == 3:
            # More clustered near 0
            nodes_init = np.sort(np.linspace(0.05, 0.95, 8) + 0.01 * np.random.randn(8))
        else:
            nodes_init = np.sort(0.1 + 0.8 * np.random.rand(8))
        
        nodes_init = np.clip(nodes_init, 0.01, 0.99)
        # Ensure strictly increasing
        for i in range(7):
            if nodes_init[i+1] <= nodes_init[i]:
                nodes_init[i+1] = nodes_init[i] + 0.01
        
        weights_init = np.random.dirichlet(np.ones(8))
        
        node_params_init = np.arctanh(np.clip(2.0 * nodes_init - 1.0, -0.999, 0.999))
        log_w_init = np.log(weights_init)
        x_init = np.concatenate([node_params_init, log_w_init])
        
        try:
            res = minimize(_fast_objective, x_init, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                node_params = res.x[:8]
                log_w = res.x[8:]
                best_nodes = 0.5 + 0.5 * np.tanh(node_params)
                best_weights = np.exp(log_w)
                best_weights = best_weights / np.sum(best_weights)
        except Exception:
            pass
    
    # Final fine-tuning with more iterations
    if time.time() < deadline - 5.0:
        try:
            init_node_params = np.arctanh(np.clip(2.0 * best_nodes - 1.0, -0.999, 0.999))
            init_log_w = np.log(best_weights)
            x0_final = np.concatenate([init_node_params, init_log_w])
            
            result_final = minimize(_fast_objective, x0_final, method='Nelder-Mead',
                                   options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if result_final.fun < best_score:
                best_score = result_final.fun
                node_params = result_final.x[:8]
                log_w = result_final.x[8:]
                best_nodes = 0.5 + 0.5 * np.tanh(node_params)
                best_weights = np.exp(log_w)
                best_weights = best_weights / np.sum(best_weights)
        except Exception:
            pass
    
    # Ensure constraints
    best_nodes = np.sort(best_nodes)
    for i in range(7):
        if best_nodes[i+1] <= best_nodes[i]:
            best_nodes[i+1] = best_nodes[i] + 1e-10
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    best_weights = np.maximum(best_weights, 1e-10)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
