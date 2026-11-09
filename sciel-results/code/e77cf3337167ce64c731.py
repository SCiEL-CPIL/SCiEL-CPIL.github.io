"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
TIME_LIMIT = 85  # seconds, leave margin before 90s kill


def _integrand_cpl(t, z, w0, w1, Omega_m=0.3):
    """1/E(z*t) for CPL model."""
    zp = z * t
    E2 = Omega_m * (1 + zp)**3 + (1 - Omega_m) * (1 + zp)**(3 * (1 + w0 + w1 * zp))
    E = np.sqrt(E2)
    return 1.0 / E


def _true_integral_fast(z, w0, w1, Omega_m=0.3):
    """High-precision integral of 1/E(z*t) dt from 0 to 1 using Gauss-Legendre with 200 points."""
    # Use a high-order Gauss-Legendre quadrature as reference
    x, w = np.polynomial.legendre.leggauss(200)
    t = 0.5 * (x + 1)
    w = 0.5 * w
    zp = z * t
    E2 = Omega_m * (1 + zp)**3 + (1 - Omega_m) * (1 + zp)**(3 * (1 + w0 + w1 * zp))
    vals = 1.0 / np.sqrt(E2)
    return np.dot(w, vals)


def _quadrature(nodes, weights, z, w0, w1, Omega_m=0.3):
    """Compute z * sum_i w_i / E(z * t_i)."""
    zt = z * nodes
    E2 = Omega_m * (1 + zt)**3 + (1 - Omega_m) * (1 + zt)**(3 * (1 + w0 + w1 * zt))
    vals = 1.0 / np.sqrt(E2)
    return z * np.dot(weights, vals)


def construct():
    n = 8
    
    # Test cases: various redshifts and CPL parameters
    z_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    w0_values = [-1.5, -1.2, -1.0, -0.8, -0.5]
    w1_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Create test cases
    test_cases = []
    for z in z_values:
        for w0 in w0_values:
            for w1 in w1_values:
                test_cases.append((z, w0, w1))
    
    # Pre-compute true values
    true_vals = np.array([_true_integral_fast(z, w0, w1) for z, w0, w1 in test_cases])
    
    # Initial guess: Gauss-Legendre nodes/weights mapped to [0,1]
    from numpy.polynomial.legendre import leggauss
    nodes0, weights0 = leggauss(n)
    nodes0 = 0.5 * (nodes0 + 1)  # map from [-1,1] to [0,1]
    weights0 = 0.5 * weights0
    
    # Parameterization: nodes via arctanh to ensure in (0,1), weights via log to ensure positive
    def _unpack(params):
        # params: 8 node params, 8 weight params
        np_params = params[:n]
        wp_params = params[n:]
        # Nodes: use sigmoid-like mapping
        raw_nodes = 0.5 + 0.49 * np.tanh(np_params)  # in (0.01, 0.99)
        raw_nodes = np.sort(raw_nodes)
        # Weights: use exp to ensure positive, then normalize
        raw_weights = np.exp(wp_params)
        raw_weights = raw_weights / raw_weights.sum()
        return raw_nodes, raw_weights
    
    def _objective(params):
        if time.time() - _start_time > TIME_LIMIT:
            return 1e10
        nodes, weights = _unpack(params)
        # Check strictly increasing
        if np.any(np.diff(nodes) <= 1e-12):
            return 1e10
        max_err = 0.0
        for i, (z, w0, w1) in enumerate(test_cases):
            zt = z * nodes
            E2 = 0.3 * (1 + zt)**3 + 0.7 * (1 + zt)**(3 * (1 + w0 + w1 * zt))
            vals = 1.0 / np.sqrt(E2)
            quad_val = z * np.dot(weights, vals)
            true_val = true_vals[i]
            if abs(true_val) > 1e-15:
                rel_err = abs(quad_val - true_val) / abs(true_val)
                if rel_err > max_err:
                    max_err = rel_err
                    if max_err > 0.01:  # early exit if error is large
                        break
        return max_err
    
    # Start from Gauss-Legendre
    params0 = np.zeros(2 * n)
    # Invert the unpack for initial
    # For nodes: 0.5 + 0.49*tanh(x) = node => x = arctanh((node-0.5)/0.49)
    for i in range(n):
        ni = nodes0[i]
        if 0.01 < ni < 0.99:
            params0[i] = np.arctanh((ni - 0.5) / 0.49)
        else:
            params0[i] = 0.0
        wi = weights0[i]
        # exp(x) / sum(exp) = wi => x = log(wi * sum_w) approximately
        sw = weights0.sum()
        wi_sw = wi * sw
        if wi_sw > 1e-15:
            params0[n + i] = np.log(wi_sw)
        else:
            params0[n + i] = 0.0
    
    # Optimize using L-BFGS-B
    best_obj = _objective(params0)
    best_params = params0.copy()
    
    # Try multiple local optimizations from different starting points
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() - _start_time > TIME_LIMIT - 10:
            break
        
        if r == 0:
            start_params = best_params.copy()
        else:
            # Add some random perturbation
            start_params = best_params + np.random.normal(0, 0.1, 2*n)
        
        res = minimize(_objective, start_params, method='L-BFGS-B', 
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    
    # Final refinement with tighter tolerance
    if time.time() - _start_time < TIME_LIMIT - 5:
        def _obj_tight(params):
            if time.time() - _start_time > TIME_LIMIT - 2:
                return 1e10
            nodes, weights = _unpack(params)
            if np.any(np.diff(nodes) <= 1e-12):
                return 1e10
            max_err = 0.0
            for i, (z, w0, w1) in enumerate(test_cases):
                zt = z * nodes
                E2 = 0.3 * (1 + zt)**3 + 0.7 * (1 + zt)**(3 * (1 + w0 + w1 * zt))
                vals = 1.0 / np.sqrt(E2)
                quad_val = z * np.dot(weights, vals)
                true_val = true_vals[i]
                if abs(true_val) > 1e-15:
                    rel_err = abs(quad_val - true_val) / abs(true_val)
                    if rel_err > max_err:
                        max_err = rel_err
                        if max_err > 0.005:
                            break
            return max_err
        
        res2 = minimize(_obj_tight, best_params, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_params = res2.x
    
    # Extract final nodes and weights
    nodes, weights = _unpack(best_params)
    nodes = np.sort(nodes)
    weights = weights / weights.sum()
    
    # Ensure strictly increasing with minimum gap
    for i in range(1, n):
        if nodes[i] - nodes[i-1] < 1e-6:
            nodes[i] = nodes[i-1] + 1e-6
    # Re-normalize weights
    weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
