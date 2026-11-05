import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # 1. Define test cosmologies
    # Each model is a dict with Omega_m, w0, wa
    # E(a) = sqrt(Omega_m a^-3 + (1-Omega_m) * a^(-3*(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0)))
    # Note: For w0=-1, wa=0, this reduces to sqrt(Omega_m a^-3 + (1-Omega_m))
    
    def E(a, Om, w0, wa):
        # Safe implementation of E(a) for CPL parametrization
        if np.abs(w0 + 1.0) < 1e-10:
            if np.abs(wa) < 1e-10:
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om))
            else:
                # Use limit: for w0=-1, wa!=0, the dark energy term is
                # (1-Om) * a^(-3(1+wa)) * exp(3*wa*(1-a)/0)
                # This is problematic. Use a small epsilon for w0+1
                w0_eff = -1.0 + 1e-10
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om) * a**(-3.0 * (1.0 + w0_eff + wa)) * np.exp(3.0 * wa * (1.0 - a) / (1.0 + w0_eff)))
        else:
            return np.sqrt(Om * a**(-3.0) + (1.0 - Om) * a**(-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (1.0 - a) / (1.0 + w0)))

    # Test Cosmologies - more diverse to capture worst-case
    models = [
        {'Om': 0.3, 'w0': -1.0, 'wa': 0.0},  # LCDM
        {'Om': 0.3, 'w0': -0.9, 'wa': 0.1},
        {'Om': 0.2, 'w0': -1.1, 'wa': 0.2},
        {'Om': 0.4, 'w0': -0.8, 'wa': -0.2},
        {'Om': 0.25, 'w0': -1.0, 'wa': 0.0},
        {'Om': 0.35, 'w0': -0.95, 'wa': 0.05},
        {'Om': 0.15, 'w0': -1.05, 'wa': 0.15},
        {'Om': 0.45, 'w0': -0.85, 'wa': -0.15},
        {'Om': 0.3, 'w0': -1.2, 'wa': 0.3},
        {'Om': 0.3, 'w0': -0.7, 'wa': -0.3},
    ]
    
    redshifts = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Pre-compute true integrals
    # Integral: int_0^1 1/E(1/(1+z*xi)) dxi
    true_integrals = []
    for m in models:
        Om = m['Om']
        w0 = m['w0']
        wa = m['wa']
        for z in redshifts:
            def integrand(xi):
                a = 1.0 / (1.0 + z * xi)
                # Handle w0=-1, wa=0 case
                if np.abs(w0 + 1.0) < 1e-10 and np.abs(wa) < 1e-10:
                    e_val = np.sqrt(Om * a**(-3.0) + (1.0 - Om))
                else:
                    e_val = E(a, Om, w0, wa)
                return 1.0 / e_val
            
            # quad returns (value, error)
            val, err = quad(integrand, 0.0, 1.0, limit=200)
            true_integrals.append(val)
    
    true_integrals = np.array(true_integrals)
    
    # 2. Define objective function
    def objective(params):
        # params: 16 values (8 nodes, 8 weights)
        nodes = params[:8]
        weights = params[8:]
        
        # Penalty for constraints
        penalty = 0.0
        
        # Nodes must be in [0, 1] and strictly increasing
        for i in range(8):
            if nodes[i] < 0 or nodes[i] > 1:
                penalty += 100.0 * (abs(nodes[i]) + abs(1.0 - nodes[i]))
        for i in range(7):
            if nodes[i] >= nodes[i+1]:
                penalty += 100.0 * (nodes[i] - nodes[i+1])
                
        # Weights must be positive
        for i in range(8):
            if weights[i] <= 0:
                penalty += 100.0 * abs(weights[i])
                
        # Weights sum to 1
        w_sum = np.sum(weights)
        penalty += 10.0 * abs(w_sum - 1.0)
        
        if penalty > 1.0:
            return penalty
            
        # Compute quadrature sums
        errors = []
        for idx, m in enumerate(models):
            Om = m['Om']
            w0 = m['w0']
            wa = m['wa']
            for j, z in enumerate(redshifts):
                true_val = true_integrals[idx * len(redshifts) + j]
                
                # Compute sum
                s = 0.0
                for i in range(8):
                    a = 1.0 / (1.0 + z * nodes[i])
                    e_val = E(a, Om, w0, wa)
                    s += weights[i] / e_val
                    
                rel_err = abs(s - true_val) / true_val
                errors.append(rel_err)
                
        errors = np.array(errors)
        # Soft max
        beta = 100.0
        max_err = np.log(np.sum(np.exp(beta * errors)) / len(errors)) / beta
        
        return max_err + penalty

    # 3. Optimization
    # Initial guess: Gauss-Legendre
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    x0 = np.concatenate([nodes_gl, weights_gl])
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Use L-BFGS-B with bounds
    bounds = [(0.0, 1.0)] * 16
    
    def callback(xk):
        nonlocal best_x, best_val
        val = objective(xk)
        if val < best_val:
            best_x = xk.copy()
            best_val = val
        return time.time() > deadline
    
    # Run multiple restarts with different initial guesses
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
            
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb from best solution
            x_start = best_x + np.random.randn(16) * 0.05
            # Ensure bounds
            x_start = np.clip(x_start, 0.0, 1.0)
            # Ensure nodes are increasing
            for i in range(1, 8):
                if x_start[i] <= x_start[i-1]:
                    x_start[i] = x_start[i-1] + 0.01
            # Ensure weights are positive
            x_start[8:] = np.clip(x_start[8:], 0.01, 0.9)
            x_start[8:] = x_start[8:] / np.sum(x_start[8:])
        
        try:
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, callback=callback, options={'maxiter': 500, 'ftol': 1e-10})
        except Exception as e:
            pass
    
    # Ensure final result is valid
    nodes = best_x[:8].copy()
    weights = best_x[8:].copy()
    
    # Final polish/normalization
    # Sort nodes if needed and reassign weights
    if not np.all(np.diff(nodes) > 0):
        sort_idx = np.argsort(nodes)
        nodes = nodes[sort_idx]
        weights = weights[sort_idx]
    
    # Check validity
    if not np.all(np.diff(nodes) > 0) or not np.all(weights > 0) or abs(np.sum(weights) - 1.0) > 1e-12:
        # Fallback to Gauss-Legendre
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    else:
        # Normalize weights to sum exactly to 1
        weights = weights / np.sum(weights)
        
    return {"nodes": nodes, "weights": weights}
