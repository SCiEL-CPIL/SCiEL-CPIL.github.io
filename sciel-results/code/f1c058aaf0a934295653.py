import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
    # 1. Define test cosmologies
    # Each model is a dict with Omega_m, w0, wa
    # E(a) = sqrt(Omega_m a^-3 + (1-Omega_m) * a^(-3*(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0)))
    # Note: For w0=-1, wa=0, this reduces to sqrt(Omega_m a^-3 + (1-Omega_m))
    
    def E(a, Om, w0, wa):
        # Safe implementation of E(a) for CPL parametrization
        # Use a stable formulation: for w0 close to -1, use the limiting form
        w0_plus_1 = w0 + 1.0
        if abs(w0_plus_1) < 1e-8:
            # Use the limit: a^(-3(1+w0+wa)) * exp(3*wa*(1-a)/(1+w0)) -> (1-a)^(3*wa) as w0->-1
            # Actually the exact limit: a^{-3(1+w0+wa)} * exp(3*wa*(1-a)/(1+w0))
            # = exp(-3(1+w0+wa)*ln(a) + 3*wa*(1-a)/(1+w0))
            # As w0->-1, let eps = 1+w0, then exponent = -3(eps+wa)*ln(a) + 3*wa*(1-a)/eps
            # = -3*wa*ln(a) - 3*eps*ln(a) + 3*wa*(1-a)/eps
            # For small eps: -3*wa*ln(a) + 3*wa*(1-a)/eps - 3*eps*ln(a)
            # The 3*wa*(1-a)/eps term diverges unless wa=0.
            # For wa=0: limit is just 1 (the dark energy term becomes constant (1-Om))
            # For wa!=0 and w0=-1 exactly, the CPL form is singular. We use a small offset.
            if abs(wa) < 1e-10:
                return np.sqrt(Om * a**(-3.0) + (1.0 - Om))
            else:
                # Use a small offset to avoid singularity
                eps = 1e-8
                w0_eff = -1.0 + eps
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
    
    redshifts = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    
    # Pre-compute true integrals
    # Integral: int_0^1 1/E(1/(1+z*xi)) dxi
    true_integrals = []
    for m in models:
        Om = m['Om']
        w0 = m['w0']
        wa = m['wa']
        for z in redshifts:
            def integrand(xi, Om=Om, w0=w0, wa=wa, z=z):
                a = 1.0 / (1.0 + z * xi)
                e_val = E(a, Om, w0, wa)
                return 1.0 / e_val
            
            # quad returns (value, error)
            val, err = quad(integrand, 0.0, 1.0, limit=200)
            true_integrals.append(val)
    
    true_integrals = np.array(true_integrals)
    n_cases = len(models) * len(redshifts)
    
    # 2. Define objective function - vectorized for speed
    # Pre-compute model parameters as arrays
    Om_arr = np.array([m['Om'] for m in models])
    w0_arr = np.array([m['w0'] for m in models])
    wa_arr = np.array([m['wa'] for m in models])
    z_arr = np.array(redshifts)
    
    # Pre-compute E values at all (model, redshift, node) combinations
    # E[mi, zj, ni] = E(a=1/(1+z_j*node_i), Om_mi, w0_mi, wa_mi)
    # This is 10 * 6 * 8 = 480 values
    
    def compute_E_table(nodes):
        """Compute E values for all models, redshifts, and given nodes."""
        # nodes: shape (8,)
        # Returns: shape (n_models, n_z, 8)
        E_table = np.zeros((len(models), len(redshifts), 8))
        for mi in range(len(models)):
            Om = Om_arr[mi]
            w0 = w0_arr[mi]
            wa = wa_arr[mi]
            for zj in range(len(redshifts)):
                z = z_arr[zj]
                for ni in range(8):
                    a = 1.0 / (1.0 + z * nodes[ni])
                    E_table[mi, zj, ni] = E(a, Om, w0, wa)
        return E_table
    
    def objective(params):
        # params: 16 values (8 nodes, 8 weights)
        nodes = params[:8]
        weights = params[8:]
        
        # Penalty for constraints
        penalty = 0.0
        
        # Nodes must be in [0, 1] and strictly increasing
        if np.any(nodes < 0) or np.any(nodes > 1):
            penalty += 100.0 * np.sum(np.maximum(0, -nodes) + np.maximum(0, nodes - 1))
        diffs = np.diff(nodes)
        if np.any(diffs <= 0):
            penalty += 100.0 * np.sum(np.maximum(0, -diffs))
                
        # Weights must be positive
        if np.any(weights <= 0):
            penalty += 100.0 * np.sum(np.maximum(0, -weights))
                
        # Weights sum to 1
        w_sum = np.sum(weights)
        penalty += 10.0 * abs(w_sum - 1.0)
        
        if penalty > 1.0:
            return penalty
            
        # Compute quadrature sums using pre-computed E table
        E_table = compute_E_table(nodes)  # shape (n_models, n_z, 8)
        
        # Compute sum for each case: sum_i weights[i] / E[mi, zj, i]
        # inv_E: shape (n_models, n_z, 8)
        inv_E = 1.0 / E_table
        # quadrature: shape (n_models, n_z) = sum over i of weights[i] * inv_E[mi, zj, i]
        quadrature = np.sum(inv_E * weights[None, None, :], axis=2)
        
        # Relative errors
        true_vals = true_integrals.reshape(len(models), len(redshifts))
        rel_errors = np.abs(quadrature - true_vals) / true_vals
        
        # Soft max with moderate beta
        beta = 50.0
        max_err = np.log(np.sum(np.exp(beta * rel_errors)) / rel_errors.size) / beta
        
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
    
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb from best solution with varying scale
            scale = 0.05 * (1.0 + 0.5 * restart)
            x_start = best_x + np.random.randn(16) * scale
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
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, callback=callback, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        except Exception as e:
            pass
    
    # Final high-precision polish: use Nelder-Mead on a constrained parameterization
    # Use log-weights and cumulative nodes to enforce constraints
    if time.time() < deadline - 5.0:
        # Convert to constrained-free parameters
        # nodes: use cumulative sums of positive values, normalized to [0,1]
        # weights: use log parameters, softmax to ensure positive and sum=1
        
        def objective_constrained(params):
            # params: 15 values (7 node gaps + 8 log weights)
            node_gaps = np.exp(params[:7])  # positive gaps
            log_w = params[7:]
            
            # Normalize nodes to [0,1]
            cum_nodes = np.concatenate([[0.0], np.cumsum(node_gaps)])
            total = cum_nodes[-1]
            if total < 1e-10:
                return 100.0
            nodes_c = cum_nodes / total
            # Ensure strictly in (0,1) with small margin
            nodes_c = np.clip(nodes_c, 1e-6, 1.0 - 1e-6)
            # Re-normalize to ensure sum of gaps = 1 (they should by construction)
            
            # Softmax weights
            log_w_shifted = log_w - np.max(log_w)
            w_exp = np.exp(log_w_shifted)
            weights_c = w_exp / np.sum(w_exp)
            
            x = np.concatenate([nodes_c, weights_c])
            return objective(x)
        
        # Initialize from best solution
        node_gaps_init = np.diff(best_x[:8])
        node_gaps_init = np.maximum(node_gaps_init, 1e-4)
        log_w_init = np.log(np.maximum(best_x[8:], 1e-10))
        x_con_init = np.concatenate([node_gaps_init, log_w_init])
        
        try:
            res2 = minimize(objective_constrained, x_con_init, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14})
            if res2.fun < best_val:
                # Convert back
                node_gaps = np.exp(res2.x[:7])
                log_w = res2.x[7:]
                cum_nodes = np.concatenate([[0.0], np.cumsum(node_gaps)])
                nodes_final = cum_nodes / cum_nodes[-1]
                nodes_final = np.clip(nodes_final, 1e-6, 1.0 - 1e-6)
                log_w_shifted = log_w - np.max(log_w)
                w_exp = np.exp(log_w_shifted)
                weights_final = w_exp / np.sum(w_exp)
                best_x = np.concatenate([nodes_final, weights_final])
                best_val = res2.fun
        except Exception:
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
