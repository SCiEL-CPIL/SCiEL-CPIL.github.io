import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Pre-compute high-precision reference quadrature nodes and weights
    # This replaces scipy.integrate.quad for speed and determinism
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(200)
    ref_nodes_01 = 0.5 * (ref_nodes + 1.0)
    ref_weights_01 = 0.5 * ref_weights
    ref_weights_01 = ref_weights_01 / np.sum(ref_weights_01)

    # Define E(a) for CPL dark energy in flat universe
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(3*w1*(1-a_safe)/a_safe))
    
    def _integrand(z_val, w0, w1):
        a = 1.0 / (1.0 + z_val)
        return (1.0 + z_val) / _E(a, w0, w1)
    
    def _true_distance(z_max, w0, w1):
        # Use pre-computed high-order Gauss-Legendre for "true" value
        z_ref = z_max * ref_nodes_01
        vals = np.array([_integrand(z, w0, w1) for z in z_ref])
        return z_max * np.sum(ref_weights_01 * vals)
    
    def _quad_distance(z_max, nodes, weights, w0, w1):
        z_nodes = z_max * nodes
        vals = np.array([_integrand(z, w0, w1) for z in z_nodes])
        return z_max * np.sum(weights * vals)
    
    def _relative_error(z_max, nodes, weights, w0, w1):
        true = _true_distance(z_max, w0, w1)
        approx = _quad_distance(z_max, nodes, weights, w0, w1)
        return abs(approx - true) / true * 100.0
    
    # Comprehensive test set: 14 cosmologies, 9 redshifts
    test_cosmos = [
        (-1.0, 0.0), (-1.0, 0.5), (-1.0, -0.5), (-0.9, 0.3),
        (-0.95, -0.3), (-1.2, 0.0), (-0.8, 0.5), (-1.0, 1.0),
        (-1.1, 0.4), (-0.85, -0.4), (-0.7, 0.6), (-1.3, -0.2),
        (-0.95, 0.7), (-1.05, -0.6)
    ]
    test_zs = [0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    def _worst_error(nodes, weights):
        max_err = 0.0
        for (w0, w1) in test_cosmos:
            for z in test_zs:
                err = _relative_error(z, nodes, weights, w0, w1)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Initial guess: Gauss-Legendre 8 nodes
    t, w = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w
    weights_init = weights_init / np.sum(weights_init)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights)
    
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        weights = weights / np.sum(weights)
        return _worst_error(nodes, weights)
    
    # Optimization loop with restarts
    np.random.seed(42)
    
    # Start with the initial GL guess
    x0 = np.concatenate([best_nodes, best_weights])
    
    # Perform multiple restarts to escape local minima
    for i in range(50):
        if time.time() > deadline - 2:
            break
        
        # If not the first iteration, perturb the best solution found so far
        if i > 0:
            perturbation = 0.1 / (i + 1)
            nodes_try = best_nodes + np.random.uniform(-perturbation, perturbation, 8)
            nodes_try = np.sort(nodes_try)
            nodes_try = np.clip(nodes_try, 0.01, 0.99)
            for j in range(1, 8):
                if nodes_try[j] <= nodes_try[j-1]:
                    nodes_try[j] = nodes_try[j-1] + 1e-6
            
            weights_try = best_weights + np.random.uniform(-0.05, 0.05, 8)
            weights_try = np.maximum(weights_try, 1e-4)
            weights_try = weights_try / np.sum(weights_try)
            
            x0 = np.concatenate([nodes_try, weights_try])
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if result.fun < best_err:
                best_nodes = result.x[:8]
                best_weights = result.x[8:] / np.sum(result.x[8:])
                best_err = result.fun
        except Exception:
            pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-8
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
