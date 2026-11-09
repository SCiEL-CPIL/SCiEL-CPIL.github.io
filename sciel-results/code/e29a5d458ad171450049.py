"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, leggauss
from scipy.integrate import quad
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit, leave 40s margin
    
    # 1. Define representative cosmologies (small grid for speed)
    omega_m_vals = [0.2, 0.3, 0.4]
    w0_vals = [-1.2, -1.0, -0.8]
    w1_vals = [0.0] # Fix w1 to 0 for speed? Or include one non-zero?
    # Let's use 5 specific models:
    cosmologies = [
        (0.27, -1.0, 0.0), # LambdaCDM
        (0.30, -1.2, 0.5), # Phantom-like
        (0.20, -0.8, -0.5), # Quintessence-like
        (0.40, -1.0, 0.0), # High Omega_m
        (0.25, -1.1, 0.2)  # Intermediate
    ]
    
    redshifts = [0.1, 1.0, 5.0]
    
    # 2. Define E(a) and integrand
    def E_of_a(a, om, w0, w1):
        w = w0 + w1 * (1 - a)
        term_dark = (1 - om) * np.power(a, -3 * (1 + w))
        term_matter = om * np.power(a, -3)
        return np.sqrt(term_matter + term_dark)
        
    def integrand(u, z, om, w0, w1):
        a = 1.0 / (1.0 + z * u)
        E = E_of_a(a, om, w0, w1)
        return (1.0 + z * u) / E
        
    def exact_distance(z, om, w0, w1):
        val, _ = quad(integrand, 0, 1, args=(z, om, w0, w1), limit=50)
        return val
        
    def quadrature_distance(nodes, weights, z, om, w0, w1):
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = E_of_a(a_vals, om, w0, w1)
        vals = (1.0 + z * nodes) / E_vals
        return np.dot(weights, vals)
        
    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        nodes = params[:8]
        weights = params[8:]
        
        # Soft constraints: penalize invalid nodes/weights
        penalty = 0.0
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            penalty += 10.0
        if not np.all(np.diff(nodes) > 1e-6):
            penalty += 10.0
        if np.any(weights < 0):
            penalty += 10.0
        if abs(np.sum(weights) - 1.0) > 1e-4:
            penalty += 10.0 * abs(np.sum(weights) - 1.0)
            
        if penalty > 0:
            return 1e10 + penalty
            
        max_err = 0.0
        for (om, w0, w1) in cosmologies:
            for z in redshifts:
                exact = exact_distance(z, om, w0, w1)
                approx = quadrature_distance(nodes, weights, z, om, w0, w1)
                rel_err = abs(approx - exact) / exact
                if rel_err > max_err:
                    max_err = rel_err
        return max_err

    # 3. Initial guess: Gauss-Legendre
    x, w = leggauss(8)
    init_nodes = (x + 1.0) / 2.0
    init_weights = w / 2.0
    init_weights = init_weights / np.sum(init_weights)
    
    x0 = np.concatenate((init_nodes, init_weights))
    best_score = objective(x0)
    best_params = x0
    
    # 4. Optimization: Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
        
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
    except Exception:
        pass
        
    # 5. Final validation and return
    final_nodes = best_params[:8]
    final_weights = best_params[8:]
    
    # Enforce constraints strictly
    if np.any(final_nodes <= 0) or np.any(final_nodes >= 1) or not np.all(np.diff(final_nodes) > 0) or np.any(final_weights < 0):
        # Fallback to Gauss-Legendre
        final_nodes = init_nodes
        final_weights = init_weights
    else:
        # Normalize weights
        final_weights = final_weights / np.sum(final_weights)
        
    return {"nodes": final_nodes, "weights": final_weights}
# EVOLVE-BLOCK-END
