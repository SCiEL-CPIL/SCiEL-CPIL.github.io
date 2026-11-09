import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Representative cosmologies: (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),   # Lambda CDM
        (0.3, -1.2, 0.0),   # Phantom
        (0.3, -0.8, 0.0),   # Quintessence
        (0.3, -1.0, 0.5),   # Evolving DE (w1 positive)
        (0.3, -1.0, -0.5),  # Evolving DE (w1 negative)
        (0.2, -1.0, 0.0),   # Low matter density
        (0.4, -1.0, 0.0),   # High matter density
        (0.3, -1.5, 0.3),   # Strongly evolving
        (0.3, -0.5, -0.3),  # Mild quintessence with evolution
        (0.35, -1.1, 0.2),  # Slightly matter-dominated
    ]
    
    # Dense grid of redshifts
    z_test = np.linspace(0.05, 2.0, 20)
    
    def E_func(z, Om, w0, w1):
        OL = 1.0 - Om
        return np.sqrt(Om * (1.0 + z)**3 + OL * (1.0 + z)**(3.0 * (1.0 + w0 + w1 * z)))
    
    def true_integral(z, Om, w0, w1):
        def integrand(u):
            return 1.0 / E_func(z * u, Om, w0, w1)
        res, _ = quad(integrand, 0, 1, limit=100)
        return res

    # Precompute true integrals
    num_cosmo = len(cosmo_params)
    num_z = len(z_test)
    true_vals = np.zeros((num_cosmo, num_z))
    
    for i, (Om, w0, w1) in enumerate(cosmo_params):
        for j, z in enumerate(z_test):
            true_vals[i, j] = true_integral(z, Om, w0, w1)
            
    if time.time() > deadline:
        # Fallback: Gauss-Legendre
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    # Structural Change: Fix nodes to Gauss-Legendre, optimize weights only
    n = 8
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (gl_nodes + 1.0)
    init_weights = 0.5 * gl_weights
    
    def objective(weights):
        # weights: 8 floats
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-8:
            return 1e6

        max_err = 0.0
        for i in range(num_cosmo):
            Om, w0, w1 = cosmo_params[i]
            for j in range(num_z):
                z = z_test[j]
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                integrand_vals = 1.0 / E_vals
                quad_val = np.dot(weights, integrand_vals)
                
                true_val = true_vals[i, j]
                if true_val > 0:
                    err = abs(quad_val - true_val) / true_val
                    if err > max_err:
                        max_err = err
        return max_err

    # Optimize weights only
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0)] * 8
    
    try:
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        best_weights = result.x
    except Exception:
        best_weights = init_weights

    # Final cleanup
    best_weights = best_weights / np.sum(best_weights)
    
    if not np.all(best_weights > 0):
        # Fallback
        best_weights = 0.5 * gl_weights
        best_weights = best_weights / np.sum(best_weights)
        
    if abs(np.sum(best_weights) - 1.0) > 1e-12:
        best_weights = best_weights / np.sum(best_weights)
        
    return {"nodes": nodes, "weights": best_weights}
