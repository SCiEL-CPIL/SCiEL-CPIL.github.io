import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def _E(a, w0, w1, Omega_m):
    """E(a) = H(a)/H0 for CPL dark energy with flat spatial geometry."""
    Omega_de = 1.0 - Omega_m
    # For CPL: w(z) = w0 + w1*z
    # In terms of a = 1/(1+z): rho_de(a) = rho_de0 * a^{-3(1+w0)} * exp(3*w1*(1-a)/a)
    # E^2(a) = Omega_m * a^{-3} + Omega_de * a^{-3(1+w0)} * exp(3*w1*(1-a)/a)
    return np.sqrt(Omega_m * a**(-3) + Omega_de * a**(-3*(1+w0)) * np.exp(3*w1*(1-a)/a))

def _integrand(z, w0, w1, Omega_m):
    """Integrand for luminosity distance: (1+z)/E(1/(1+z))"""
    a = 1.0 / (1.0 + z)
    return (1.0 + z) / _E(a, w0, w1, Omega_m)

def _true_distance(z, w0, w1, Omega_m, n_points=10000):
    """Compute true luminosity distance via adaptive quadrature."""
    val, _ = quad(lambda zz: _integrand(zz, w0, w1, Omega_m), 0, z, limit=200)
    return val

def _quad_distance(z, nodes, weights, w0, w1, Omega_m):
    """Compute quadrature approximation of luminosity distance."""
    z_nodes = z * nodes
    a_nodes = 1.0 / (1.0 + z_nodes)
    vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1, Omega_m)
    return z * np.sum(weights * vals)

def _relative_error(z, nodes, weights, w0, w1, Omega_m):
    """Compute relative error in percent."""
    true = _true_distance(z, w0, w1, Omega_m)
    approx = _quad_distance(z, nodes, weights, w0, w1, Omega_m)
    return abs(approx - true) / true * 100.0

def _worst_error(nodes, weights, test_cosmos, test_zs):
    """Compute worst relative error over all test cosmologies and redshifts."""
    max_err = 0.0
    for (w0, w1, Om) in test_cosmos:
        for z in test_zs:
            err = _relative_error(z, nodes, weights, w0, w1, Om)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin before 90s kill
    
    # Test cosmologies: (w0, w1, Omega_m) triples
    # Include standard LambdaCDM, phantom, quintessence, varying Omega_m
    test_cosmos = [
        (-1.0, 0.0, 0.3),      # Standard LambdaCDM
        (-1.0, 0.5, 0.3),      # CPL with w1>0
        (-1.0, -0.5, 0.3),     # CPL with w1<0
        (-0.9, 0.3, 0.3),      # Quintessence-like
        (-0.95, -0.3, 0.3),    # Another variant
        (-1.2, 0.0, 0.3),      # Phantom
        (-0.8, 0.5, 0.3),      # Strong evolution
        (-1.0, 1.0, 0.3),      # Strong CPL
        (-1.0, 0.0, 0.27),     # Varying Omega_m
        (-1.0, 0.3, 0.35),     # Varying Omega_m
        (-0.9, 0.0, 0.25),     # Combined variation
        (-1.1, -0.2, 0.32),    # Combined variation
    ]
    
    test_zs = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    
    # Start with Gauss-Legendre as initial guess
    t, w = np.polynomial.legendre.leggauss(8)
    nodes_init = 0.5 * (t + 1.0)
    weights_init = 0.5 * w
    weights_init = weights_init / np.sum(weights_init)
    
    best_nodes = nodes_init.copy()
    best_weights = weights_init.copy()
    best_err = _worst_error(best_nodes, best_weights, test_cosmos, test_zs)
    
    # Objective function with constraint handling
    def _objective(params):
        if time.time() > deadline:
            return 1e10
        nodes = params[:8]
        weights = params[8:]
        
        # Check constraints
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return _worst_error(nodes, weights, test_cosmos, test_zs)
    
    # Initial params
    x0 = np.concatenate([best_nodes, best_weights])
    
    # Try Nelder-Mead with multiple restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            x0 = np.concatenate([best_nodes, best_weights])
        else:
            # Random perturbation of GL nodes
            nodes_try = nodes_init + np.random.uniform(-0.08, 0.08, 8)
            nodes_try = np.sort(nodes_try)
            nodes_try = np.clip(nodes_try, 0.005, 0.995)
            # Ensure strict increase
            for j in range(1, 8):
                if nodes_try[j] <= nodes_try[j-1]:
                    nodes_try[j] = nodes_try[j-1] + 1e-6
            
            weights_try = weights_init + np.random.uniform(-0.03, 0.03, 8)
            weights_try = np.maximum(weights_try, 1e-6)
            weights_try = weights_try / np.sum(weights_try)
            x0 = np.concatenate([nodes_try, weights_try])
        
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-10})
            if time.time() < deadline and result.fun < best_err:
                best_nodes = result.x[:8]
                best_weights = result.x[8:] / np.sum(result.x[8:])
                best_err = result.fun
        except Exception:
            pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    # Ensure strict increase
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-10
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
