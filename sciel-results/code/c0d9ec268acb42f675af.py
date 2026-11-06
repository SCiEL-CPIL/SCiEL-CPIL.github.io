import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Define E(a) for CPL dark energy in flat universe
    # w(z) = w0 + w1*z, a = 1/(1+z)
    # E^2(a) = Omega_m/a^3 + Omega_de * a^(-3*(1+w0)) * exp(3*w1*(1-a)/a)
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        # Handle a=0 edge case (shouldn't happen with nodes in (0,1))
        a_safe = np.maximum(a, 1e-10)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(3*w1*(1-a_safe)/a_safe))
    
    def _integrand(z_node, z_max, w0, w1):
        """Integrand value at quadrature node for given max redshift."""
        z_val = z_node * z_max
        a = 1.0 / (1.0 + z_val)
        return (1.0 + z_val) / _E(a, w0, w1)
    
    def _true_distance(z_max, w0, w1):
        """True luminosity distance via adaptive quadrature."""
        def f(z):
            a = 1.0 / (1.0 + z)
            return (1.0 + z) / _E(a, w0, w1)
        val, _ = quad(f, 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val
    
    def _quad_distance(z_max, nodes, weights, w0, w1):
        """Quadrature approximation of luminosity distance."""
        z_nodes = z_max * nodes
        a_nodes = 1.0 / (1.0 + z_nodes)
        vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1)
        return z_max * np.sum(weights * vals)
    
    def _relative_error(z_max, nodes, weights, w0, w1):
        """Compute relative error in percent."""
        true = _true_distance(z_max, w0, w1)
        approx = _quad_distance(z_max, nodes, weights, w0, w1)
        return abs(approx - true) / true * 100.0
    
    # Test cosmologies: diverse CPL models
    test_cosmos = [
        (-1.0, 0.0),      # LambdaCDM
        (-1.0, 0.5),      # CPL w1>0
        (-1.0, -0.5),     # CPL w1<0
        (-0.9, 0.3),      # Quintessence-like
        (-0.95, -0.3),    # Another variant
        (-1.2, 0.0),      # Phantom
        (-0.8, 0.5),      # Strong evolution
        (-1.0, 1.0),      # Strong CPL
        (-1.1, 0.4),      # Additional
        (-0.85, -0.4),    # Additional
    ]
    
    # Redshifts to test
    test_zs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    def _worst_error(nodes, weights):
        """Compute worst relative error over all test cosmologies and redshifts."""
        max_err = 0.0
        for (w0, w1) in test_cosmos:
            for z in test_zs:
                err = _relative_error(z, nodes, weights, w0, w1)
                if err > max_err:
                    max_err = err
        return max_err
    
    # Start with Gauss-Legendre as initial guess
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
        
        # Check node constraints: strictly increasing in (0,1)
        if np.any(nodes[1:] <= nodes[:-1]):
            return 1e10
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e10
        if np.any(weights <= 0):
            return 1e10
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return _worst_error(nodes, weights)
    
    # Local optimization from GL starting point
    x0 = np.concatenate([nodes_init, weights_init])
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        if time.time() < deadline and result.fun < best_err:
            best_nodes = result.x[:8]
            best_weights = result.x[8:] / np.sum(result.x[8:])
            best_err = result.fun
    except Exception:
        pass
    
    # Try several random restarts with perturbed GL nodes
    np.random.seed(42)
    for i in range(15):
        if time.time() > deadline - 5:
            break
        
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
        
        err = _worst_error(nodes_try, weights_try)
        if err < best_err:
            best_nodes = nodes_try.copy()
            best_weights = weights_try.copy()
            best_err = err
            
            # Refine from this better point
            if time.time() < deadline - 3:
                x0 = np.concatenate([best_nodes, best_weights])
                try:
                    result = minimize(_objective, x0, method='Nelder-Mead', 
                                     options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                    if time.time() < deadline and result.fun < best_err:
                        best_nodes = result.x[:8]
                        best_weights = result.x[8:] / np.sum(result.x[8:])
                        best_err = result.fun
                except Exception:
                    pass
    
    # Final refinement from best point
    if time.time() < deadline - 2:
        x0 = np.concatenate([best_nodes, best_weights])
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if time.time() < deadline and result.fun < best_err:
                best_nodes = result.x[:8]
                best_weights = result.x[8:] / np.sum(result.x[8:])
                best_err = result.fun
        except Exception:
            pass
    
    # Final constraint enforcement
    best_nodes = np.sort(best_nodes)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    # Ensure strict increase
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-8
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
