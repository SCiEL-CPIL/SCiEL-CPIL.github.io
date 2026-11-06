import numpy as np
import time
from scipy.optimize import differential_evolution
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # Define E(a) for CPL dark energy in flat universe
    def _E(a, w0, w1, Omega_m=0.3):
        Omega_de = 1.0 - Omega_m
        a_safe = np.maximum(a, 1e-10)
        # E^2(a) = Omega_m/a^3 + Omega_de * a^(-3*(1+w0)) * exp(3*w1*(1-a)/a)
        return np.sqrt(Omega_m * a_safe**(-3) + Omega_de * a_safe**(-3*(1+w0)) * np.exp(3*w1*(1-a_safe)/a_safe))

    def _integrand(z, w0, w1):
        a = 1.0 / (1.0 + z)
        return (1.0 + z) / _E(a, w0, w1)

    def _true_distance(z_max, w0, w1):
        val, _ = quad(lambda z: _integrand(z, w0, w1), 0, z_max, limit=200, epsabs=1e-12, epsrel=1e-12)
        return val

    # Test cosmologies: diverse CPL models
    # (w0, w1)
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
        (-0.7, 0.8),      # Very strong evolution
        (-1.3, -0.2),     # Phantom with slight evolution
        (-1.0, 0.2),      # Moderate CPL
        (-1.0, -0.2),     # Moderate CPL
        (-0.5, 0.0),      # Strong Quintessence
    ]

    # Redshifts to test
    test_zs = [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    # Pre-compute true distances
    true_dists = {}
    for (w0, w1) in test_cosmos:
        for z in test_zs:
            key = (w0, w1, z)
            true_dists[key] = _true_distance(z, w0, w1)
    
    # Cache true distances in a list for faster access in objective
    # Index: i * len(test_zs) + j
    true_vals = []
    for (w0, w1) in test_cosmos:
        for z in test_zs:
            true_vals.append(true_dists[(w0, w1, z)])
    true_vals = np.array(true_vals)
    n_cosmos = len(test_cosmos)
    n_zs = len(test_zs)
    total_cases = n_cosmos * n_zs

    def _objective(params):
        if time.time() > deadline:
            return 1e10
            
        # Unpack parameters
        u_nodes = params[:8]
        v_weights = params[8:16]
        
        # Transform nodes to strictly increasing in (0,1)
        # t_k = (sum_{j=1}^k exp(u_j)) / (sum_{j=1}^8 exp(u_j))
        exp_u = np.exp(u_nodes)
        cumsum_u = np.cumsum(exp_u)
        total_u = cumsum_u[-1]
        nodes = cumsum_u / total_u
        
        # Transform weights to positive summing to 1
        exp_v = np.exp(v_weights)
        weights = exp_v / np.sum(exp_v)
        
        max_err = 0.0
        idx = 0
        for i, (w0, w1) in enumerate(test_cosmos):
            for j, z in enumerate(test_zs):
                # Compute quadrature approximation
                z_nodes = z * nodes
                a_nodes = 1.0 / (1.0 + z_nodes)
                vals = (1.0 + z_nodes) / _E(a_nodes, w0, w1)
                approx = z * np.sum(weights * vals)
                
                true_val = true_vals[idx]
                if true_val == 0:
                    err = abs(approx)
                else:
                    err = abs(approx - true_val) / true_val * 100.0
                
                if err > max_err:
                    max_err = err
                idx += 1
        
        return max_err

    # Initial bounds for DE
    # u_nodes: [-5, 5]
    # v_weights: [-5, 5]
    bounds = [(-5, 5)] * 16

    # Run Differential Evolution
    # maxiter: number of generations
    # popsize: size of population
    # tol: relative tolerance
    # atol: absolute tolerance
    # polish: use L-BFGS-B for final polish
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            atol=1e-8,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.success and result.fun < 1e10:
            best_params = result.x
        else:
            # If DE failed or timed out, use best from DE if available, else fallback
            if hasattr(result, 'x') and result.x is not None:
                best_params = result.x
            else:
                # Fallback to Gauss-Legendre
                t, w = np.polynomial.legendre.leggauss(8)
                nodes_init = 0.5 * (t + 1.0)
                weights_init = 0.5 * w
                weights_init = weights_init / np.sum(weights_init)
                # Invert transformation? No, just return GL directly if DE fails completely
                return {"nodes": nodes_init, "weights": weights_init}
                
    except Exception as e:
        # Fallback to Gauss-Legendre
        t, w = np.polynomial.legendre.leggauss(8)
        nodes_init = 0.5 * (t + 1.0)
        weights_init = 0.5 * w
        weights_init = weights_init / np.sum(weights_init)
        return {"nodes": nodes_init, "weights": weights_init}

    # Extract best nodes and weights
    u_nodes = best_params[:8]
    v_weights = best_params[8:16]
    
    exp_u = np.exp(u_nodes)
    cumsum_u = np.cumsum(exp_u)
    total_u = cumsum_u[-1]
    best_nodes = cumsum_u / total_u
    
    exp_v = np.exp(v_weights)
    best_weights = exp_v / np.sum(exp_v)
    
    # Final constraint check and adjustment
    # Ensure strictly increasing
    for j in range(1, 8):
        if best_nodes[j] <= best_nodes[j-1]:
            best_nodes[j] = best_nodes[j-1] + 1e-8
    # Ensure in (0,1)
    best_nodes = np.clip(best_nodes, 1e-8, 1.0 - 1e-8)
    # Ensure weights positive and sum to 1
    best_weights = np.maximum(best_weights, 1e-12)
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
