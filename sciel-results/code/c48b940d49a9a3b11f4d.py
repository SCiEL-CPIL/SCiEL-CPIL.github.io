"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _E_cpl(a, w0, wa):
    """E(a) = H(a)/H0 for CPL model in flat universe."""
    return a**(-3.0) * (1.0 + (1.0 - a) * (w0 + wa * (1.0 - a))) ** 1.5

def _integrand(z, w0, wa):
    """(1+z)*z / E(a) where a = 1/(1+z)."""
    a = 1.0 / (1.0 + z)
    return (1.0 + z) * z / _E_cpl(a, w0, wa)

def _true_integral(z, w0, wa, n_points=10000):
    """High-accuracy adaptive-like integral using many points."""
    # Use Gauss-Legendre with many points for high accuracy
    xs, ws = np.polynomial.legendre.leggauss(n_points)
    # Map from [-1,1] to [0, z]
    xs_mapped = 0.5 * z * (xs + 1.0)
    ws_mapped = 0.5 * z * ws
    vals = _integrand(xs_mapped, w0, wa)
    return np.sum(ws_mapped * vals)

def _quadrature_error(nodes, weights, w0, wa, z):
    """Compute relative error of quadrature vs true integral."""
    a_i = 1.0 / (1.0 + z * nodes)
    quad_val = z * np.sum(weights / _E_cpl(a_i, w0, wa))
    true_val = _true_integral(z, w0, wa, 200)
    if abs(true_val) < 1e-15:
        return 0.0
    return abs(quad_val - true_val) / abs(true_val)

def _optimize_quadrature(deadline):
    """Optimize 8-node positive quadrature rule."""
    n = 8
    
    # Start with midpoint rule
    nodes = (np.arange(n) + 0.5) / n
    weights = np.full(n, 1.0 / n)
    
    # Define a diverse set of cosmologies (w0, wa) pairs
    cosmologies = []
    w0_vals = [-0.5, -0.8, -1.0, -1.2, -1.5]
    wa_vals = [-0.3, -0.1, 0.0, 0.1, 0.3]
    for w0 in w0_vals:
        for wa in wa_vals:
            cosmologies.append((w0, wa))
    
    # Redshifts to test
    z_vals = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    best_nodes = nodes.copy()
    best_weights = weights.copy()
    best_max_err = float('inf')
    
    # Use a gradient-free optimization approach with adaptive step sizes
    # Represent nodes as log-spaced or use simplex-like perturbations
    
    # Try multiple random restarts with local optimization
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    max_iter_per_restart = 200
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize with perturbed midpoint
        cur_nodes = nodes.copy()
        cur_weights = weights.copy()
        
        if restart > 0:
            # Perturb nodes
            cur_nodes = nodes + rng.normal(0, 0.05, n)
            cur_nodes = np.sort(cur_nodes)
            # Ensure strictly increasing and in [0,1]
            cur_nodes[0] = max(cur_nodes[0], 1e-6)
            cur_nodes[-1] = min(cur_nodes[-1], 1 - 1e-6)
            for i in range(1, n):
                cur_nodes[i] = max(cur_nodes[i], cur_nodes[i-1] + 1e-6)
            
            # Perturb weights
            cur_weights = weights + rng.normal(0, 0.02, n)
            cur_weights = np.abs(cur_weights)
            cur_weights /= cur_weights.sum()
        
        cur_max_err = float('inf')
        cur_max_err = _evaluate_max_error(cur_nodes, cur_weights, cosmologies, z_vals)
        
        if cur_max_err < best_max_err:
            best_max_err = cur_max_err
            best_nodes = cur_nodes.copy()
            best_weights = cur_weights.copy()
        
        # Local optimization via coordinate descent
        step_size = 0.01
        for iteration in range(max_iter_per_restart):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            
            # Try perturbing each node
            for i in range(n):
                for direction in [-1, 1]:
                    new_nodes = cur_nodes.copy()
                    new_nodes[i] += direction * step_size
                    
                    # Check validity
                    if new_nodes[i] < 1e-6 or new_nodes[i] > 1 - 1e-6:
                        continue
                    if i > 0 and new_nodes[i] <= new_nodes[i-1] + 1e-6:
                        continue
                    if i < n-1 and new_nodes[i] >= new_nodes[i+1] - 1e-6:
                        continue
                    
                    # Recompute weights to maintain sum=1 (keep weights fixed for now)
                    err = _evaluate_max_error(new_nodes, cur_weights, cosmologies, z_vals)
                    if err < cur_max_err:
                        cur_nodes = new_nodes
                        cur_max_err = err
                        improved = True
                        if cur_max_err < best_max_err:
                            best_max_err = cur_max_err
                            best_nodes = cur_nodes.copy()
                            best_weights = cur_weights.copy()
                        break
            
            # Try perturbing each weight
            for i in range(n):
                for direction in [-1, 1]:
                    new_weights = cur_weights.copy()
                    new_weights[i] += direction * step_size
                    new_weights[i] = max(new_weights[i], 1e-12)
                    new_weights /= new_weights.sum()
                    
                    err = _evaluate_max_error(cur_nodes, new_weights, cosmologies, z_vals)
                    if err < cur_max_err:
                        cur_weights = new_weights
                        cur_max_err = err
                        improved = True
                        if cur_max_err < best_max_err:
                            best_max_err = cur_max_err
                            best_nodes = cur_nodes.copy()
                            best_weights = cur_weights.copy()
                        break
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        # Decay step size for next restart
        step_size *= 0.8
    
    # Final normalization
    best_weights = np.abs(best_weights)
    best_weights /= best_weights.sum()
    best_nodes = np.sort(best_nodes)
    
    # Ensure strictly increasing
    for i in range(1, n):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-6
    best_nodes[0] = max(best_nodes[0], 1e-6)
    best_nodes[-1] = min(best_nodes[-1], 1 - 1e-6)
    
    return best_nodes, best_weights

def _evaluate_max_error(nodes, weights, cosmologies, z_vals):
    """Evaluate maximum relative error across all cosmologies and redshifts."""
    max_err = 0.0
    for w0, wa in cosmologies:
        for z in z_vals:
            a_i = 1.0 / (1.0 + z * nodes)
            quad_val = z * np.sum(weights / _E_cpl(a_i, w0, wa))
            true_val = _true_integral(z, w0, wa, 100)
            if abs(true_val) < 1e-15:
                continue
            rel_err = abs(quad_val - true_val) / abs(true_val)
            if rel_err > max_err:
                max_err = rel_err
    return max_err

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    try:
        nodes, weights = _optimize_quadrature(deadline)
    except Exception:
        # Fallback to midpoint rule
        n = 8
        nodes = (np.arange(n) + 0.5) / n
        weights = np.full(n, 1.0 / n)
    
    # Final validation and normalization
    weights = np.abs(weights)
    weights /= weights.sum()
    nodes = np.sort(nodes)
    
    # Ensure strictly increasing in [0,1]
    for i in range(1, len(nodes)):
        if nodes[i] <= nodes[i-1]:
            nodes[i] = nodes[i-1] + 1e-6
    nodes[0] = max(nodes[0], 1e-6)
    nodes[-1] = min(nodes[-1], 1 - 1e-6)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
