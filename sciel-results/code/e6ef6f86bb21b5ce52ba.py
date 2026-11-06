"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, differential_evolution

def _E_of_a(a, wa, wb, w0):
    """H(a)/H0 for flat LCDM-like model with dark energy."""
    matter = (1.0 - a)**3
    de = w0 + wb * (1.0 - a)
    de_factor = a**(-3.0 * (1.0 + de))
    return np.sqrt(matter + de_factor)

def _integrand_z(z, wa, wb, w0):
    """Integrand for dL/(c/H0) = (1+z)*z*int_0^z dz'/E(a')."""
    a = 1.0 / (1.0 + z)
    return a / _E_of_a(a, wa, wb, w0)

def _exact_dL(z, wa, wb, w0):
    """High-accuracy adaptive integration of dL/(c/H0)."""
    from scipy.integrate import quad
    def f(zp):
        return _integrand_z(zp, wa, wb, w0)
    val, _ = quad(f, 0, z, limit=200)
    return (1.0 + z) * z * val

def _quad_dL(z, nodes, weights, wa, wb, w0):
    """Quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, wa, wb, w0)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _relative_error(z, nodes, weights, wa, wb, w0):
    exact = _exact_dL(z, wa, wb, w0)
    approx = _quad_dL(z, nodes, weights, wa, wb, w0)
    return abs(approx - exact) / exact * 100.0

def _chebyshev_nodes(n):
    """Generate Chebyshev-Gauss-Lobatto nodes mapped to [0,1]."""
    if n == 1:
        return np.array([0.5])
    k = np.arange(n)
    # Standard Chebyshev-Lobatto nodes on [-1, 1]
    x = np.cos(np.pi * k / (n - 1))
    # Map from [-1, 1] to [0, 1]
    x = (x + 1.0) / 2.0
    # Sort in increasing order
    x = np.sort(x)
    return x

def _chebyshev_weights(n):
    """Generate Chebyshev-Gauss-Lobatto weights mapped to [0,1]."""
    if n == 1:
        return np.array([1.0])
    k = np.arange(n)
    # Standard Chebyshev-Lobatto weights on [-1, 1]
    w = np.ones(n)
    w[0] = 0.5
    w[-1] = 0.5
    # Multiply by pi/(n-1) for the integral
    w = w * np.pi / (n - 1)
    # Map from [-1, 1] to [0, 1]: integral over [0,1] = integral over [-1,1] / 2
    w = w / 2.0
    # Sort weights to match sorted nodes
    # Need to reverse since we sorted x in increasing order
    w = w[::-1]
    # Normalize to sum to 1
    w = w / np.sum(w)
    return w

def _find_optimal_weights_for_nodes(nodes, z_samples, models, max_iter=30):
    """Find weights that minimize worst-case relative error via iterative LP."""
    n = len(nodes)
    n_cosmo = len(models)
    n_z = len(z_samples)
    
    # Build constraint matrix: for each (cosmo, z), we want
    # sum_i w_i / E_i(cosmo, z) ≈ exact_integral(cosmo, z)
    # where exact_integral = exact_dL / ((1+z)*z)
    
    A = np.zeros((n_cosmo * n_z, n))
    b_exact = np.zeros(n_cosmo * n_z)
    
    for ci, (wa, wb, w0) in enumerate(models):
        for zi, z in enumerate(z_samples):
            idx = ci * n_z + zi
            a = 1.0 / (1.0 + z * nodes)
            E_vals = _E_of_a(a, wa, wb, w0)
            A[idx, :] = 1.0 / E_vals
            exact_dL_val = _exact_dL(z, wa, wb, w0)
            if z > 0:
                b_exact[idx] = exact_dL_val / ((1.0 + z) * z)
            else:
                b_exact[idx] = 0.0
    
    # Start with Chebyshev weights as initial guess
    w = _chebyshev_weights(n)
    
    for it in range(max_iter):
        # Compute current errors
        Aw = A @ w
        rel_errs = np.abs(Aw - b_exact) / np.maximum(b_exact, 1e-15)
        t = np.max(rel_errs)
        
        # Identify active constraints (largest errors)
        active_idx = np.where(rel_errs > 0.8 * t)[0]
        
        # Solve LP to minimize t with active constraints
        # Variables: w_0..w_{n-1}, t
        # Minimize t
        c = np.zeros(n + 1)
        c[-1] = 1.0
        
        # Constraints:
        # For each active row i: A_i @ w - t * b_i <= 0  and  -A_i @ w + t * b_i <= 0
        A_ub = np.zeros((2 * len(active_idx), n + 1))
        b_ub = np.zeros(2 * len(active_idx))
        
        for k, idx in enumerate(active_idx):
            A_ub[2*k, :n] = A[idx, :]
            A_ub[2*k, n] = -b_exact[idx]
            A_ub[2*k+1, :n] = -A[idx, :]
            A_ub[2*k+1, n] = b_exact[idx]
        
        # Equality: sum(w) = 1
        A_eq = np.zeros((1, n + 1))
        A_eq[0, :n] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: w >= 1e-10, t >= 0
        bounds = [(1e-10, None)] * n + [(0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                         bounds=bounds, method='highs')
            if res.success:
                w_new = res.x[:n]
                t_new = res.x[n]
                if t_new < t * 0.999:
                    w = w_new
                    t = t_new
                else:
                    break
            else:
                break
        except:
            break
    
    return w, t

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline (10s margin)
    
    # Diverse cosmology sample for optimization
    models = [
        (0.0, 0.0, -1.0),       # LambdaCDM
        (0.1, 0.0, -0.9),       # Mild DE
        (0.0, 0.3, -1.0),       # CPL with w_a>0
        (0.0, -0.5, -1.0),      # CPL with w_a<0
        (0.2, 0.5, -0.8),       # Stronger matter, mild DE
        (0.0, 0.0, -0.5),       # Quintessence-like
        (0.1, -0.3, -1.2),      # Phantom-like
        (0.0, 0.8, -0.7),       # Strong w_a
    ]
    
    # Z samples for evaluation
    z_samples = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    n = 8
    
    # Use Chebyshev nodes as initial and primary candidate
    best_nodes = _chebyshev_nodes(n)
    best_weights = _chebyshev_weights(n)
    
    # Evaluate initial Chebyshev rule
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    best_score = np.max(errors)
    
    # Try to improve weights for Chebyshev nodes
    if time.time() < deadline - 10:
        improved_weights, t = _find_optimal_weights_for_nodes(best_nodes, z_samples, models, max_iter=20)
        errors = []
        for z in z_samples:
            for wa, wb, w0 in models:
                err = _relative_error(z, best_nodes, improved_weights, wa, wb, w0)
                errors.append(err)
        score_improved = np.max(errors)
        if score_improved < best_score:
            best_weights = improved_weights
            best_score = score_improved
    
    # Try differential evolution with Chebyshev as starting point
    if time.time() < deadline - 30:
        def objective(nodes_flat):
            nodes = np.sort(nodes_flat)
            # Check strictly increasing
            if np.any(np.diff(nodes) <= 1e-12):
                return 1e6
            if nodes[0] < 0 or nodes[-1] > 1:
                return 1e6
            
            # Find optimal weights for these nodes
            weights, t = _find_optimal_weights_for_nodes(nodes, z_samples, models, max_iter=15)
            
            # Evaluate worst-case error
            errs = []
            for z in z_samples:
                for wa, wb, w0 in models:
                    err = _relative_error(z, nodes, weights, wa, wb, w0)
                    errs.append(err)
            return np.max(errs)
        
        bounds = [(0.001, 0.999)] * n
        
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=30, popsize=10,
                tol=1e-5, seed=42,
                polish=False,
                workers=1,
                time_limit=25
            )
            
            if result.fun < best_score:
                candidate_nodes = np.sort(result.x)
                if np.all(np.diff(candidate_nodes) > 1e-12):
                    candidate_weights, _ = _find_optimal_weights_for_nodes(candidate_nodes, z_samples, models, max_iter=20)
                    
                    # Final evaluation
                    errors = []
                    for z in z_samples:
                        for wa, wb, w0 in models:
                            err = _relative_error(z, candidate_nodes, candidate_weights, wa, wb, w0)
                            errors.append(err)
                    final_score = np.max(errors)
                    if final_score < best_score:
                        best_nodes = candidate_nodes
                        best_weights = candidate_weights
                        best_score = final_score
        except Exception as e:
            pass
    
    # Ensure valid output
    best_nodes = np.sort(best_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    
    # Final validation
    errors = []
    for z in z_samples:
        for wa, wb, w0 in models:
            err = _relative_error(z, best_nodes, best_weights, wa, wb, w0)
            errors.append(err)
    
    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
