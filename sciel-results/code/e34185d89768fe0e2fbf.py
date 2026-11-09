import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Representative cosmologies: (Omega_m, w0, w1)
    cosmo_params = [
        (0.3, -1.0, 0.0),
        (0.3, -1.2, 0.0),
        (0.3, -0.8, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
        (0.2, -1.0, 0.0),
        (0.4, -1.0, 0.0),
        (0.3, -1.5, 0.3),
        (0.3, -0.5, -0.3),
        (0.35, -1.1, 0.2),
    ]

    # Test redshifts - use a dense set for robustness
    z_test = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])

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
        n = 8
        nodes, weights = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        weights = weights / weights.sum()
        return {"nodes": nodes, "weights": weights}

    def parametric_nodes(p):
        """
        Generate 8 strictly increasing nodes in (0,1) from parameter p.
        p controls the spacing: p=0 gives uniform, p>0 clusters near 0, p<0 clusters near 1.
        Uses a power-law transform: x_i = (i/8)^(1/(1+p)) for i=1..8, then mapped to (0,1)
        Actually, let's use: t_i = (i - 0.5) / 8 for i=1..8 (midpoints), then apply a smooth
        deformation.
        
        Better approach: use a beta-like distribution.
        nodes_i = i / (N+1) transformed by a smooth function.
        
        Let's use: base_i = i / (N+1) for i=1..N
        Then apply: node_i = base_i^alpha * (1-base_i)^(1-alpha) / (max_value)
        This is getting complex. Let's use a simpler approach:
        
        nodes_i = (i/N) for i=1..N, then apply a smooth warp.
        
        Simplest: use a single parameter that blends between uniform and a specific non-uniform pattern.
        Or: use two parameters to control clustering at both ends.
        """
        N = 8
        # Base nodes: uniform spacing
        base = np.linspace(0.0, 1.0, N + 2)[1:-1]  # 8 points, excluding 0 and 1
        
        # Apply a smooth deformation using parameter p
        # For p=0: uniform
        # For p>0: cluster more near 0 (where integrand may vary faster for some models)
        # For p<0: cluster more near 1
        
        # Use: x' = x^a where a = 1/(1+p) for p > -1
        # But this only affects one end. Let's use a symmetric approach.
        
        # Alternative: use a quadratic warp
        # x' = x + p * x * (1-x)  -- this is a simple symmetric warp
        # For p>0: pushes middle out, clusters ends
        # For p<0: pulls middle in, spreads ends
        
        p_val = p
        warped = base + p_val * base * (1.0 - base)
        
        # Ensure nodes are in (0, 1) and strictly increasing
        warped = np.clip(warped, 1e-8, 1.0 - 1e-8)
        
        # Ensure strict increase
        for i in range(1, N):
            if warped[i] <= warped[i-1]:
                warped[i] = warped[i-1] + 1e-10
        
        return warped

    def compute_errors(nodes):
        """Given fixed nodes, compute the least-squares optimal weights and return worst error."""
        N = 8
        num_cases = num_cosmo * num_z
        
        # Build the design matrix: each row is one (cosmo, z) case
        # A[row, col] = integrand at node col for that case
        A = np.zeros((num_cases, N))
        b = np.zeros(num_cases)
        
        case_idx = 0
        for i, (Om, w0, w1) in enumerate(cosmo_params):
            for j, z in enumerate(z_test):
                z_u = z * nodes
                E_vals = E_func(z_u, Om, w0, w1)
                A[case_idx, :] = 1.0 / E_vals
                b[case_idx] = true_vals[i, j]
                case_idx += 1
        
        # Solve least squares: minimize ||A @ w - b||^2
        # With constraint that weights are positive and sum to 1
        # First, get unconstrained solution
        w_ls, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Project to positive weights summing to 1
        # Use a simple iterative projection
        w = w_ls.copy()
        
        # Iterative projection onto {w >= 0, sum(w) = 1}
        for _ in range(100):
            # Project onto positive orthant
            w = np.maximum(w, 0)
            # Project onto simplex (sum = 1)
            if w.sum() > 0:
                w = w / w.sum()
            else:
                w = np.ones(N) / N
                break
            # Check convergence
            if np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-12:
                break
        
        # Ensure strict positivity
        w = np.maximum(w, 1e-15)
        w = w / w.sum()
        
        # Compute worst relative error
        max_err = 0.0
        case_idx = 0
        for i in range(num_cosmo):
            for j in range(num_z):
                quad_val = np.dot(w, A[case_idx, :])
                true_val = b[case_idx]
                if true_val > 0:
                    err = abs(quad_val - true_val) / true_val
                    if err > max_err:
                        max_err = err
                case_idx += 1
        
        return max_err, w

    # Search over the node parameter p
    best_p = 0.0
    best_w = None
    best_err = float('inf')
    
    # Coarse search
    p_values = np.linspace(-0.5, 0.5, 21)
    for p in p_values:
        if time.time() > deadline - 10:
            break
        nodes = parametric_nodes(p)
        err, w = compute_errors(nodes)
        if err < best_err:
            best_err = err
            best_p = p
            best_w = w
    
    # Fine search around best
    p_fine = np.linspace(best_p - 0.1, best_p + 0.1, 21)
    for p in p_fine:
        if time.time() > deadline - 5:
            break
        nodes = parametric_nodes(p)
        err, w = compute_errors(nodes)
        if err < best_err:
            best_err = err
            best_p = p
            best_w = w
    
    nodes = parametric_nodes(best_p)
    weights = best_w
    
    # Final validation
    if not (len(nodes) == 8 and len(weights) == 8):
        n = 8
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
        gl_nodes = 0.5 * (gl_nodes + 1.0)
        gl_weights = 0.5 * gl_weights
        gl_weights = gl_weights / gl_weights.sum()
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    if not np.all(np.diff(nodes) > 0):
        n = 8
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
        gl_nodes = 0.5 * (gl_nodes + 1.0)
        gl_weights = 0.5 * gl_weights
        gl_weights = gl_weights / gl_weights.sum()
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    if not np.all(nodes > 0) or not np.all(nodes < 1):
        n = 8
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
        gl_nodes = 0.5 * (gl_nodes + 1.0)
        gl_weights = 0.5 * gl_weights
        gl_weights = gl_weights / gl_weights.sum()
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    if not np.all(weights > 0):
        n = 8
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n)
        gl_nodes = 0.5 * (gl_nodes + 1.0)
        gl_weights = 0.5 * gl_weights
        gl_weights = gl_weights / gl_weights.sum()
        return {"nodes": gl_nodes, "weights": gl_weights}
    
    if abs(np.sum(weights) - 1.0) > 1e-12:
        weights = weights / weights.sum()
    
    return {"nodes": nodes, "weights": weights}
