"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _log_energy(x):
    """
    Computes the logarithmic energy for points on the sphere.
    x is (N, 3) array of points on the unit sphere.
    Energy: E = sum_{i<j} log(1/||x_i - x_j||)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist is (N, N)
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    # sum_{i<j} log(1/dist) = 0.5 * sum_{i != j} log(1/dist)
    # log(1/d) = -log(d)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    return energy

def _log_energy_and_grad(x):
    """
    Computes the logarithmic energy and its gradient for points on the sphere.
    x is (N, 3) array of points on the unit sphere.
    Energy: E = sum_{i<j} log(1/||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    Gradient: grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # x is (N, 3)
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist is (N, N)
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    # sum_{i<j} log(1/dist) = 0.5 * sum_{i != j} log(1/dist)
    # log(1/d) = -log(d)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_sphere(n, bases=(2, 3, 5)):
    """Generate points on the sphere using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        points[i-1, 0] = halton(i, bases[0])
        points[i-1, 1] = halton(i, bases[1])
        points[i-1, 2] = halton(i, bases[2])
    
    # Map to sphere using inverse CDF of uniform distribution on sphere
    # z = 2*u - 1, phi = 2*pi*v
    # w = sqrt(1-z^2)
    z = 2.0 * points[:, 0] - 1.0
    phi = 2.0 * np.pi * points[:, 1]
    w = np.sqrt(1.0 - z * z)
    
    return np.column_stack([w * np.cos(phi), w * np.sin(phi), z])

def _normalize(x):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    return x / norms

def _exponential_map(x, v):
    """
    Exponential map on the unit sphere.
    x: point on sphere (N, 3)
    v: tangent vector (N, 3)
    Returns: new points on sphere
    """
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    
    # Handle small theta
    small = theta < 1e-8
    theta_safe = np.where(small, 1.0, theta)
    
    sin_theta = np.sin(theta_safe)
    cos_theta = np.cos(theta_safe)
    
    # v / sin(theta)
    v_over_sin = np.where(small, v, v / sin_theta)
    
    # x * cos(theta) + v/sin(theta) * sin(theta)
    # Wait, exp_x(v) = x cos(||v||) + v/sin(||v||) * sin(||v||)
    # If theta is small, v/sin(theta) ~ 1/theta, so v/sin(theta)*sin(theta) ~ v
    # Actually: exp_x(v) = x cos(theta) + (v/theta) sin(theta)
    
    v_over_theta = np.where(small, v, v / theta_safe)
    
    new_x = x * cos_theta + v_over_theta * sin_theta
    
    return _normalize(new_x)

def _project_to_tangent(x, v):
    """
    Project vector v onto the tangent space of the sphere at x.
    v_tangent = v - (v . x) x
    """
    dot = np.sum(v * x, axis=1, keepdims=True)
    return v - dot * x

def _riemannian_cg(x0, deadline, max_iter=100, tol=1e-12):
    """
    Riemannian Conjugate Gradient method on the sphere.
    """
    x = x0.copy()
    e, g = _log_energy_and_grad(x)
    
    # Project gradient to tangent space
    g_tan = _project_to_tangent(x, g)
    
    d = -g_tan
    d_norm = np.linalg.norm(d)
    if d_norm < tol:
        return x, e
    
    # Normalize initial direction? No, keep magnitude for step size calculation.
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        # Line search: minimize f(exp_x(t * d))
        # Use simple backtracking line search
        alpha = 1.0
        e_new = None
        
        for ls_iter in range(20):
            x_trial = _exponential_map(x, alpha * d)
            e_trial, g_trial = _log_energy_and_grad(x_trial)
            
            # Check for decrease
            # Simple check: e_trial < e
            if e_trial < e:
                e_new = e_trial
                g_trial_tan = _project_to_tangent(x_trial, g_trial)
                break
            
            alpha *= 0.5
            if alpha < 1e-10:
                # No progress, break
                return x, e
                
        if e_new is None:
            break
            
        # Update x
        x = x_trial
        e = e_new
        
        # Check convergence
        if np.linalg.norm(g_trial_tan) < tol:
            break
            
        # Update direction (Polak-Ribiere)
        # beta = <g_new, g_new - g_old> / ||g_old||^2
        # Note: g_old is g_tan, g_new is g_trial_tan
        # But we need to parallel transport g_old to x_new?
        # For simplicity, we use the standard CG update in tangent space at the new point.
        # However, g_old is in T_x M, g_new is in T_x_new M.
        # A common approximation is to just compute beta using the norms or a simplified dot product.
        # Let's use a simpler restart strategy if beta < 0.
        
        g_old_norm_sq = np.sum(g_tan * g_tan)
        if g_old_norm_sq < 1e-20:
            d = -g_trial_tan
        else:
            # Approximate inner product by projecting g_old to x_new? 
            # Or just restart if it's not decreasing.
            # Let's try a simple momentum-like update or just restart.
            # Standard RCG often uses parallel transport.
            # For now, let's restart if the new gradient is not aligned or just use standard CG formula 
            # which assumes Euclidean. This might be unstable.
            
            # Let's use a safe restart: d = -g_trial_tan
            # This is essentially gradient descent with momentum if we keep some history, 
            # but pure restart is safer.
            
            # Actually, let's try to implement a basic parallel transport for g_old.
            # PT(g_old) = g_old + (g_old . x) x_new - (g_old . x_new) x
            # But g_old is tangent to x, so g_old . x = 0.
            # PT(g_old) = g_old - (g_old . x_new) x
            # Wait, x is old x, x_new is new x.
            # PT(g_old) from x to x_new:
            # v_new = v_old + (v_old . x_old) x_new - (v_old . x_new) x_old
            # Since v_old is tangent to x_old, v_old . x_old = 0.
            # v_new = v_old - (v_old . x_new) x_old
            
            g_old_pt = g_tan - np.sum(g_tan * x, axis=1, keepdims=True) * x_new
            # Wait, x_new is x_trial.
            g_old_pt = g_tan - np.sum(g_tan * x, axis=1, keepdims=True) * x_trial
            # This projects g_tan (which is in T_x) to be tangent to x_trial?
            # No, parallel transport maps T_x to T_x_new.
            # The formula is: PT_{x -> x_new}(v) = v + (v . x) x_new - (v . x_new) x
            # Since v . x = 0, PT(v) = v - (v . x_new) x
            # Let's check if this is tangent to x_new:
            # PT(v) . x_new = (v - (v . x_new) x) . x_new = v . x_new - (v . x_new) (x . x_new)
            # This is not necessarily 0.
            
            # Correct parallel transport on sphere:
            # PT(v) = v + (v . x) x_new - (v . x_new) x
            # If v is tangent to x, v . x = 0.
            # PT(v) = v - (v . x_new) x
            # Let's re-evaluate tangency:
            # PT(v) . x_new = v . x_new - (v . x_new) (x . x_new)
            # This is only 0 if x . x_new = 1 (same point) or v . x_new = 0.
            
            # Actually, the correct parallel transport of a tangent vector v at x to x' is:
            # v' = v + (v . x) x' - (v . x') x
            # If v . x = 0, v' = v - (v . x') x
            # Let's check v' . x' = v . x' - (v . x') (x . x')
            # This is not zero.
            
            # Let's use a simpler approach: Restart CG every 5 iterations or if beta < 0.
            # Or just use Gradient Descent with adaptive step size.
            # Given the complexity and time, let's stick to a robust Gradient Descent 
            # with Armijo line search, which is effectively what L-BFGS does but simpler.
            
            d = -g_trial_tan
            g_tan = g_trial_tan
            continue
            
        # If we get here, we updated d (which we set to -g_trial_tan in the else block above effectively)
        # Let's just use gradient descent to be safe and robust.
        g_tan = g_trial_tan
        
    return x, e

def _polish_with_lbfgs(x, deadline, tight=True):
    """
    Polish a configuration using L-BFGS-B in stereographic coordinates.
    Returns the polished configuration and its energy.
    """
    # Check if any point is too close to the south pole (0,0,-1)
    # Stereographic projection from south pole is singular at south pole.
    # If a point is close to (0,0,-1), u,v become large.
    # This is numerically okay if we use double precision, but let's check.
    
    # Convert to stereographic
    eps = 1e-9
    z_safe = np.where(x[:, 2] < 1.0 - eps, x[:, 2], 1.0 - eps)
    u = x[:, 0] / (1.0 - z_safe)
    v = x[:, 1] / (1.0 - z_safe)
    
    p0 = np.zeros(2 * N)
    p0[0::2] = u
    p0[1::2] = v
    
    # Define objective and gradient in parameter space
    def obj_and_grad(params):
        u = params[0::2]
        v = params[1::2]
        denom = 1.0 + u**2 + v**2
        x = np.column_stack([
            2.0 * u / denom,
            2.0 * v / denom,
            (1.0 - u**2 - v**2) / denom
        ])
        
        # Compute energy and gradient in R^3
        e, grad_r3 = _log_energy_and_grad(x)
        
        # Chain rule: grad_params = J^T grad_r3
        d = denom
        d_inv2 = 1.0 / (d * d)
        
        dx_du = 2.0 * (1.0 - u**2 + v**2) * d_inv2
        dx_dv = -4.0 * u * v * d_inv2
        dy_du = -4.0 * u * v * d_inv2
        dy_dv = 2.0 * (1.0 + u**2 - v**2) * d_inv2
        dz_du = -4.0 * u * d_inv2
        dz_dv = -4.0 * v * d_inv2
        
        grad_params = np.zeros(2 * N)
        grad_params[0::2] = grad_r3[:, 0] * dx_du + grad_r3[:, 1] * dy_du + grad_r3[:, 2] * dz_du
        grad_params[1::2] = grad_r3[:, 0] * dx_dv + grad_r3[:, 1] * dy_dv + grad_r3[:, 2] * dz_dv
        
        return e, grad_params
    
    try:
        # Use a tighter tolerance for polishing
        if tight:
             options = {'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-14}
        else:
             options = {'maxiter': 200, 'maxfun': 400, 'gtol': 1e-10}
             
        res = minimize(obj_and_grad, p0, method='L-BFGS-B', jac=True, options=options)
        
        if res.success or res.status == 0:
            p_opt = res.x
            u = p_opt[0::2]
            v = p_opt[1::2]
            denom = 1.0 + u**2 + v**2
            x_opt = np.column_stack([
                2.0 * u / denom,
                2.0 * v / denom,
                (1.0 - u**2 - v**2) / denom
            ])
            x_opt = _normalize(x_opt)
            e_opt, _ = _log_energy_and_grad(x_opt)
            return x_opt, e_opt
    except Exception:
        pass
    
    return x, _log_energy(x)

def construct():
    start_time = time.time()
    # Reserve 15 seconds for final polishing and safety margin
    deadline_search = start_time + 100.0
    deadline_polish = start_time + 115.0
    deadline_total = start_time + 118.0
    
    # Generate multiple initial guesses
    rng = np.random.default_rng(42)
    
    # 1. Fibonacci sphere
    x_fib = _fibonacci_sphere(N)
    
    # 2. Halton sphere
    x_halton = _halton_sphere(N)
    
    # 3-5. Random points
    initial_guesses = [x_fib, x_halton]
    for _ in range(3):
        pts = rng.normal(size=(N, 3))
        pts = _normalize(pts)
        initial_guesses.append(pts)
        
    best_overall_x = None
    best_overall_energy = float('inf')
    
    # Run RCG on each initial guess
    for x0 in initial_guesses:
        if time.time() > deadline_search:
            break
            
        # Run RCG (which is essentially robust gradient descent)
        x_rcg, e_rcg = _riemannian_cg(x0, deadline_search, max_iter=200)
        
        if e_rcg < best_overall_energy:
            best_overall_energy = e_rcg
            best_overall_x = x_rcg.copy()
            
    # If no RCG succeeded or we didn't find anything, fallback to Fibonacci
    if best_overall_x is None:
        best_overall_x = x_fib
        best_overall_energy = _log_energy(x_fib)
        
    # Final polish with L-BFGS (tight)
    x_polished, e_polished = _polish_with_lbfgs(best_overall_x, deadline_polish, tight=True)
    
    if e_polished < best_overall_energy:
        best_overall_x = x_polished
        best_overall_energy = e_polished
        
    # Final normalization to be safe
    best_overall_x = _normalize(best_overall_x)
    
    return best_overall_x

# EVOLVE-BLOCK-END
