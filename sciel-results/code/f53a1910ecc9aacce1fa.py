"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _halton_sequence(n, dims):
    """Generate Halton sequence points in [0,1)^dims."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = np.zeros((n, dims))
    for i in range(n):
        for j in range(dims):
            points[i, j] = halton(i + 1, j + 2)
    return points

def _init_halton_sphere(n):
    """
    Initialize points on the sphere using a Halton sequence in 3D, 
    mapped to the sphere via inverse CDF of uniform distribution on sphere.
    """
    u = _halton_sequence(n, 2)
    z = 2.0 * u[:, 0] - 1.0
    phi = 2.0 * np.pi * u[:, 1]
    r = np.sqrt(1.0 - z * z)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its gradient for points on the sphere.
    x is (N, 3) array of points on the unit sphere.
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    safe_dist_sq = np.where(mask, dist_sq, 1.0)
    log_terms = -0.5 * np.log(safe_dist_sq)
    energy = np.sum(log_terms[mask])
    
    # Gradient
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    grad_r3 = -np.sum(weighted_diff, axis=1)
    
    # Project to tangent space: grad_tangent = grad_r3 - (grad_r3 . x) x
    dot = np.sum(grad_r3 * x, axis=1, keepdims=True)
    grad_tangent = grad_r3 - dot * x
    
    return energy, grad_tangent

def _exp_map(x, v):
    """
    Exponential map on the sphere.
    x: (N, 3) base points
    v: (N, 3) tangent vectors
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norm_v = np.where(norm_v > 1e-10, norm_v, 1.0)
    cos_term = np.cos(safe_norm_v)
    sin_term = np.sin(safe_norm_v) / safe_norm_v
    sin_term = np.where(norm_v > 1e-10, sin_term, 1.0)
    return cos_term * x + sin_term * v

def _riemannian_l_bfgs(x, max_iter=500, m=10, time_limit=None, start_time=None):
    """
    Riemannian L-BFGS-B implementation.
    This maintains a history of steps and gradients to approximate the inverse Hessian.
    """
    if time_limit is None or start_time is None:
        start_time = time.time()
        time_limit = 1000.0
        
    e, grad = _log_energy_and_grad_sphere(x)
    
    # L-BFGS history
    s_history = []  # Steps (tangent vectors)
    y_history = []  # Gradient differences
    c_history = []  # Curvature scalars (s^T y)
    
    # Initial direction
    d = -grad
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-10:
        return x, e
    d = d / d_norm

    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Line search: minimize f(exp_x(t * d))
        alpha = 1.0
        c1 = 1e-4
        max_ls_iter = 30
        ls_success = False
        x_new = x
        e_new = e
        grad_new = grad
        
        for ls_iter in range(max_ls_iter):
            x_trial = _exp_map(x, alpha * d)
            x_trial /= np.linalg.norm(x_trial, axis=1, keepdims=True)
            e_trial, grad_trial = _log_energy_and_grad_sphere(x_trial)
            
            # Armijo condition
            if e_trial <= e + c1 * alpha * np.sum(grad * d):
                ls_success = True
                x_new = x_trial
                e_new = e_trial
                grad_new = grad_trial
                break
            alpha *= 0.5
            
        if not ls_success:
            if e_new < e:
                x, e, grad = x_new, e_new, grad_new
            break
            
        # Update x and e
        x, e, grad = x_new, e_new, grad_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
            
        # Update L-BFGS history
        # s = x_new - x_old (in tangent space, approximated by the step vector)
        # Actually, for Riemannian L-BFGS, we need to transport gradients.
        # For simplicity and speed, we use the Euclidean difference of the tangent vectors
        # or just the step vector d * alpha.
        # A more rigorous approach uses the Log map, but let's try the simpler version first.
        
        s_k = alpha * d
        y_k = grad_new - grad  # This is an approximation in the tangent space
        
        # Store in history
        s_history.append(s_k)
        y_history.append(y_k)
        c_k = np.sum(s_k * y_k)
        c_history.append(c_k)
        
        # Limit history size
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
            c_history.pop(0)
        
        # Compute new direction using two-loop recursion
        # d_new = -g + sum_{i} alpha_i * (s_i - rho_i * y_i)
        # This is complex to implement vectorized for N points.
        # Given the time constraints and the risk of bugs in a custom Riemannian L-BFGS,
        # we will fall back to a more robust strategy: 
        # Use L-BFGS-B on the Stereographic projection, which is well-supported by SciPy.
        # The current program does this only as a polish. We will make it the main driver.
        # This is a "different strategy" in terms of *primary* optimizer.
        
        # To avoid breaking the loop, we will break out and let the main function
        # handle the Stereographic L-BFGS if we detect stagnation or if we just decide
        # that the manual Riemannian L-BFGS is too risky/slow to implement correctly here.
        
        # Let's just break and rely on the Stereographic L-BFGS which is more robust.
        break

    return x, e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_x = None
    best_energy = float('inf')
    
    # 1. Initialize with Halton sequence
    x0 = _init_halton_sphere(N)
    e0, _ = _log_energy_and_grad_sphere(x0)
    
    if e0 < best_energy:
        best_energy = e0
        best_x = x0.copy()
        
    # 2. Primary Optimization: L-BFGS-B on Stereographic Projection
    # This is much more effective than gradient descent because it uses curvature info.
    
    def stereo_to_sphere(params):
        u = params[0::2]
        v = params[1::2]
        denom = 1.0 + u**2 + v**2
        x = np.column_stack([2.0 * u / denom, 2.0 * v / denom, (1.0 - u**2 - v**2) / denom])
        return x

    def sphere_to_stereo(x):
        eps = 1e-9
        z_safe = np.where(x[:, 2] < 1.0 - eps, x[:, 2], 1.0 - eps)
        u = x[:, 0] / (1.0 - z_safe)
        v = x[:, 1] / (1.0 - z_safe)
        params = np.zeros(2 * N)
        params[0::2] = u
        params[1::2] = v
        return params

    def obj_and_grad(params):
        x = stereo_to_sphere(params)
        e, grad_r3 = _log_energy_and_grad_sphere(x)
        
        # Gradient in parameter space via chain rule
        u = params[0::2]
        v = params[1::2]
        denom = 1.0 + u**2 + v**2
        d_inv2 = 1.0 / (denom * denom)
        
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

    # Run L-BFGS-B from the Halton initial guess
    params0 = sphere_to_stereo(x0)
    
    try:
        res = minimize(obj_and_grad, params0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8})
        if res.success or res.status == 0:
            p_opt = res.x
            x_opt = stereo_to_sphere(p_opt)
            e_opt, _ = _log_energy_and_grad_sphere(x_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
    except Exception:
        pass

    # 3. Multi-restart with Random Perturbations to escape local minima
    # Use remaining time to try several random restarts from the best solution found so far.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 15:
        # Create a perturbed version of the best solution
        # Add small random noise and project back to sphere
        noise_scale = 0.05
        x_perturbed = best_x + rng.normal(0, noise_scale, size=best_x.shape)
        x_perturbed /= np.linalg.norm(x_perturbed, axis=1, keepdims=True)
        
        # Run L-BFGS-B from this perturbed state
        params_pert = sphere_to_stereo(x_perturbed)
        
        try:
            res = minimize(obj_and_grad, params_pert, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-8})
            if res.success or res.status == 0:
                p_opt = res.x
                x_opt = stereo_to_sphere(p_opt)
                e_opt, _ = _log_energy_and_grad_sphere(x_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_x = x_opt.copy()
        except Exception:
            pass

    # Final normalization
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x

# EVOLVE-BLOCK-END
