import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _energy_and_grad(x_flat):
    """
    Calculate the logarithmic energy and its Euclidean gradient 
    for points in R^3.
    x_flat: (3*N,) array
    Returns: energy (float), gradient (3*N,) array
    """
    x = x_flat.reshape(N, 3)
    
    # Pairwise differences
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask] + 1e-30)
    
    # Energy: sum log(1/d) = -sum log(d)
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient
    # d/dx_i E = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    weights = 1.0 / (dist_sq + 1e-30)
    np.fill_diagonal(weights, 0.0)
    
    # grad[i] = sum_j weights[i, j] * (x[j] - x[i])
    w_x = weights @ x
    w_sum = np.sum(weights, axis=1)
    
    grad_euclid = w_x - x * w_sum[:, np.newaxis]
    
    return energy, grad_euclid.flatten()

def _penalty_objective_and_grad(x_flat, lambda_val):
    """
    Objective function with a penalty for deviating from the unit sphere.
    F(x) = E(x) + lambda * sum_i (||x_i||^2 - 1)^2
    """
    x = x_flat.reshape(N, 3)
    
    # Get energy and gradient of E
    e, g_e = _energy_and_grad(x_flat)
    
    # Penalty term
    norms_sq = np.sum(x**2, axis=1)
    penalty = np.sum((norms_sq - 1.0)**2)
    
    # Gradient of penalty
    g_p = 4.0 * lambda_val * (norms_sq - 1.0)[:, np.newaxis] * x
    
    total_obj = e + lambda_val * penalty
    total_grad = g_e + g_p.flatten()
    
    return total_obj, total_grad

def _optimize_l_bfgs(x0, lambda_val=100.0, max_iter=500):
    """
    Optimize using L-BFGS-B with penalty.
    """
    x0_flat = x0.flatten()
    
    res = minimize(
        _penalty_objective_and_grad, 
        x0_flat, 
        args=(lambda_val,),
        method='L-BFGS-B', 
        jac=True,
        options={
            'maxiter': max_iter,
            'ftol': 1e-14,
            'gtol': 1e-10,
            'maxls': 50,
            'maxcor': 30,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    
    # Normalize to ensure validity
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _riemannian_grad_descent(x0, max_iter=3000, tol=1e-10, lr_init=0.05):
    """
    Riemannian gradient descent on the sphere with adaptive learning rate.
    """
    x = x0.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    lr = lr_init
    prev_energy = _energy_and_grad(x.flatten())[0]
    best_x = x.copy()
    best_e = prev_energy
    
    for i in range(max_iter):
        e, g = _energy_and_grad(x.flatten())
        g = g.reshape(N, 3)
        
        # Project gradient onto tangent space
        x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        g_tangent = g - np.sum(g * x_norm, axis=1, keepdims=True) * x_norm
        
        # Adaptive learning rate: decrease if energy increased
        x_new = x - lr * g_tangent
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
        
        e_new = _energy_and_grad(x_new.flatten())[0]
        
        if e_new < e:
            x = x_new
            lr = min(lr * 1.05, 0.1)
            if e_new < best_e:
                best_e = e_new
                best_x = x.copy()
        else:
            lr *= 0.7
            if lr < 1e-6:
                break
        
        # Convergence check
        if abs(prev_energy - e) < tol and i > 100:
            break
        prev_energy = e
    
    return best_x, best_e

def _newton_polish(x0, max_iter=50, tol=1e-12):
    """
    Newton-style polish using Gauss-Newton on the sphere constraint.
    Uses the Hessian approximation from the penalty formulation.
    """
    x = x0.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    for i in range(max_iter):
        e, g = _energy_and_grad(x.flatten())
        g = g.reshape(N, 3)
        
        # Project to tangent space
        g_tangent = g - np.sum(g * x, axis=1, keepdims=True) * x
        
        # Check convergence
        if np.max(np.linalg.norm(g_tangent, axis=1)) < tol:
            break
        
        # Simple Newton step: use diagonal approximation of Hessian
        # For log energy, Hessian diagonal ~ sum_j 1/||x_i - x_j||^2
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)
        hess_diag = np.sum(1.0 / dist_sq, axis=1)
        
        # Damped Newton step in tangent space
        step = g_tangent / hess_diag[:, np.newaxis]
        step_norm = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.where(step_norm > 1e-8, step_norm, 1.0)
        
        x_new = x - 0.1 * step
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
        
        e_new = _energy_and_grad(x_new.flatten())[0]
        if e_new < e:
            x = x_new
        else:
            break
    
    return x, _energy_and_grad(x.flatten())[0]

def _energy_eval(x):
    """
    High precision energy calculation for final evaluation.
    """
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    d_pairs = np.sqrt(dist_sq[mask] + 1e-30)
    return -math.fsum(np.log(d_pairs).tolist())

def _generate_initializations(rng):
    """Generate diverse initializations."""
    inits = []
    
    # Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # Perturbed Fibonacci Spirals with various noise levels and patterns
    for scale in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
    
    # Random initializations
    for _ in range(15):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
    
    # Spherical cap-based initialization
    # Split sphere into caps and place points
    for n_caps in [5, 10]:
        cap_centers = rng.normal(size=(n_caps, 3))
        cap_centers /= np.linalg.norm(cap_centers, axis=1, keepdims=True)
        x_cap = np.zeros((N, 3))
        points_per_cap = N // n_caps
        for c_idx in range(n_caps):
            start = c_idx * points_per_cap
            end = start + points_per_cap if c_idx < n_caps - 1 else N
            n_pts = end - start
            # Generate points in a small cap around cap_centers[c_idx]
            angles = rng.uniform(0, 2*np.pi, n_pts)
            radii = rng.uniform(0, 0.5, n_pts)
            local_x = radii * np.cos(angles)
            local_y = radii * np.sin(angles)
            local_z = np.sqrt(np.maximum(0, 1 - local_x**2 - local_y**2))
            # Rotate to align with cap center
            v = cap_centers[c_idx]
            if np.linalg.norm(v) < 1e-10:
                v = np.array([0, 0, 1.0])
            v = v / np.linalg.norm(v)
            # Build rotation matrix from z-axis to v
            z_axis = np.array([0, 0, 1.0])
            cross = np.cross(z_axis, v)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-10:
                R = np.eye(3)
            else:
                K = np.array([[0, -cross[2], cross[1]],
                             [cross[2], 0, -cross[0]],
                             [-cross[1], cross[0], 0]])
                cos_theta = np.dot(z_axis, v)
                R = np.eye(3) + K + K @ K * (1 - cos_theta) / cross_norm**2
            local_pts = np.column_stack([local_x, local_y, local_z])
            rotated = local_pts @ R.T
            x_cap[start:end] = rotated
        x_cap /= np.linalg.norm(x_cap, axis=1, keepdims=True)
        inits.append(x_cap)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    # Generate initializations
    inits = _generate_initializations(rng)
    
    # Phase 1: Optimize each initialization with high lambda
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        try:
            x_opt, e_opt = _optimize_l_bfgs(x0, lambda_val=50.0, max_iter=300)
            e_precise = _energy_eval(x_opt)
            
            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_e = _energy_eval(best_x)
        
    # Phase 2: Refine best solution with lower lambda
    if time.time() < deadline - 20.0:
        try:
            x_final, e_final = _optimize_l_bfgs(best_x, lambda_val=5.0, max_iter=500)
            e_final_precise = _energy_eval(x_final)
            if e_final_precise < best_e:
                best_x = x_final
                best_e = e_final_precise
        except Exception:
            pass
            
    # Phase 3: Riemannian gradient descent polish
    if time.time() < deadline - 15.0:
        try:
            x_rg, e_rg = _riemannian_grad_descent(best_x, max_iter=2000, tol=1e-10)
            e_rg_precise = _energy_eval(x_rg)
            if e_rg_precise < best_e:
                best_x = x_rg
                best_e = e_rg_precise
        except Exception:
            pass
            
    # Phase 4: Final Newton polish
    if time.time() < deadline - 5.0:
        try:
            x_newton, e_newton = _newton_polish(best_x, max_iter=30, tol=1e-12)
            e_newton_precise = _energy_eval(x_newton)
            if e_newton_precise < best_e:
                best_x = x_newton
                best_e = e_newton_precise
        except Exception:
            pass
    
    # Ensure valid output
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_x = best_x / norms
    
    return best_x
