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
    
    # Add small epsilon to avoid division by zero in gradient, 
    # but for energy we need to be careful. 
    # If distance is 0, energy is inf. 
    # We assume points are distinct.
    d_pairs = np.sqrt(dist_sq[mask] + 1e-30)
    
    # Energy: sum log(1/d) = -sum log(d)
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient
    # d/dx_i E = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    # Note: dist_sq[i, j] = ||x_i - x_j||^2
    # weights[i, j] = 1 / dist_sq[i, j]
    weights = 1.0 / (dist_sq + 1e-30)
    np.fill_diagonal(weights, 0.0)
    
    # grad[i] = sum_j weights[i, j] * (x[j] - x[i])
    # sum_j weights[i, j] * x[j] - x[i] * sum_j weights[i, j]
    w_x = weights @ x
    w_sum = np.sum(weights, axis=1)
    
    grad_euclid = w_x - x * w_sum[:, np.newaxis]
    
    return energy, grad_euclid.flatten()

def _riemannian_grad_descent(x0, max_iter=2000, tol=1e-8, adaptive_lr=True):
    """
    Perform Riemannian Gradient Descent on the unit sphere.
    """
    x = x0.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    lr = 0.01
    prev_energy = _energy_and_grad(x.flatten())[0]
    
    for i in range(max_iter):
        e, g_euclid = _energy_and_grad(x.flatten())
        
        # Project gradient to tangent space: g_tan = g - (g . x) x
        x = x.reshape(N, 3)
        g = g_euclid.reshape(N, 3)
        dot = np.sum(g * x, axis=1, keepdims=True)
        g_tan = g - dot * x
        
        # Check convergence
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
        
        if adaptive_lr:
            # Simple adaptive step size: try to reduce energy
            # We can use a simple backtracking line search or fixed small step
            # For stability, we use a small fixed step initially, 
            # but adaptive is better. Let's use a simple heuristic.
            # If energy increases, reduce lr.
            pass # Keeping it simple for now, fixed lr might be enough for coarse phase
            
        # Update: x_new = x - lr * g_tan
        x_new = x - lr * g_tan
        
        # Retraction: normalize to sphere
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
        
        # Check if energy decreased
        e_new, _ = _energy_and_grad(x_new.flatten())
        
        if e_new > e:
            # Reduce learning rate
            lr *= 0.5
            if lr < 1e-8:
                break
        else:
            x = x_new
            # Optionally increase lr slightly if progress is good? 
            # Keeping it conservative.
            
    return x

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
    # d/dx_i (||x_i||^2 - 1)^2 = 2 * (||x_i||^2 - 1) * 2 * x_i = 4 * (||x_i||^2 - 1) * x_i
    g_p = 4.0 * lambda_val * (norms_sq - 1.0)[:, np.newaxis] * x
    
    total_obj = e + lambda_val * penalty
    total_grad = g_e + g_p.flatten()
    
    return total_obj, total_grad

def _optimize_l_bfgs(x0, lambda_val=100.0, max_iter=1000, ftol=1e-12):
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
            'ftol': ftol,
            'gtol': 1e-8,
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

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Leave a safety margin of 10s.
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_x = None
    best_e = float('inf')
    
    # 1. Generate Initializations
    inits = []
    
    # Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(x_fib)
    
    # Perturbed Fibonacci Spirals
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
        
    # Random Initializations
    for _ in range(20):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
        
    # Optimize each initialization
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        try:
            # Phase 1: Coarse Riemannian Gradient Descent
            # This helps escape very shallow local minima and move to a better basin
            # We use a moderate number of iterations
            x_rgd = _riemannian_grad_descent(x0, max_iter=500, tol=1e-6)
            
            # Phase 2: Fine L-BFGS-B optimization
            # Use a high lambda to enforce sphere constraint strongly
            x_opt, e_opt = _optimize_l_bfgs(x_rgd, lambda_val=100.0, max_iter=500, ftol=1e-12)
            
            # Verify with high precision energy
            e_precise = _energy_eval(x_opt)
            
            if e_precise < best_e:
                best_e = e_precise
                best_x = x_opt.copy()
                
        except Exception:
            continue
            
    if best_x is None:
        best_x = x_fib
        best_e = _energy_eval(best_x)
        
    # Final polish on the best solution
    # Try a second round with lower lambda and tighter tolerance to refine
    if time.time() < deadline - 5.0:
        try:
            # Refine with lower lambda, allowing the energy landscape to dominate
            # This can help if the penalty was distorting the local minimum
            x_final, e_final = _optimize_l_bfgs(best_x, lambda_val=10.0, max_iter=1000, ftol=1e-14)
            e_final_precise = _energy_eval(x_final)
            if e_final_precise < best_e:
                best_x = x_final
                best_e = e_final_precise
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
