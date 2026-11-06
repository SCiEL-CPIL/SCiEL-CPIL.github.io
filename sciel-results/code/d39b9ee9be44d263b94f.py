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
    Perform Riemannian Gradient Descent on the unit sphere with backtracking line search.
    """
    x = x0.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    lr = 0.05
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
        
        # Backtracking line search
        alpha = lr
        accepted = False
        for _ in range(20):
            x_new = x - alpha * g_tan
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            e_new, _ = _energy_and_grad(x_new.flatten())
            
            if e_new < e - 1e-4 * alpha * grad_norm * grad_norm:
                x = x_new
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # If no step accepted, try a smaller fixed step
            alpha = max(lr * 0.01, 1e-6)
            x_new = x - alpha * g_tan
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            e_new, _ = _energy_and_grad(x_new.flatten())
            if e_new < e:
                x = x_new
            else:
                break
        
        # Adapt learning rate based on progress
        if e_new < prev_energy - 1e-6:
            lr = min(lr * 1.2, 0.5)  # Increase slightly if making good progress
        else:
            lr *= 0.8  # Decrease if progress is slow
        prev_energy = e_new
        
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
            'gtol': 1e-10,
            'maxls': 100,
            'maxcor': 50,
            'disp': False
        }
    )
    
    x_opt = res.x.reshape(N, 3)
    
    # Normalize to ensure validity
    norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    x_opt = x_opt / norms
    
    return x_opt, res.fun

def _polish_newton(x0, max_iter=50, tol=1e-14):
    """
    Final polish using Newton's method on the Riemannian manifold.
    This solves the KKT conditions more precisely.
    """
    x = x0.copy()
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    
    for _ in range(max_iter):
        e, g_euclid = _energy_and_grad(x.flatten())
        g = g_euclid.reshape(N, 3)
        
        # Project to tangent space
        dot = np.sum(g * x, axis=1, keepdims=True)
        g_tan = g - dot * x
        
        grad_norm = np.linalg.norm(g_tan)
        if grad_norm < tol:
            break
        
        # Build Hessian approximation using finite differences of gradient
        # For efficiency, use a simplified approach: assume Hessian is approximately
        # the identity in tangent space scaled by a factor based on gradient magnitude
        
        # Simple Newton step: step = -H^{-1} * g_tan
        # We'll use a diagonal approximation
        # The Hessian of log energy has diagonal entries related to sum of 1/d^2
        
        # Compute diagonal of Hessian approximation
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        np.fill_diagonal(dist_sq, 1.0)  # Avoid division by zero
        
        # H_ii ≈ sum_{j != i} 2 / ||x_i - x_j||^2 (for the log kernel)
        # More precisely, for E = -sum log(d_ij), the Hessian diagonal is
        # sum_{j != i} (1/d_ij^2) * (something)
        
        # Use a simple scaling: h_diag = max(1, sum of weights)
        weights = 1.0 / (dist_sq + 1e-10)
        np.fill_diagonal(weights, 0.0)
        h_diag = np.sum(weights, axis=1)
        h_diag = np.maximum(h_diag, 1e-6)
        
        # Newton step in tangent space
        step = -g_tan / h_diag[:, np.newaxis]
        
        # Damping to ensure convergence
        alpha = 1.0
        for _ in range(10):
            x_new = x + alpha * step
            x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
            e_new, _ = _energy_and_grad(x_new.flatten())
            
            if e_new < e:
                x = x_new
                break
            alpha *= 0.5
        
        if alpha < 1e-6:
            break
    
    return x

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
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
        noise = rng.normal(0, scale, (N, 3))
        x_pert = x_fib + noise
        x_pert /= np.linalg.norm(x_pert, axis=1, keepdims=True)
        inits.append(x_pert)
        
    # Random Initializations
    for _ in range(15):
        x_rand = rng.normal(size=(N, 3))
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        inits.append(x_rand)
        
    # Spherical designs (approximate) using tetrahedral/octahedral symmetry
    # Start with icosahedron vertices and subdivide
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Generate points by projecting icosahedron vertices with perturbations
    for _ in range(3):
        x_ico_pert = ico + rng.normal(0, 0.3, ico.shape)
        x_ico_pert /= np.linalg.norm(x_ico_pert, axis=1, keepdims=True)
        # Expand to N points by adding more perturbed copies
        while len(x_ico_pert) < N:
            additional = rng.normal(size=(N - len(x_ico_pert), 3))
            additional /= np.linalg.norm(additional, axis=1, keepdims=True)
            x_ico_pert = np.vstack([x_ico_pert, additional])
        x_ico_pert = x_ico_pert[:N]
        inits.append(x_ico_pert)
        
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
    # Multi-stage refinement to extract maximum precision
    if time.time() < deadline - 5.0:
        try:
            # Stage 1: L-BFGS with low lambda
            x_refined, _ = _optimize_l_bfgs(best_x, lambda_val=1.0, max_iter=500, ftol=1e-14)
            e_refined = _energy_eval(x_refined)
            if e_refined < best_e:
                best_x = x_refined
                best_e = e_refined
        except Exception:
            pass
        
        # Stage 2: Newton polish
        if time.time() < deadline - 3.0:
            try:
                x_polished = _polish_newton(best_x, max_iter=50, tol=1e-15)
                e_polished = _energy_eval(x_polished)
                if e_polished < best_e:
                    best_x = x_polished
                    best_e = e_polished
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
