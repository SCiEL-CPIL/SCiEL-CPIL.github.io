"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math
import time

N = 150

def _energy_and_grad(X, penalty_weight=0.0, min_dist=1e-6):
    """Compute logarithmic energy with optional penalty for minimum distance.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Riemannian Gradient w.r.t. x_i = (I - x_i x_i^T) * (sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2)
    
    If penalty_weight > 0, adds a penalty term for pairs closer than min_dist:
    penalty = penalty_weight * sum_{i<j} max(0, min_dist - |x_i - x_j|)^2
    """
    n = X.shape[0]
    # Compute all pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)  # avoid division by zero on diagonal
    dist = np.sqrt(dist2)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Euclidean Gradient: grad_eucl[i] = sum_{j!=i} (x_j - x_i) / dist[i,j]^2
    # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # grad_eucl[i] = - sum_j diff[i,j] / dist2[i,j]
    eucl_grad = -np.sum(diff / dist2[:, :, np.newaxis], axis=1)
    
    # Add penalty gradient if penalty_weight > 0
    if penalty_weight > 0:
        # Penalty: sum_{i<j} max(0, min_dist - dist[i,j])^2
        # Gradient of penalty w.r.t. x_i: for each j, if dist[i,j] < min_dist:
        #   2*(min_dist - dist[i,j]) * (x_i - x_j) / dist[i,j]
        penalty_grad = np.zeros_like(X)
        for i in range(n):
            for j in range(i+1, n):
                d = dist[i, j]
                if d < min_dist and d > 1e-15:
                    factor = 2.0 * penalty_weight * (min_dist - d) / d
                    penalty_grad[i] += factor * (X[i] - X[j])
                    penalty_grad[j] -= factor * (X[i] - X[j])
        
        eucl_grad += penalty_grad
        # Add penalty to energy
        penalty_val = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = dist[i, j]
                if d < min_dist:
                    penalty_val += penalty_weight * (min_dist - d)**2
        energy += penalty_val
    
    # Project to tangent space: grad_riem[i] = eucl_grad[i] - (eucl_grad[i] . X[i]) X[i]
    dot = np.sum(eucl_grad * X, axis=1, keepdims=True)
    riem_grad = eucl_grad - dot * X
    
    return energy, riem_grad

def _project_to_sphere(X):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _halton_init(n, num_dims=3):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(1, n + 1):
        points[i - 1] = [halton(i, 2), halton(i, 3), halton(i, 5)]
    
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms
    
    return points

def _fibonacci_init(n):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _riemannian_cg(X0, max_iter=1000, tol=1e-10, time_limit=30.0, 
                   penalty_weight=0.0, min_dist=1e-6):
    """Riemannian Conjugate Gradient method with backtracking line search.
    
    Uses the exponential map for steps and Polak-Ribiere beta update.
    """
    start_time = time.time()
    X = X0.copy()
    energy, grad = _energy_and_grad(X, penalty_weight, min_dist)
    grad_norm = np.linalg.norm(grad)
    
    if grad_norm < tol:
        return X, energy
    
    # Initialize conjugate gradient variables
    beta = 0.0
    p = -grad  # initial search direction
    
    alpha = 0.1  # initial step size
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Backtracking line search
        alpha_trial = alpha
        count = 0
        while True:
            # Take step using exponential map (approximately: X_new = normalize(X + alpha * p))
            X_new = X + alpha_trial * p
            X_new = _project_to_sphere(X_new)
            
            energy_new, _ = _energy_and_grad(X_new, penalty_weight, min_dist)
            
            # Armijo condition
            if energy_new <= energy - 1e-4 * alpha_trial * np.sum(grad * p):
                break
            alpha_trial *= 0.5
            count += 1
            if alpha_trial < 1e-15 or count > 50:
                break
        
        if alpha_trial < 1e-15:
            # Reset direction if stuck
            p = -grad
            beta = 0.0
            alpha = max(alpha * 0.5, 1e-10)
            continue
        
        # Update
        X = X_new
        energy = energy_new
        _, grad_new = _energy_and_grad(X, penalty_weight, min_dist)
        grad_norm = np.linalg.norm(grad_new)
        
        if grad_norm < tol:
            break
        
        # Polak-Ribiere beta
        dot_old_new = np.sum(grad * grad_new)
        dot_new = np.sum(grad_new * grad_new)
        if dot_new > 1e-15:
            beta = max(0.0, dot_old_new / dot_new)
        
        p_new = -grad_new + beta * p
        p = p_new
        
        # Adapt step size
        if count == 0:
            alpha = min(alpha * 1.5, 1.0)
        else:
            alpha = max(alpha * 0.7, 1e-10)
        
        # Reset direction if beta is small
        if beta < 1e-4:
            p = -grad_new
    
    return X, energy

def _lbfgs_optimize(X0, time_limit=20.0, maxiter=1000):
    """Polish using L-BFGS-B in ambient space with projection."""
    def objective(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        e, _ = _energy_and_grad(X)
        return e
    
    def gradient(x_flat):
        X = x_flat.reshape(N, 3)
        X = _project_to_sphere(X)
        e, g = _energy_and_grad(X)
        return g.flatten()
    
    result = minimize(
        objective,
        X0.flatten(),
        jac=gradient,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13, 'm': 25}
    )
    
    X_opt = _project_to_sphere(result.x.reshape(N, 3))
    e_opt, _ = _energy_and_grad(X_opt)
    
    return X_opt, e_opt

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (5s buffer before 150s kill)
    
    best_energy = np.inf
    best_points = None
    
    # Phase 1: Halton initialization with penalty-based Riemannian CG
    # Start with Halton sequence (low-discrepancy for good coverage)
    X_halton = _halton_init(N)
    
    # Use penalty to enforce minimum separation during optimization
    # This helps avoid clustering artifacts
    penalty_schedule = [
        (50.0, 1e-5),   # (weight, min_dist)
        (20.0, 5e-6),
        (5.0, 1e-5),
        (1.0, 2e-5),
        (0.1, 5e-5),
        (0.0, 1e-6),    # No penalty for final polish
    ]
    
    X_current = X_halton.copy()
    
    for pw, md in penalty_schedule:
        if time.time() > deadline - 30.0:
            break
        
        # Determine time budget for this penalty stage
        remaining_stages = len(penalty_schedule) - penalty_schedule.index((pw, md))
        time_budget = min(15.0, (deadline - 30.0 - time.time()) / max(1, remaining_stages))
        
        try:
            X_current, e_current = _riemannian_cg(
                X_current, 
                max_iter=300, 
                tol=1e-9, 
                time_limit=time_budget,
                penalty_weight=pw,
                min_dist=md
            )
            
            if e_current < best_energy:
                best_energy = e_current
                best_points = X_current.copy()
                
        except Exception:
            pass
    
    # Phase 2: Try Fibonacci initialization as well
    if time.time() < deadline - 25.0:
        X_fib = _fibonacci_init(N)
        
        for pw, md in penalty_schedule[:3]:  # Fewer stages for second init
            if time.time() > deadline - 20.0:
                break
            
            time_budget = min(10.0, (deadline - 20.0 - time.time()) / 3)
            
            try:
                X_fib_opt, e_fib = _riemannian_cg(
                    X_fib,
                    max_iter=200,
                    tol=1e-9,
                    time_limit=time_budget,
                    penalty_weight=pw,
                    min_dist=md
                )
                
                if e_fib < best_energy:
                    best_energy = e_fib
                    best_points = X_fib_opt.copy()
                    
            except Exception:
                pass
    
    # Phase 3: L-BFGS-B polish on the best result
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            X_polished, e_polished = _lbfgs_optimize(
                best_points,
                time_limit=15.0,
                maxiter=2000
            )
            
            if e_polished < best_energy:
                best_energy = e_polished
                best_points = X_polished
                
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish if we have time
    if best_points is not None and time.time() < deadline - 5.0:
        try:
            def objective(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, _ = _energy_and_grad(X)
                return e
            
            def gradient(x_flat):
                X = x_flat.reshape(N, 3)
                X = _project_to_sphere(X)
                e, g = _energy_and_grad(X)
                return g.flatten()
                
            result = minimize(
                objective,
                best_points.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14, 'm': 30}
            )
            
            X_final = _project_to_sphere(result.x.reshape(N, 3))
            e_final, _ = _energy_and_grad(X_final)
            
            if e_final < best_energy:
                best_energy = e_final
                best_points = X_final
                
        except Exception:
            pass
    
    # If no valid result, fall back to Halton
    if best_points is None:
        best_points = _halton_init(N)
    
    # Ensure all points are on the unit sphere and valid
    best_points = _project_to_sphere(best_points)
    
    # Check for points too close together
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.sqrt(dist2[iu]).min()
    
    if min_dist < 1e-9:
        # Fall back to Halton initialization
        return _halton_init(N)
    
    return best_points
# EVOLVE-BLOCK-END
