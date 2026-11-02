"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time


def _log_energy_and_grad(X):
    """Compute log energy and gradient (Euclidean) for points on sphere."""
    n = X.shape[0]
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    # Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} diff[i,j] / dist[i,j]^2
    # But diff[i,j] = x_i - x_j, so grad_i = sum_j (x_i - x_j)/dist_ij^2
    # Note: dist^2 = sum(diff^2), so grad_i = sum_j diff[i,j] / dist[i,j]^2
    grad = np.zeros_like(X)
    # For each i, sum over j != i of (x_i - x_j)/dist_ij^2
    # diff[i,j] = x_i - x_j
    # We need sum_j diff[i,j] / dist[i,j]^2 for each i
    # inv_dist_sq = 1/dist^2, with diagonal zeroed
    inv_dist_sq = np.zeros_like(dist)
    mask = dist > 1e-12
    inv_dist_sq[mask] = 1.0 / (dist[mask] * dist[mask])
    np.fill_diagonal(inv_dist_sq, 0.0)
    # grad[i] = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_tangent(X, G):
    """Project Euclidean gradient G to tangent space at each point X."""
    # G_tangent = G - (G·X) X
    dot = np.sum(G * X, axis=1)  # (n,)
    return G - dot[:, np.newaxis] * X


def _retract(X, V):
    """Exponential map on unit sphere: retraction of tangent vector V at point X."""
    # For each point, compute exp_X(V)
    # If |V| is very small, just X + V (normalized)
    # Otherwise: X*cos(|V|) + (V/|V|)*sin(|V|)
    norm_V = np.sqrt(np.sum(V * V, axis=1))  # (n,)
    result = np.zeros_like(X)
    small = norm_V < 1e-12
    if np.any(small):
        result[small] = X[small] + V[small]
        # Normalize
        norms = np.sqrt(np.sum(result[small] * result[small], axis=1))
        norms[norms < 1e-15] = 1.0
        result[small] = result[small] / norms[:, np.newaxis]
    if np.any(~small):
        Vs = V[~small]
        Xs = X[~small]
        ns = norm_V[~small]
        cos_n = np.cos(ns)
        sin_n = np.sin(ns)
        result[~small] = Xs * cos_n[:, np.newaxis] + Vs * (sin_n / ns)[:, np.newaxis]
        # Normalize to ensure on sphere
        norms = np.sqrt(np.sum(result[~small] * result[~small], axis=1))
        norms[norms < 1e-15] = 1.0
        result[~small] = result[~small] / norms[:, np.newaxis]
    return result


def _fibonacci_sphere():
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Normalize
    norms = np.sqrt(np.sum(X * X, axis=1))
    X = X / norms[:, np.newaxis]
    return X


def _riemannian_cg(X0, max_iter=5000, tol=1e-10, time_budget=130.0):
    """Riemannian conjugate gradient with backtracking line search."""
    start_time = time.time()
    X = X0.copy()
    
    # Compute initial gradient in tangent space
    energy, eucl_grad = _log_energy_and_grad(X)
    grad_t = _project_tangent(X, eucl_grad)
    
    # Initial search direction
    d = -grad_t.copy()
    # Normalize d in tangent space (it already is)
    
    best_energy = energy
    best_X = X.copy()
    
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Check convergence
        grad_norm = np.sqrt(np.sum(grad_t * grad_t))
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4  # Armijo condition
        # Compute initial energy
        E0 = energy
        
        for ls_iter in range(30):
            # Propose new point
            X_new = _retract(X, alpha * d)
            E_new, eucl_grad_new = _log_energy_and_grad(X_new)
            
            # Check validity
            if not np.all(np.isfinite(E_new)):
                alpha *= 0.5
                continue
            
            # Check minimum distance
            diff = X_new[:, np.newaxis, :] - X_new[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, 1.0)
            min_dist = np.min(dist)
            if min_dist < 1e-9:
                alpha *= 0.5
                continue
            
            # Armijo condition
            if E_new <= E0 + c1 * alpha * np.sum(grad_t * d):
                break
            alpha *= 0.5
        
        else:
            # Line search failed, try steepest descent
            break
        
        # Update
        X = X_new
        energy = E_new
        eucl_grad = eucl_grad_new
        grad_t = _project_tangent(X, eucl_grad)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Compute beta (Polak-Ribiere)
        grad_t_prev = _project_tangent(X, eucl_grad)  # This is the current grad_t actually
        # Wait, I need the previous grad_t
        # Let me restructure...
        
        # Actually, let me use Fletcher-Reeves for simplicity
        # beta = ||grad_t_new||^2 / ||grad_t_old||^2
        # But I need to store the old norm
        
        # Let me just do steepest descent with momentum-like approach
        # Actually, let's do proper CG
        # I'll restructure the loop
    
    return best_X, best_energy


def _riemannian_cg_v2(X0, max_iter=3000, tol=1e-10, time_budget=130.0):
    """Riemannian conjugate gradient with backtracking line search - v2."""
    start_time = time.time()
    X = X0.copy()
    
    # Compute initial gradient in tangent space
    energy, eucl_grad = _log_energy_and_grad(X)
    grad_t = _project_tangent(X, eucl_grad)
    grad_norm_sq = np.sum(grad_t * grad_t)
    
    # Initial search direction
    d = -grad_t.copy()
    
    best_energy = energy
    best_X = X.copy()
    
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Check convergence
        if grad_norm_sq < tol * tol:
            break
        
        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4  # Armijo condition
        E0 = energy
        grad_dot_d = np.sum(grad_t * d)
        
        success = False
        X_new = None
        E_new = None
        eucl_grad_new = None
        
        for ls_iter in range(40):
            # Propose new point
            X_trial = _retract(X, alpha * d)
            E_trial, eucl_grad_trial = _log_energy_and_grad(X_trial)
            
            # Check validity
            if not np.all(np.isfinite(E_trial)):
                alpha *= 0.5
                continue
            
            # Check minimum distance
            diff = X_trial[:, np.newaxis, :] - X_trial[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, 1.0)
            min_dist = np.min(dist)
            if min_dist < 1e-9:
                alpha *= 0.5
                continue
            
            # Armijo condition
            if E_trial <= E0 + c1 * alpha * grad_dot_d:
                success = True
                X_new = X_trial
                E_new = E_trial
                eucl_grad_new = eucl_grad_trial
                break
            alpha *= 0.5
        
        if not success:
            # Try smaller step in steepest descent direction
            alpha = 1e-4
            X_trial = _retract(X, alpha * (-grad_t))
            E_trial, eucl_grad_trial = _log_energy_and_grad(X_trial)
            if np.all(np.isfinite(E_trial)) and E_trial < E0:
                X = X_trial
                energy = E_trial
                eucl_grad = eucl_grad_trial
                grad_t = _project_tangent(X, eucl_grad)
                grad_norm_sq = np.sum(grad_t * grad_t)
                d = -grad_t
                if energy < best_energy:
                    best_energy = energy
                    best_X = X.copy()
                continue
            break
        
        # Update
        X = X_new
        energy = E_new
        eucl_grad = eucl_grad_new
        grad_t_new = _project_tangent(X, eucl_grad)
        grad_norm_sq_new = np.sum(grad_t_new * grad_t_new)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Fletcher-Reeves beta
        if grad_norm_sq > 1e-30:
            beta = grad_norm_sq_new / grad_norm_sq
        else:
            beta = 0.0
        
        # Reset direction if beta is too large or negative
        if beta > 10.0 or beta < 0.0:
            beta = 0.0
        
        d_new = -grad_t_new + beta * d
        # Project d_new to tangent space (it should already be, but ensure)
        d_new = _project_tangent(X, d_new)
        
        grad_t = grad_t_new
        grad_norm_sq = grad_norm_sq_new
        d = d_new
    
    return best_X, best_energy


def _polish(X, max_iter=2000, tol=1e-12, time_budget=15.0):
    """High-precision polish pass with steepest descent and very tight tolerance."""
    start_time = time.time()
    X = X.copy()
    
    energy, eucl_grad = _log_energy_and_grad(X)
    grad_t = _project_tangent(X, eucl_grad)
    
    best_energy = energy
    best_X = X.copy()
    
    alpha = 1e-3
    for it in range(max_iter):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        grad_norm = np.sqrt(np.sum(grad_t * grad_t))
        if grad_norm < tol:
            break
        
        # Backtracking line search
        E0 = energy
        grad_dot_d = -np.sum(grad_t * grad_t)  # d = -grad_t
        c1 = 1e-4
        
        success = False
        for ls_iter in range(30):
            X_trial = _retract(X, alpha * (-grad_t))
            E_trial, eucl_grad_trial = _log_energy_and_grad(X_trial)
            
            if not np.all(np.isfinite(E_trial)):
                alpha *= 0.5
                continue
            
            diff = X_trial[:, np.newaxis, :] - X_trial[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, 1.0)
            min_dist = np.min(dist)
            if min_dist < 1e-9:
                alpha *= 0.5
                continue
            
            if E_trial <= E0 + c1 * alpha * grad_dot_d:
                success = True
                X = X_trial
                energy = E_trial
                eucl_grad = eucl_grad_trial
                break
            alpha *= 0.5
        
        if not success:
            alpha *= 0.1
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        grad_t = _project_tangent(X, eucl_grad)
    
    return best_X, best_energy


def construct():
    start_time = time.time()
    
    # Start from Fibonacci spiral
    X = _fibonacci_sphere()
    
    # Run Riemannian CG
    X, energy = _riemannian_cg_v2(X, max_iter=3000, tol=1e-10, time_budget=120.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < 140.0:
        # Polish pass
        X, energy = _polish(X, max_iter=2000, tol=1e-12, time_budget=145.0)
    
    # Final normalization
    norms = np.sqrt(np.sum(X * X, axis=1))
    norms[norms < 1e-15] = 1.0
    X = X / norms[:, np.newaxis]
    
    return X
# EVOLVE-BLOCK-END
