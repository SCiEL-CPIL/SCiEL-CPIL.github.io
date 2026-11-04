"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 150

# EVOLVE-BLOCK-START
def _energy_grad(X):
    """Compute logarithmic energy and its gradient.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient wrt x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    
    # Mask off-diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum over i<j of log(1/dist) = -sum_{i<j} log(dist)
    # Use upper triangle to avoid double counting
    triu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[triu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so we want sum_j (x_i - x_j)/|x_i - x_j|^2
    # = sum_j diff[i,j] / dist[i,j]^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def _riemannian_gradient_descent(X0, max_iter=5000, tol=1e-12, time_limit=100.0):
    """Riemannian gradient descent with Armijo line search on the sphere."""
    start_time = time.time()
    X = X0.copy()
    X = _project_to_sphere(X)
    
    best_energy, best_X = None, X.copy()
    
    # Compute initial energy
    E, G = _energy_grad(X)
    if best_energy is None or E < best_energy:
        best_energy = E
        best_X = X.copy()
    
    # Riemannian gradient is the tangential component of Euclidean gradient
    # proj_tangent(v) = v - (v·x)x for unit x
    def rgrad(X):
        E, G = _energy_grad(X)
        # Project gradient onto tangent space
        G_tan = G - np.sum(G * X, axis=1, keepdims=True) * X
        return E, G_tan
    
    E, Gt = rgrad(X)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Check convergence
        if np.linalg.norm(Gt) < tol:
            break
        
        # Direction: negative Riemannian gradient
        direction = -Gt
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-15:
            break
        direction = direction / dir_norm
        
        # Armijo line search
        # Riemannian step: X_new = normalize(X + alpha * direction)
        # For small alpha, the Riemannian gradient inner product with direction is -||Gt||^2/||Gt|| = -||Gt||
        c1 = 1e-4
        alpha = 1.0
        max_alpha = 10.0
        min_alpha = 1e-10
        
        # Initial alpha based on gradient magnitude
        alpha = min(0.1 / max(np.linalg.norm(Gt), 1e-8), 1.0)
        
        improved = False
        for ls_iter in range(50):
            X_trial = _project_to_sphere(X + alpha * direction)
            E_trial, Gt_trial = rgrad(X_trial)
            
            if E_trial < E - c1 * alpha * np.linalg.norm(Gt) ** 2:
                improved = True
                break
            alpha *= 0.5
            if alpha < min_alpha:
                break
        
        if not improved:
            # Try smaller step or break
            alpha = min_alpha
            X_trial = _project_to_sphere(X + alpha * direction)
            E_trial, Gt_trial = rgrad(X_trial)
            if E_trial >= E:
                break
        
        X = X_trial
        E, Gt = rgrad(X)
        
        if E < best_energy:
            best_energy = E
            best_X = X.copy()
    
    return best_X, best_energy


def _random_restart(X0, n_restarts=5, time_limit=100.0):
    """Multiple random restarts with Riemannian GD."""
    start_time = time.time()
    best_X, best_E = X0.copy(), None
    
    # Evaluate initial
    E0, _ = _energy_grad(best_X)
    best_E = E0
    
    for r in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Create perturbed starting point
        noise = np.random.randn(N, 3) * 0.1
        X_pert = _project_to_sphere(X0 + noise)
        
        X_opt, E_opt = _riemannian_gradient_descent(
            X_pert, max_iter=3000, tol=1e-12,
            time_limit=time_limit - (time.time() - start_time)
        )
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
    
    return best_X, best_E


def _polish(X, time_limit=30.0):
    """High-precision polish using scipy minimize on tangent space."""
    start_time = time.time()
    X = X0 = X.copy()
    X = _project_to_sphere(X)
    
    # Use scipy's L-BFGS-B on the full 3N variables with sphere constraint handled by projection
    # Actually, let's do a few more iterations of Riemannian GD with tighter tolerance
    X_opt, E_opt = _riemannian_gradient_descent(
        X, max_iter=10000, tol=1e-14,
        time_limit=time_limit
    )
    
    return X_opt, E_opt


def construct():
    np.random.seed(42)
    deadline = time.time() + 140.0  # safety margin
    
    # Step 1: Generate initial points (Fibonacci spiral)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Step 2: Riemannian gradient descent from this start
    X1, E1 = _riemannian_gradient_descent(X0, max_iter=5000, tol=1e-12, time_limit=60.0)
    
    # Step 3: Random restarts
    remaining = deadline - time.time()
    if remaining > 20:
        X2, E2 = _random_restart(X1, n_restarts=3, time_limit=min(remaining - 10, 50.0))
        if E2 < E1:
            X1, E1 = X2, E2
    
    # Step 4: Final polish
    remaining = deadline - time.time()
    if remaining > 15:
        X3, E3 = _polish(X1, time_limit=min(remaining - 5, 30.0))
        if E3 < E1:
            X1, E1 = X3, E3
    
    # Final validation
    X1 = _project_to_sphere(X1)
    
    # Ensure no points are too close
    for _ in range(100):
        diff = X1[:, np.newaxis, :] - X1[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        min_dist = dist.min()
        if min_dist > 1e-9:
            break
        # Nudge the closest pair apart
        i_idx, j_idx = np.unravel_index(np.argmin(dist), dist.shape)
        # Move point i slightly away from point j
        direction = X1[i_idx] - X1[j_idx]
        if np.linalg.norm(direction) < 1e-12:
            direction = np.random.randn(3)
        X1[i_idx] = _project_to_sphere(X1[i_idx] + 1e-7 * direction)
    
    return X1
# EVOLVE-BLOCK-END
