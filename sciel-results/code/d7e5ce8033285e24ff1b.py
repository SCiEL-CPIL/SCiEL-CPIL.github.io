"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 150

# EVOLVE-BLOCK-START
import math

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient w.r.t. x_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2  (projected to tangent)
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist2, 1.0)  # avoid log(0)
    dist = np.sqrt(dist2)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point k, grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # But we need the full gradient, then project to tangent
    # grad_k = sum_{j != k} (x_k - x_j) / dist2[k,j]
    inv_dist2 = 1.0 / dist2  # (n, n)
    np.fill_diagonal(inv_dist2, 0.0)
    # grad_k = sum_j (x_k - x_j) * inv_dist2[k,j]
    # = sum_j x_k * inv_dist2[k,j] - sum_j x_j * inv_dist2[k,j]
    # = x_k * sum_j inv_dist2[k,j] - sum_j x_j * inv_dist2[k,j]
    row_sum = np.sum(inv_dist2, axis=1)  # (n,)
    # sum_j x_j * inv_dist2[k,j] = inv_dist2 @ X  (n, 3)
    weighted_sum = inv_dist2 @ X  # (n, 3)
    grad = row_sum[:, np.newaxis] * X - weighted_sum  # (n, 3)
    
    # Project to tangent plane: grad_t = grad - (grad . x_k) x_k
    dot = np.sum(grad * X, axis=1)  # (n,)
    grad_t = grad - dot[:, np.newaxis] * X
    
    return energy, grad_t

def _project_to_sphere(X):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _lbfgs(X0, max_iter=5000, tol=1e-12, m=20):
    """Simple L-BFGS on the sphere using retraction (projection)."""
    X = _project_to_sphere(np.array(X0, dtype=np.float64))
    n = X.shape[0]
    dim = n * 3
    
    # L-BFGS memory
    s_list = []
    y_list = []
    
    # Initial gradient
    f, g = _energy_and_grad(X)
    
    # L-BFGS two-loop recursion
    def lbgfs_direction(g, s_list, y_list, m):
        q = g.copy()
        alphas = []
        for i in range(len(s_list) - 1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]
            gamma = np.dot(y_i, y_i)
            if gamma < 1e-30:
                alphas.append(0.0)
                continue
            alpha = np.dot(s_i, q) / gamma
            alphas.append(alpha)
            q = q - alpha * y_i
        # Approximate Hessian inverse: H0 = gamma * I where gamma = <s,y>/<y,y> for last pair
        if len(s_list) > 0:
            gamma = np.dot(s_list[-1], y_list[-1]) / max(np.dot(y_list[-1], y_list[-1]), 1e-30)
        else:
            gamma = 1.0
        r = gamma * q
        for i in range(len(s_list)):
            s_i = s_list[i]
            y_i = y_list[i]
            gamma = np.dot(y_i, y_i)
            if gamma < 1e-30:
                beta = 0.0
            else:
                beta = np.dot(y_i, r) / gamma
            rho = 1.0 / gamma
            r = r + (alphas[i] - rho * beta) * s_i
        return r
    
    # Line search (Wolfe conditions)
    def line_search(X, g, d, f, c1=1e-4, c2=0.9, max_ls=30):
        # Ensure d is a descent direction
        if np.dot(g, d) > 0:
            d = -g
        step = 1.0
        g_dot_d = np.dot(g, d)
        for _ in range(max_ls):
            X_new = _project_to_sphere(X + step * d)
            f_new, g_new = _energy_and_grad(X_new)
            # Wolfe conditions
            if f_new <= f + c1 * step * g_dot_d and np.dot(g_new, d) <= c2 * g_dot_d:
                return step, X_new, f_new, g_new
            step *= 0.5
        return step, X_new, f_new, g_new
    
    for it in range(max_iter):
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Compute L-BFGS direction
        d = -lbgfs_direction(g, s_list, y_list, m)
        
        # Project direction to tangent space
        dot_d = np.sum(d * X, axis=1)
        d = d - dot_d[:, np.newaxis] * X
        
        # Line search
        step, X_new, f_new, g_new = line_search(X, g, d, f)
        
        # Update L-BFGS memory
        s = X_new - X
        y = g_new - g
        sy = np.dot(s, y)
        if sy > 1e-16:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        X = X_new
        f = f_new
        g = g_new
    
    return X, f

def construct():
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_X = X0.copy()
    best_E, _ = _energy_and_grad(best_X)
    
    # Run L-BFGS
    X_opt, E_opt = _lbfgs(X0, max_iter=3000, tol=1e-13)
    
    if E_opt < best_E:
        best_X = X_opt
        best_E = E_opt
    
    elapsed = time.time() - t_start
    remaining = deadline - time.time()
    
    # If we have time, try additional restarts with perturbed initial conditions
    if remaining > 30:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb the best solution
            noise = rng.randn(N, 3) * 0.1
            X_pert = _project_to_sphere(best_X + noise)
            X_opt2, E_opt2 = _lbfgs(X_pert, max_iter=2000, tol=1e-13)
            if E_opt2 < best_E:
                best_X = X_opt2
                best_E = E_opt2
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        X_final, E_final = _lbfgs(best_X, max_iter=5000, tol=1e-14)
        if E_final < best_E:
            best_X = X_final
            best_E = E_final
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Separate them
                best_X[j] = best_X[i] + 1e-8 * np.array([1, 0, 0])
                best_X[j] = best_X[j] / np.linalg.norm(best_X[j])
    
    return best_X
# EVOLVE-BLOCK-END
