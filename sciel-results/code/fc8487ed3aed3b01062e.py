"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy_and_grad(X):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    and its gradient w.r.t. X (shape (N,3)).
    
    |x_i - x_j|^2 = 2 - 2*x_i.x_j
    d/dx_i log(1/|x_i - x_j|) = d/dx_i (-log|x_i - x_j|) = (x_i - x_j)/|x_i - x_j|^2
    So grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    # X shape (N, 3)
    # Compute pairwise dot products
    dots = X @ X.T  # (N, N), dots[i,j] = x_i . x_j
    # |x_i - x_j|^2 = 2 - 2*dots[i,j]
    d2 = 2.0 - 2.0 * dots
    # Set diagonal to 1 to avoid division by zero (we'll zero out diagonal in grad)
    np.fill_diagonal(d2, 1.0)
    
    # Energy: sum_{i<j} -log(sqrt(d2[i,j])) = -0.5 * sum_{i<j} log(d2[i,j])
    # Use upper triangle
    iu = np.triu_indices(N, 1)
    log_d2 = np.log(d2[iu])
    E = -0.5 * math.fsum(log_d2.tolist())
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / d2[i,j]
    # (x_i - x_j)/d2[i,j] = x_i/d2[i,j] - x_j/d2[i,j]
    # grad = X * sum_j (1/d2[i,j]) - sum_j (x_j / d2[i,j])
    inv_d2 = 1.0 / d2  # (N, N)
    # Zero out diagonal
    np.fill_diagonal(inv_d2, 0.0)
    
    # row_sums[i] = sum_j inv_d2[i,j]
    row_sums = inv_d2.sum(axis=1)  # (N,)
    
    # grad = X * row_sums[:, None] - (X.T @ inv_d2)
    # X.T @ inv_d2 has shape (3, N), so we need to transpose back
    term2 = inv_d2 @ X  # (N, 3), term2[i] = sum_j inv_d2[i,j] * x_j
    grad = X * row_sums[:, None] - term2
    
    return E, grad

def _retract(X, step):
    """
    Riemannian retraction: move along tangent direction, then project back to sphere.
    X: (N, 3) on sphere
    step: (N, 3) tangent vector (step[i] . X[i] = 0 ideally)
    """
    # First, project step to tangent space at X
    # tangent_step = step - (step . X) X
    step_dot_X = np.sum(step * X, axis=1, keepdims=True)  # (N, 1)
    tangent_step = step - step_dot_X * X
    
    # Move
    X_new = X + tangent_step
    
    # Normalize back to unit sphere
    norms = np.linalg.norm(X_new, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    X_new = X_new / norms
    
    return X_new

def _optimize_lbfgs(X0, max_iter=5000, tol=1e-12, deadline=None):
    """
    L-BFGS optimization on the sphere using retraction.
    """
    X = X0.copy()
    n = X.shape[0]
    
    # L-BFGS memory
    m = 30
    s_list = []  # list of step vectors (flattened)
    y_list = []  # list of gradient difference vectors (flattened)
    
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    start_time = time.time()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Check convergence
        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break
        
        # L-BFGS two-loop recursion
        if len(s_list) == 0:
            # Steepest descent
            direction = -g
        else:
            # Two-loop recursion
            q = g_flat.copy()
            alphas = []
            rhos = []
            
            for i in range(len(s_list) - 1, -1, -1):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = 1.0 / np.dot(s_i, y_i)
                rhos.append(rho_i)
                alpha_i = rho_i * np.dot(s_i, q)
                alphas.append(alpha_i)
                q = q - alpha_i * y_i
            
            # Initial Hessian approximation (identity)
            r = q
            
            for i in range(len(s_list)):
                s_i = s_list[i]
                y_i = y_list[i]
                rho_i = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta_i = rho_i * np.dot(y_i, r)
                r = r + (alpha_i - beta_i) * s_i
            
            direction = r.reshape(n, 3)
        
        # Line search: backtracking
        step_size = 1.0
        g_flat_dir = np.dot(g_flat, direction.ravel())
        
        X_trial = X
        E_trial = E
        g_trial = g
        
        max_ls = 50
        for ls in range(max_ls):
            X_trial = _retract(X, step_size * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
            
            if E_trial <= E + 1e-4 * step_size * g_flat_dir:
                break
            step_size *= 0.5
        
        if step_size < 1e-15:
            # Can't take a step
            break
        
        s_vec = (X_trial - X).ravel()
        y_vec = (g_trial - g).ravel()
        
        if np.dot(s_vec, y_vec) > 1e-10:
            s_list.append(s_vec)
            y_list.append(y_vec)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        X = X_trial
        E = E_trial
        g = g_trial
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    return best_X, best_E

def _polish_steepest_descent(X0, max_iter=20000, tol=1e-14, deadline=None):
    """
    High-precision steepest descent polish with adaptive step size.
    """
    X = X0.copy()
    n = X.shape[0]
    E, g = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    start_time = time.time()
    
    step_size = 1e-2
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        g_flat = g.ravel()
        g_norm = np.linalg.norm(g_flat)
        if g_norm < tol:
            break
        
        direction = -g / g_norm  # normalized steepest descent
        
        # Backtracking line search
        E_new = None
        trial_step = step_size
        for ls in range(40):
            X_trial = _retract(X, trial_step * direction)
            E_trial, g_trial = _energy_and_grad(X_trial)
            
            if E_trial < E:
                E_new = E_trial
                g_new = g_trial
                X_trial_final = X_trial
                step_size = trial_step
                break
            trial_step *= 0.5
        
        if E_new is None:
            # Can't improve, try smaller step or break
            step_size *= 0.5
            if step_size < 1e-20:
                break
            continue
        
        X = X_trial_final
        E = E_new
        g = g_new
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
        
        # Adapt step size upward if we made good progress
        if E_new < E - 1e-14:
            step_size = min(step_size * 1.3, 1e-1)
    
    return best_X, best_E

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate n random points on the sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms

def construct():
    """
    Construct 100 points on the unit sphere to minimize logarithmic energy.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Multi-start: try Fibonacci spiral and a couple of random starts
    starts = [_fibonacci_sphere(N)]
    starts.append(_random_sphere(N, seed=42))
    starts.append(_random_sphere(N, seed=123))
    
    best_X = None
    best_E = np.inf
    
    for X0 in starts:
        if time.time() > deadline - 30.0:
            break
        
        # Optimize with L-BFGS
        X, E = _optimize_lbfgs(X0, max_iter=3000, tol=1e-12, deadline=deadline - 10.0)
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
    
    # Check time
    if time.time() > deadline - 10.0:
        X = best_X
        norms = np.linalg.norm(X, axis=1)
        return X / norms[:, None]
    
    # Polish with steepest descent
    X, E = _polish_steepest_descent(best_X, max_iter=20000, tol=1e-14, deadline=deadline)
    
    # Final check: ensure all points are on the unit sphere and valid
    norms = np.linalg.norm(X, axis=1)
    X = X / norms[:, None]
    
    # Check for very close points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(X[i] - X[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # This shouldn't happen, but if it does, return the Fibonacci start
        return _fibonacci_sphere(N)
    
    return X
# EVOLVE-BLOCK-END
