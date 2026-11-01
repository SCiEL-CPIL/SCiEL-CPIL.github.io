"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient with respect to x_k (before projection):
      grad_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    
    For Riemannian optimization on the sphere, we need the projected gradient:
      grad_proj_k = grad_k - (grad_k . x_k) * x_k
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    
    # Compute pairwise distances
    # D[i,j] = |x_i - x_j|
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-12:
                d = 1e-12
            D[i, j] = d
            D[j, i] = d
    
    # Compute energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            energy += -np.log(D[i, j])
    
    # Compute gradient
    grad = np.zeros_like(X)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = D[i, j]
            if d < 1e-12:
                continue
            # grad contribution from pair (i,j): (x_i - x_j) / d^2
            grad[i] += (X[i] - X[j]) / (d * d)
    
    # Project gradient to tangent space of sphere
    for i in range(n):
        grad[i] -= np.dot(grad[i], X[i]) * X[i]
    
    return energy, grad


def _energy_fsum(X):
    """Compute energy using math.fsum for high precision."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    
    # Compute all pairwise distances
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(X[i] - X[j])
            D[i, j] = d
            D[j, i] = d
    
    # Sum log(1/d) for all pairs i<j
    terms = []
    for i in range(n):
        for j in range(i+1, n):
            d = D[i, j]
            if d < 1e-15:
                d = 1e-15
            terms.append(-np.log(d))
    
    return math.fsum(terms)


def _project_to_sphere(X):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1e-15
    return X / norms


def _riemannian_cg(X0, max_iter=500, tol=1e-10, deadline=None):
    """Riemannian conjugate gradient method for optimization on the sphere."""
    X = _project_to_sphere(X0.copy())
    n = X.shape[0]
    
    # Initial gradient
    f, g = _energy_and_grad(X)
    g_norm = np.linalg.norm(g)
    
    if g_norm < tol:
        return X
    
    # Initial search direction
    d = -g.copy()
    d -= np.sum(d * X, axis=1, keepdims=True) * X  # project to tangent
    
    alpha = 1.0
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Line search: find alpha that minimizes f along the geodesic
        # Approximate with straight line in ambient space, then project
        best_alpha = alpha
        best_f = f
        best_X = X.copy()
        
        # Try a few step sizes
        for a in [alpha, alpha*0.5, alpha*0.25, alpha*2.0, alpha*4.0]:
            X_trial = X + a * d
            X_trial = _project_to_sphere(X_trial)
            
            # Check validity
            if not np.all(np.isfinite(X_trial)):
                continue
            # Check minimum distance
            min_d = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    dd = np.linalg.norm(X_trial[i] - X_trial[j])
                    if dd < min_d:
                        min_d = dd
            if min_d < 1e-9:
                continue
            
            f_trial = _energy_fsum(X_trial)
            if f_trial < best_f:
                best_f = f_trial
                best_alpha = a
                best_X = X_trial.copy()
        
        if best_f >= f:
            # No improvement, reduce step size
            alpha *= 0.5
            if alpha < 1e-12:
                break
            continue
        
        # Update
        X = best_X
        f = best_f
        
        # New gradient
        f_new, g_new = _energy_and_grad(X)
        g_new_norm = np.linalg.norm(g_new)
        
        if g_new_norm < tol:
            break
        
        # Polak-Ribiere beta
        beta = np.dot(g_new, g_new - g) / (np.dot(g, g) + 1e-15)
        beta = max(0, beta)
        
        d = -g_new + beta * d
        d -= np.sum(d * X, axis=1, keepdims=True) * X  # project to tangent
        
        # Normalize direction
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            d = -g_new
            d -= np.sum(d * X, axis=1, keepdims=True) * X
            d_norm = np.linalg.norm(d)
        d = d / d_norm
        
        alpha = best_alpha
        g = g_new
    
    return X


def _lbfgs_polish(X0, max_iter=1000, deadline=None):
    """L-BFGS-B polish with penalty for sphere constraint."""
    X = _project_to_sphere(X0.copy())
    n = X.shape[0]
    
    def objective_and_grad(x_flat):
        X = x_flat.reshape(n, 3)
        X = _project_to_sphere(X)
        f, g = _energy_and_grad(X)
        # Gradient in flat coordinates (approximate)
        g_flat = g.flatten()
        return f, g_flat
    
    result = minimize(
        lambda x: _energy_fsum(_project_to_sphere(x.reshape(n, 3))),
        X.flatten(),
        jac=lambda x: _energy_and_grad(_project_to_sphere(x.reshape(n, 3)))[1].flatten(),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-12}
    )
    
    X_opt = _project_to_sphere(result.x.reshape(n, 3))
    
    # Check validity
    if not np.all(np.isfinite(X_opt)):
        return X
    
    min_d = np.inf
    for i in range(n):
        for j in range(i+1, n):
            dd = np.linalg.norm(X_opt[i] - X_opt[j])
            if dd < min_d:
                min_d = dd
    if min_d < 1e-9:
        return X
    
    return X_opt


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=None):
    """Generate random points on the sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    X = rng.randn(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def _icosahedral_seed():
    """Generate an icosahedral-like symmetric seed."""
    # Start with a regular icosahedron and extend
    t = (1 + np.sqrt(5)) / 2
    points = []
    # Icosahedron vertices
    ico = [
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
        [1, t, 0], [-1, -t, 0], [1, -t, 0], [-1, t, 0]
    ]
    # Normalize
    for p in ico:
        p = np.array(p, dtype=np.float64)
        p = p / np.linalg.norm(p)
        points.append(p)
    
    # Add more points by subdividing and pushing to sphere
    # For simplicity, use a dense Fibonacci spiral as base
    fib = _fibonacci_spiral()
    return fib


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit, leave margin
    
    # Multiple diverse initializations
    seeds = []
    seeds.append(_fibonacci_spiral())
    seeds.append(_random_points(seed=42))
    seeds.append(_random_points(seed=123))
    seeds.append(_random_points(seed=999))
    
    best_X = None
    best_energy = np.inf
    
    for seed_idx, X0 in enumerate(seeds):
        if time.time() > deadline - 10.0:
            break
        
        # Riemannian CG refinement
        X_rcg = _riemannian_cg(X0, max_iter=300, deadline=deadline - 5.0)
        
        # L-BFGS polish
        X_polished = _lbfgs_polish(X_rcg, max_iter=500, deadline=deadline - 2.0)
        
        # Compute final energy
        energy = _energy_fsum(X_polished)
        
        if energy < best_energy:
            best_energy = energy
            best_X = X_polished.copy()
    
    # If we still have time, do another round of refinement on the best
    if best_X is not None and time.time() < deadline - 5.0:
        # Fine-tune with more iterations
        X_final = _lbfgs_polish(best_X, max_iter=1000, deadline=deadline - 1.0)
        energy_final = _energy_fsum(X_final)
        if energy_final < best_energy:
            best_X = X_final
            best_energy = energy_final
    
    # Final safety check
    if best_X is None:
        best_X = _fibonacci_spiral()
    
    # Ensure all points are on the unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Verify minimum distance
    min_d = np.inf
    for i in range(N):
        for j in range(i+1, N):
            dd = np.linalg.norm(best_X[i] - best_X[j])
            if dd < min_d:
                min_d = dd
    
    if min_d < 1e-9:
        # Fallback to Fibonacci
        best_X = _fibonacci_spiral()
    
    return best_X

# EVOLVE-BLOCK-END
