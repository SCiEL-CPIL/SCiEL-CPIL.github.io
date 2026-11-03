"""Rewritten program: deterministic gradient-based local minimization on the sphere
with multiple restarts, Sobol initializations, gradient flow annealing, and a deadline guard."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
DEADLINE_SECONDS = 145.0  # safety margin under 150s wall-clock


def _energy_and_grad(X):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i-x_j|) and its gradient w.r.t. X.
    X is (N,3) on the unit sphere. Returns (energy, grad) where grad is the
    Euclidean gradient (not yet projected)."""
    # Pairwise distances
    diff = X[:, None, :] - X[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)   # (N, N)
    # Avoid log(0) and division by zero
    np.fill_diagonal(dist, 1.0)
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    e = -np.sum(np.log(d_ij))
    
    # Gradient: d/dx_i [-log|x_i - x_j|] = (x_i - x_j)/|x_i - x_j|^2
    # For each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_i - x_j) / dist[i,j]^2
    # Note: for j != i, (x_i - x_j)/dist^2
    inv_dist2 = np.zeros_like(dist)
    mask = ~np.eye(N, dtype=bool)
    inv_dist2[mask] = 1.0 / (dist[mask] ** 2)
    
    # grad[i] = sum_j (X[i] - X[j]) * inv_dist2[i,j]
    # = X[i] * sum_j inv_dist2[i,j] - sum_j X[j] * inv_dist2[i,j]
    row_sum = np.sum(inv_dist2, axis=1)  # (N,)
    # sum_j X[j] * inv_dist2[i,j] = inv_dist2 @ X
    weighted_sum = inv_dist2 @ X  # (N, 3)
    grad = X * row_sum[:, None] - weighted_sum
    
    return e, grad


def _project_to_sphere(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral with offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sobol_sphere(n, seed=0):
    """Generate n points on the unit sphere using a Sobol quasi-random sequence."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    # Use enough points and take the last n for better uniformity
    m = max(n, 128)
    # 2^11 = 2048 points, take last n
    points = sampler.random(2048)
    # Normalize to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    points = points / norms
    # Take last n points (better distributed than first n for low-discrepancy)
    return points[-n:]


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms


def _gradient_flow(X, n_steps=500, lr=0.01, cooling=True, deadline=None):
    """Run gradient flow on the sphere with optional annealing.
    
    Args:
        X: (N,3) array of points on the unit sphere
        n_steps: number of gradient flow steps
        lr: learning rate
        cooling: if True, use annealing schedule for learning rate
        deadline: absolute time limit to stop early
    
    Returns:
        (X, energy) - optimized points and their energy
    """
    X = X.copy()
    current_lr = lr
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient
        _, grad = _energy_and_grad(X)
        
        # Project gradient to tangent space: g_tan = g - (g . x) * x
        grad_dot = np.sum(grad * X, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot * X
        
        # Take gradient step
        if cooling:
            # Linear cooling: lr decreases linearly
            current_lr = lr * (1.0 - step / n_steps) + lr * 0.1 * (step / n_steps)
        
        X = X - current_lr * grad_tangent
        
        # Project back to sphere
        X = _project_to_sphere(X)
    
    e, _ = _energy_and_grad(X)
    return X, e


def _objective(X_flat):
    """Objective for scipy minimize: energy of points reshaped from flat vector."""
    X = X_flat.reshape(N, 3)
    X = _project_to_sphere(X)
    e, _ = _energy_and_grad(X)
    return e


def _objective_grad(X_flat):
    """Objective and gradient for scipy minimize."""
    X = X_flat.reshape(N, 3)
    X = _project_to_sphere(X)
    e, grad = _energy_and_grad(X)
    return e, grad


def construct():
    start_time = time.time()
    
    best_X = None
    best_energy = float('inf')
    
    # Generate multiple initial configurations
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral (known good starting point)
    initial_configs = [_fibonacci_sphere(N)]
    
    # Add random perturbations of Fibonacci
    for _ in range(3):
        X0 = _fibonacci_sphere(N)
        noise = rng.normal(scale=0.1, size=X0.shape)
        X0 = _project_to_sphere(X0 + noise)
        initial_configs.append(X0)
    
    # Add fully random configurations
    for _ in range(3):
        initial_configs.append(_random_sphere(N, rng))
    
    # Add scaled Fibonacci with different offsets
    for offset in [0.1, 0.25, 0.4]:
        initial_configs.append(_fibonacci_sphere(N, seed_offset=offset))
    
    # Add Sobol sequence initializations for better uniform coverage
    for sobol_seed in [0, 1, 2]:
        initial_configs.append(_sobol_sphere(N, seed=sobol_seed))
    
    # Allocate time: use ~60% for initial optimization, ~25% for gradient flow, ~15% for final polish
    deadline_init = start_time + DEADLINE_SECONDS * 0.60
    deadline_flow = start_time + DEADLINE_SECONDS * 0.85
    
    # Phase 1: L-BFGS optimization from each initial config
    for X0 in initial_configs:
        if time.time() > deadline_init:
            break
        
        x0_flat = X0.flatten()
        
        try:
            res = minimize(
                _objective_grad,
                x0_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 5000,
                }
            )
        except Exception:
            continue
        
        X_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt, _ = _energy_and_grad(X_opt)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt.copy()
    
    # Phase 2: Gradient flow with annealing on best solution to escape local minima
    if best_X is not None and time.time() < deadline_flow:
        try:
            # Run gradient flow with annealing
            X_flow, e_flow = _gradient_flow(
                best_X, 
                n_steps=300, 
                lr=0.005, 
                cooling=True,
                deadline=deadline_flow
            )
            
            if e_flow < best_energy:
                best_energy = e_flow
                best_X = X_flow.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if best_X is not None and time.time() < DEADLINE_SECONDS - 3:
        try:
            x0_flat = best_X.flatten()
            res = minimize(
                _objective_grad,
                x0_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 8000,
                    'ftol': 1e-16,
                    'gtol': 1e-15,
                    'maxfun': 15000,
                }
            )
            X_polish = _project_to_sphere(res.x.reshape(N, 3))
            e_polish, _ = _energy_and_grad(X_polish)
            if e_polish < best_energy:
                best_energy = e_polish
                best_X = X_polish.copy()
        except Exception:
            pass
    
    if best_X is None:
        # Fallback: return Fibonacci spiral
        best_X = _fibonacci_sphere(N)
    
    return best_X
