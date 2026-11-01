import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_full(X):
    """Compute total log energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    mask = ~np.eye(N, dtype=bool)
    return -0.5 * np.sum(np.log(np.sqrt(dist_sq[mask])))

def _energy_point_k(X, k):
    """Energy contribution of point k: sum_{j != k} log(1/|x_k - x_j|)."""
    diff = X[k] - X
    dist_sq = np.sum(diff * diff, axis=1)
    dist_sq[k] = 1.0  # ignore self
    return -np.sum(np.log(np.sqrt(dist_sq)))

def _energy_point_k_and_grad(X, k):
    """
    Energy and gradient (in local 2D tangent coordinates) for point k.
    We parameterize x_k = base + u*t1 + v*t2, then normalize.
    Actually, simpler: use the fact that for a point on S^2, we can
    parameterize by two angles (theta, phi) or by a 2D vector in tangent space.
    
    Here we use: given current position x_k, build orthonormal tangent basis (t1, t2).
    A displacement (a, b) in tangent space gives new point:
      x_new = normalize(x_k + a*t1 + b*t2)
    
    The energy of point k given others fixed:
      E_k(a,b) = sum_{j != k} -log(|normalize(x_k + a*t1 + b*t2) - x_j|)
    
    Gradient wrt (a, b):
      dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2  (Euclidean gradient)
      Project to tangent: g_t = g_eucl - (g_eucl . x_k) * x_k
      Then dE/da = g_t . t1, dE/db = g_t . t2
    """
    xk = X[k]
    
    # Build tangent basis
    # Choose a reference vector not parallel to xk
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(xk, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(xk, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(xk, t1)
    t2 /= np.linalg.norm(t2)
    
    # Compute energy
    diff = xk - X
    dist_sq = np.sum(diff * diff, axis=1)
    dist_sq[k] = 1.0
    energy = -np.sum(np.log(np.sqrt(dist_sq)))
    
    # Compute Euclidean gradient: sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # diff[i] = x_k - x_i
    # grad_eucl = sum_i diff[i] / dist_sq[i]  (for i != k)
    inv_dist_sq = np.zeros(N)
    inv_dist_sq[:] = 1.0 / dist_sq
    inv_dist_sq[k] = 0.0
    grad_eucl = np.sum(diff * inv_dist_sq[:, np.newaxis], axis=0)
    
    # Project to tangent
    grad_tang = grad_eucl - np.dot(grad_eucl, xk) * xk
    
    # Gradient in (a, b) coordinates
    grad_ab = np.array([np.dot(grad_tang, t1), np.dot(grad_tang, t2)])
    
    return energy, grad_ab, t1, t2

def _optimize_point_k(X, k, deadline, max_iter=200):
    """
    Optimize point k on the sphere using L-BFGS-B in 2D tangent coordinates.
    Returns updated X.
    """
    xk = X[k].copy()
    
    # Build tangent basis
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(xk, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(xk, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(xk, t1)
    t2 /= np.linalg.norm(t2)
    
    def obj_and_grad(ab):
        if time.time() > deadline:
            # Return large value to signal stop
            return 1e10, np.array([0.0, 0.0])
        
        # Reconstruct point on sphere
        x_new = xk + ab[0] * t1 + ab[1] * t2
        norm = np.linalg.norm(x_new)
        if norm < 1e-15:
            return 1e10, np.array([0.0, 0.0])
        x_new = x_new / norm
        
        # Temporarily place x_new in X
        X[k] = x_new
        e, g_ab, _, _ = _energy_point_k_and_grad(X, k)
        return e, g_ab
    
    # Initial point in tangent coordinates
    ab0 = np.array([0.0, 0.0])
    
    res = minimize(
        obj_and_grad,
        ab0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': max_iter}
    )
    
    # Extract best point
    ab_best = res.x
    x_best = xk + ab_best[0] * t1 + ab_best[1] * t2
    x_best /= np.linalg.norm(x_best)
    
    X[k] = x_best
    return X

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    rng = np.random.RandomState(seed)
    v = rng.randn(n, 3)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v

def _coordinate_descent(X, deadline, max_sweeps=50):
    """
    Coordinate descent: repeatedly optimize each point in sequence.
    Each point is optimized via 2D L-BFGS-B on its tangent plane.
    """
    X = X.copy()
    best_E = _energy_full(X)
    best_X = X.copy()
    
    sweep = 0
    while time.time() < deadline:
        sweep += 1
        improved = False
        
        for k in range(N):
            if time.time() > deadline - 0.5:
                break
            
            # Time per point: allocate remaining time / remaining points
            remaining_points = N - k
            time_per_point = max(0.1, (deadline - 0.5 - time.time()) / remaining_points)
            point_deadline = time.time() + min(time_per_point, 2.0)  # cap at 2s per point
            
            X_before = X[k].copy()
            X = _optimize_point_k(X, k, point_deadline, max_iter=100)
            
            if not np.allclose(X[k], X_before, atol=1e-10):
                improved = True
        
        # Evaluate after full sweep
        E = _energy_full(X)
        if E < best_E - 1e-12:
            best_E = E
            best_X = X.copy()
        
        if not improved and sweep > 2:
            break
    
    return best_X, best_E

def _coordinate_descent_random_order(X, deadline, max_sweeps=50):
    """
    Coordinate descent with random point ordering to avoid bias.
    """
    X = X.copy()
    best_E = _energy_full(X)
    best_X = X.copy()
    rng = np.random.RandomState(42)
    
    sweep = 0
    while time.time() < deadline:
        sweep += 1
        improved = False
        
        order = rng.permutation(N)
        for idx in range(N):
            k = order[idx]
            if time.time() > deadline - 0.5:
                break
            
            remaining_points = N - idx
            time_per_point = max(0.1, (deadline - 0.5 - time.time()) / remaining_points)
            point_deadline = time.time() + min(time_per_point, 2.0)
            
            X_before = X[k].copy()
            X = _optimize_point_k(X, k, point_deadline, max_iter=100)
            
            if not np.allclose(X[k], X_before, atol=1e-10):
                improved = True
        
        E = _energy_full(X)
        if E < best_E - 1e-12:
            best_E = E
            best_X = X.copy()
        
        if not improved and sweep > 2:
            break
    
    return best_X, best_E

def _simulated_annealing_coord(X, deadline, max_sweeps=30):
    """
    Coordinate descent with occasional random perturbations (annealing).
    """
    X = X.copy()
    best_E = _energy_full(X)
    best_X = X.copy()
    rng = np.random.RandomState(123)
    
    temperature = 0.1
    cool_rate = 0.95
    
    sweep = 0
    while time.time() < deadline:
        sweep += 1
        improved = False
        
        order = rng.permutation(N)
        for idx in range(N):
            k = order[idx]
            if time.time() > deadline - 0.5:
                break
            
            remaining_points = N - idx
            time_per_point = max(0.1, (deadline - 0.5 - time.time()) / remaining_points)
            point_deadline = time.time() + min(time_per_point, 2.0)
            
            # Occasionally perturb before optimizing
            if temperature > 0.01 and rng.random() < 0.1:
                noise = rng.randn(3) * temperature
                X[k] = X[k] + noise
                X[k] /= np.linalg.norm(X[k])
            
            X_before = X[k].copy()
            X = _optimize_point_k(X, k, point_deadline, max_iter=80)
            
            if not np.allclose(X[k], X_before, atol=1e-10):
                improved = True
        
        E = _energy_full(X)
        if E < best_E - 1e-12:
            best_E = E
            best_X = X.copy()
        
        temperature *= cool_rate
        
        if not improved and sweep > 3:
            break
    
    return best_X, best_E

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Generate diverse starting configurations
    np.random.seed(0)
    
    starts = []
    
    # 1. Fibonacci sphere
    X_fib = _fibonacci_sphere(N)
    starts.append(('fib', X_fib))
    
    # 2. Fibonacci with small perturbations
    X_pert = X_fib.copy()
    noise = np.random.randn(N, 3) * 0.02
    X_pert += noise
    X_pert /= np.linalg.norm(X_pert, axis=1, keepdims=True)
    starts.append(('pert', X_pert))
    
    # 3. Random sphere
    X_rand = _random_sphere(N, seed=7)
    starts.append(('rand', X_rand))
    
    # 4. Another random with different seed
    X_rand2 = _random_sphere(N, seed=99)
    starts.append(('rand2', X_rand2))
    
    best_X_global = None
    best_E_global = float('inf')
    
    # Phase 1: Coordinate descent from each start (sequential, time-allocated)
    # Allocate time proportionally; structured starts get more time
    time_alloc = [40.0, 30.0, 35.0, 30.0]
    
    for i, (name, X0) in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        
        t_end = min(time.time() + time_alloc[i], deadline - 10.0)
        
        # Choose method based on start type
        if name in ('fib', 'pert'):
            # Structured: use ordered coordinate descent
            X_opt, E_opt = _coordinate_descent(X0, t_end)
        else:
            # Random: use random-order coordinate descent
            X_opt, E_opt = _coordinate_descent_random_order(X0, t_end)
        
        if E_opt < best_E_global:
            best_E_global = E_opt
            best_X_global = X_opt.copy()
    
    # Phase 2: Polish the best result with simulated annealing coordinate descent
    if best_X_global is not None and time.time() < deadline - 5.0:
        t_end = deadline - 2.0
        X_polished, E_polished = _simulated_annealing_coord(best_X_global, t_end)
        if E_polished < best_E_global:
            best_E_global = E_polished
            best_X_global = X_polished
    
    # Phase 3: Final tight coordinate descent polish
    if best_X_global is not None and time.time() < deadline - 3.0:
        t_end = deadline - 1.0
        X_final, E_final = _coordinate_descent(best_X_global, t_end)
        if E_final < best_E_global:
            best_E_global = E_final
            best_X_global = X_final
    
    # Final normalization and validation
    if best_X_global is None:
        best_X_global = _fibonacci_sphere(N)
    
    best_X_global = best_X_global / np.linalg.norm(best_X_global, axis=1, keepdims=True)
    
    # Validate: no points too close, all finite, unit norm
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_X_global[i] - best_X_global[j])
            if d < 1e-9:
                # Nudge apart
                best_X_global[j] += np.random.randn(3) * 1e-8
                best_X_global[j] /= np.linalg.norm(best_X_global[j])
    
    return best_X_global
