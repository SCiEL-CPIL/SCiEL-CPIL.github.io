"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _energy_and_grad(x):
    """
    Compute logarithmic energy and gradient for N points on unit sphere.
    x: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal and very small distances
    eps = 1e-15
    dist_safe = np.maximum(dist, eps)
    np.fill_diagonal(dist_safe, 1.0)
    
    # Energy: sum over i<j of -log(dist[i,j])
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad[i] = -sum_{j != i} (x_i - x_j) / dist[i,j]^2
    inv_dist_sq = 1.0 / (dist_safe * dist_safe)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    S = np.sum(inv_dist_sq, axis=1)  # (N,)
    W = inv_dist_sq @ x  # (N, 3)
    grad = -x * S[:, np.newaxis] + W
    
    return energy, grad

def _project_and_normalize(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms

def _fmin_on_sphere(x0, maxiter=5000, tol=1e-12, gtol=1e-10):
    """
    Minimize energy on the sphere using L-BFGS-B with tangent-space gradient projection.
    """
    x = _project_and_normalize(x0.copy())
    
    def objective_and_grad(x_flat):
        x_2d = x_flat.reshape(N, 3)
        x_2d = _project_and_normalize(x_2d)
        e, g = _energy_and_grad(x_2d)
        # Project gradient to tangent space of sphere
        g_tangent = g - (g * x_2d).sum(axis=1, keepdims=True) * x_2d
        return e, g_tangent.flatten()
    
    res = minimize(
        objective_and_grad,
        x.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': gtol, 'maxls': 30}
    )
    
    x_opt = _project_and_normalize(res.x.reshape(N, 3))
    return x_opt

def _energy(x):
    """Compute energy only."""
    e, _ = _energy_and_grad(x)
    return e

def _energy_fsum(x):
    """Compute energy using math.fsum to match evaluator exactly."""
    import math
    terms = []
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(x[i] - x[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _fibonacci_spiral(n=N):
    """Generate points via Fibonacci spiral on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n=N, seed=None):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _perturbed_fibonacci(n=N, sigma=0.1, seed=None):
    """Generate Fibonacci spiral with small Gaussian perturbation."""
    pts = _fibonacci_spiral(n).copy()
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * sigma
    pts = pts + noise
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _simulated_annealing(x_start, best_e, max_time=60.0, seed=42):
    """
    Simulated annealing on the sphere to escape local minima.
    Randomly perturb a random point, accept if energy decreases or with
    Boltzmann probability. Track global best.
    """
    rng = np.random.RandomState(seed)
    x = x_start.copy()
    e = _energy(x)
    best_x = x.copy()
    best_e_local = e
    t0 = time.time()
    T = 0.5  # initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    while time.time() - t0 < max_time:
        # Pick a random point to perturb
        idx = rng.randint(N)
        # Random tangent direction
        tangent = rng.randn(3)
        tangent -= (tangent @ x[idx]) * x[idx]
        tn = np.linalg.norm(tangent)
        if tn < 1e-12:
            continue
        tangent /= tn
        
        # Perturbation size scales with temperature
        step = 0.05 * T / 0.5  # scale step with T
        x_new = x.copy()
        x_new[idx] = x[idx] + step * tangent
        x_new[idx] /= np.linalg.norm(x_new[idx])
        
        e_new = _energy(x_new)
        
        if e_new < e or rng.rand() < np.exp(-(e_new - e) / max(T, T_min)):
            x = x_new
            e = e_new
            if e < best_e_local:
                best_e_local = e
                best_x = x.copy()
        
        T = max(T * cooling, T_min)
    
    return best_x, best_e_local

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Initial best: Fibonacci spiral
    x0 = _fibonacci_spiral()
    best_x = x0.copy()
    best_e = _energy(best_x)
    
    # Phase 1: Multiple restarts with moderate iterations
    np.random.seed(12345)
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline - 40.0:
            break
        
        if restart == 0:
            x_init = x0.copy()
        elif restart == 1:
            x_init = _random_points(seed=100 + restart)
        elif restart % 3 == 0:
            x_init = _perturbed_fibonacci(sigma=0.05, seed=200 + restart)
        elif restart % 3 == 1:
            x_init = _random_points(seed=300 + restart)
        else:
            x_init = _perturbed_fibonacci(sigma=0.15, seed=400 + restart)
        
        try:
            x_opt = _fmin_on_sphere(x_init, maxiter=3000, tol=1e-10, gtol=1e-8)
            e_opt = _energy(x_opt)
            
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Simulated annealing to escape local minima
    remaining = deadline - time.time()
    if remaining > 30.0:
        sa_time = min(remaining - 25.0, 50.0)
        try:
            sa_x, sa_e = _simulated_annealing(best_x, best_e, max_time=sa_time, seed=42)
            if sa_e < best_e:
                best_e = sa_e
                best_x = sa_x.copy()
        except Exception:
            pass
        
        # Polish after SA
        if time.time() < deadline - 15.0:
            try:
                x_pol = _fmin_on_sphere(best_x, maxiter=5000, tol=1e-12, gtol=1e-10)
                e_pol = _energy(x_pol)
                if e_pol < best_e:
                    best_e = e_pol
                    best_x = x_pol.copy()
            except Exception:
                pass
    
    # Phase 3: High-precision polish on the best configuration
    if time.time() < deadline - 15.0:
        try:
            x_final = _fmin_on_sphere(best_x, maxiter=10000, tol=1e-14, gtol=1e-12)
            e_final = _energy(x_final)
            if e_final < best_e:
                best_e = e_final
                best_x = x_final
        except Exception:
            pass
    
    # Phase 4: One more ultra-tight polish if time permits
    if time.time() < deadline - 5.0:
        try:
            x_final2 = _fmin_on_sphere(best_x, maxiter=5000, tol=1e-16, gtol=1e-14)
            e_final2 = _energy(x_final2)
            if e_final2 < best_e:
                best_e = e_final2
                best_x = x_final2
        except Exception:
            pass
    
    # Ensure points are on unit sphere and valid
    best_x = _project_and_normalize(best_x)
    
    # Check for very close points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        return _fibonacci_spiral()
    
    return best_x
# EVOLVE-BLOCK-END
