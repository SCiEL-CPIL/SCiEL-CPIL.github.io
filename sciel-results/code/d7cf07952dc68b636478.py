"""Combined approach: L-BFGS-B with tangent gradient + multi-start + SA polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = points.shape[0]
    dots = points @ points.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    return -math.fsum(np.log(d).tolist())


def _energy_grad(points):
    """Compute gradient of logarithmic energy."""
    n = points.shape[0]
    dots = points @ points.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    coeff = 1.0 / dist_sq
    np.fill_diagonal(coeff, 0.0)
    coeff_sum = coeff.sum(axis=1, keepdims=True)
    grad = coeff @ points - coeff_sum * points
    return grad


def _project_to_sphere(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _tangent_project(grad, x):
    """Project gradient to tangent space of sphere at each point."""
    return grad - np.sum(grad * x, axis=1, keepdims=True) * x


def _fibonacci_sphere(n, offset=0.0):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + offset)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(n, rng):
    """Generate random points on the unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _lbfgs_optimize(x0, time_budget, maxiter=3000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from x0 with a time budget."""
    start_time = time.time()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        e = _energy(x)
        g = _energy_grad(x)
        g_tangent = _tangent_project(g, x)
        return e, g_tangent.flatten()
    
    x_flat = x0.flatten()
    best_x = x0.copy()
    best_e = _energy(x0)
    
    try:
        res = minimize(
            objective_and_grad,
            x_flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 3, 'ftol': ftol, 'gtol': gtol}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        e_opt = _energy(x_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    # Second pass with tighter tolerance if time permits
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.6:
        try:
            res2 = minimize(
                objective_and_grad,
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxiter * 3, 
                         'ftol': ftol * 0.01, 'gtol': gtol * 0.01}
            )
            x_opt2 = _project_to_sphere(res2.x.reshape(-1, 3))
            e_opt2 = _energy(x_opt2)
            if e_opt2 < best_e:
                best_e = e_opt2
                best_x = x_opt2.copy()
        except Exception:
            pass
    
    return best_x, best_e


def _simulated_annealing(x0, time_budget, rng):
    """Simulated annealing on the sphere."""
    start_time = time.time()
    x = x0.copy()
    e = _energy(x)
    best_x = x.copy()
    best_e = e
    
    n = x.shape[0]
    T_start = 0.5
    T_end = 1e-5
    n_steps = 3000
    temps = np.geomspace(T_start, T_end, n_steps)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        T = temps[step]
        step_size = 0.05 * T + 0.002
        
        # Perturb random subset
        n_perturb = np.random.randint(1, 6)
        indices = np.random.choice(n, size=n_perturb, replace=False)
        perturb = rng.standard_normal((n_perturb, 3)) * step_size
        
        x_prop = x.copy()
        x_prop[indices] += perturb
        x_prop = _project_to_sphere(x_prop)
        
        e_prop = _energy(x_prop)
        delta_e = e_prop - e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-12)):
            x = x_prop
            e = e_prop
            if e < best_e:
                best_e = e
                best_x = x.copy()
    
    return best_x, best_e


def construct():
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # Base Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # Perturbed Fibonacci spirals
    for scale in [0.1, 0.05, 0.02, 0.01]:
        perturbed = _fibonacci_sphere(N) + scale * rng.standard_normal((N, 3))
        inits.append(_project_to_sphere(perturbed))
    
    # Different spiral offsets
    for offset in [1.0, 2.0, 0.5, 3.0, 1.5, 4.0, 0.3, 5.0, 2.7]:
        inits.append(_fibonacci_sphere(N, offset))
    
    # Random points
    for seed in [123, 456, 789, 101, 202, 303]:
        r = np.random.default_rng(seed)
        inits.append(_random_sphere_points(N, r))
    
    best_x = None
    best_e = float('inf')
    
    num_inits = len(inits)
    
    for i, x0 in enumerate(inits):
        remaining = _remaining()
        if remaining < 15:
            break
        
        # Allocate time for this initialization
        time_budget = max(4.0, remaining / max(1, num_inits - i))
        
        # L-BFGS optimization
        x_opt, e_opt = _lbfgs_optimize(x0, time_budget * 0.6, maxiter=2000, ftol=1e-14, gtol=1e-11)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # If time permits, try SA from this result
        remaining = _remaining()
        if remaining > 20 and e_opt < best_e + 1e-6:
            sa_budget = min(8.0, remaining * 0.25)
            x_sa, e_sa = _simulated_annealing(x_opt, sa_budget, rng)
            if e_sa < best_e:
                best_e = e_sa
                best_x = x_sa.copy()
                # Polish after SA
                remaining = _remaining()
                if remaining > 10:
                    x_pol, e_pol = _lbfgs_optimize(x_sa, min(5.0, remaining * 0.3), maxiter=1000, ftol=1e-15, gtol=1e-12)
                    if e_pol < best_e:
                        best_e = e_pol
                        best_x = x_pol.copy()
    
    # Final high-precision polish on the best solution
    if best_x is not None:
        remaining = _remaining()
        if remaining > 8:
            x_final, e_final = _lbfgs_optimize(
                best_x, min(6.0, remaining * 0.4), 
                maxiter=5000, ftol=1e-16, gtol=1e-13
            )
            if e_final < best_e:
                best_e = e_final
                best_x = x_final.copy()
    
    if best_x is None:
        best_x = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_x = _project_to_sphere(best_x)
    
    # Check for too-close points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                best_x[j] += rng.standard_normal(3) * 1e-6
                best_x = _project_to_sphere(best_x)
                break
    
    # Verify all finite
    if not np.all(np.isfinite(best_x)):
        best_x = _fibonacci_sphere(N)
        best_x, _ = _lbfgs_optimize(best_x, 5.0, maxiter=1000)
        best_x = _project_to_sphere(best_x)
    
    return best_x
