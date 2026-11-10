"""Optimized logarithmic energy minimization for 100 points on the unit sphere."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START
def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _log_energy(pts):
    """
    Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)
    Using chordal distances on unit sphere: |x_i - x_j| = sqrt(2 - 2*x_i.x_j)
    So log(1/|x_i - x_j|) = -0.5 * log(2 - 2*x_i.x_j)
    """
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    return -0.5 * math.fsum(np.log(vals).tolist())


def _log_energy_and_grad(pts):
    """
    Compute energy and gradient of log(1/|x_i - x_j|) on unit sphere.
    
    E = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    dE/dx_k = sum_{j!=k} x_j / (2 - 2*x_k.x_j)
    
    Then project to tangent: grad_tangent = grad - (grad . x_k) * x_k
    """
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy
    iu = np.triu_indices(N, 1)
    vals = dist_sq[iu]
    vals = np.maximum(vals, 1e-30)
    energy = -0.5 * math.fsum(np.log(vals).tolist())
    
    # Gradient: grad_k = sum_{j != k} x_j / dist_sq[k,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = inv_dist_sq @ pts  # (N, 3)
    
    # Project gradient to tangent space
    grad_dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - grad_dot * pts
    
    return energy, grad


def _optimize_lbfgs(pts, max_iter=2000, ftol=1e-16, gtol=1e-12):
    """
    L-BFGS optimization on the sphere using retraction (project back to sphere).
    """
    from scipy.optimize import minimize
    
    def objective_and_grad(x):
        pts_local = _project_to_sphere(x.reshape(N, 3))
        e, g = _log_energy_and_grad(pts_local)
        return e, g.flatten()
    
    result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B',
                      jac=True, options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                         'maxls': 50, 'maxfun': 5000})
    pts_opt = _project_to_sphere(result.x.reshape(N, 3))
    return pts_opt, result.fun, result


def _polish(pts, time_budget=5.0, gtol=1e-15):
    """High-precision polish using small gradient steps with tight tolerances."""
    start = time.time()
    pts = pts.copy()
    
    for step in range(500):
        if time.time() - start > time_budget:
            break
        e, g = _log_energy_and_grad(pts)
        
        gnorm = np.linalg.norm(g)
        if gnorm < gtol:
            break
        
        # Adaptive step size: start small, try a few
        best_e = e
        best_pts = pts.copy()
        for lr in [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]:
            new_pts = pts - lr * g
            new_pts = _project_to_sphere(new_pts)
            new_e, _ = _log_energy_and_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts
        
        if np.allclose(best_pts, pts, atol=1e-16):
            break
        pts = best_pts
    
    return pts


def _adam_optimize(pts, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=5000,
                   time_budget=30.0):
    """
    Adam optimization on the sphere with retraction.
    Uses per-point adaptive learning rates.
    """
    start = time.time()
    pts = pts.copy()
    m = np.zeros_like(pts)  # first moment
    v = np.zeros_like(pts)  # second moment
    t = 0
    best_e = _log_energy(pts)
    best_pts = pts.copy()
    
    for step in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        t += 1
        e, g = _log_energy_and_grad(pts)
        
        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        
        new_pts = pts - update
        new_pts = _project_to_sphere(new_pts)
        
        new_e, _ = _log_energy_and_grad(new_pts)
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
        
        pts = new_pts
    
    return best_pts, best_e


def _simulated_annealing(pts, time_budget=30.0, rng=None):
    """
    Simulated annealing on the sphere with periodic re-optimization.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    start = time.time()
    pts = pts.copy()
    current_e = _log_energy(pts)
    best_e = current_e
    best_pts = pts.copy()
    
    # Temperature schedule
    T_start = 0.01
    T_end = 0.0001
    total_steps = 5000
    steps_done = 0
    
    while steps_done < total_steps:
        if time.time() - start > time_budget:
            break
        
        # Periodic re-optimization
        if steps_done % 500 == 0 and steps_done > 0:
            try:
                pts_opt, e_opt, _ = _optimize_lbfgs(pts, max_iter=200, ftol=1e-14, gtol=1e-12)
                if e_opt < current_e:
                    current_e = e_opt
                    pts = pts_opt
                if current_e < best_e:
                    best_e = current_e
                    best_pts = pts.copy()
            except Exception:
                pass
        
        # Temperature
        progress = steps_done / total_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        idx = rng.integers(0, N)
        direction = rng.standard_normal(3)
        direction = direction / np.linalg.norm(direction)
        # Project to tangent
        tangent = direction - np.dot(direction, pts[idx]) * pts[idx]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            tangent = direction
            tangent_norm = 1.0
        tangent = tangent / tangent_norm
        
        step_size = T * 0.1
        new_pts = pts.copy()
        new_pts[idx] += step_size * tangent
        new_pts = _project_to_sphere(new_pts)
        
        new_e = _log_energy(new_pts)
        delta = new_e - current_e
        
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pts = pts.copy()
        
        steps_done += 1
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = [_fibonacci_spiral()]
    
    # Add random starts
    for _ in range(3):
        initializations.append(_random_sphere(rng))
    
    # Add perturbed Fibonacci
    fib = _fibonacci_spiral()
    for _ in range(2):
        perturbed = fib + rng.standard_normal((N, 3)) * 0.1
        initializations.append(_project_to_sphere(perturbed))
    
    best_pts = None
    best_e = float('inf')
    
    for init_pts in initializations:
        if time.time() > deadline - 20.0:
            break
        
        init_pts = _project_to_sphere(init_pts)
        
        # L-BFGS from this initialization
        try:
            pts_opt, e_opt, _ = _optimize_lbfgs(init_pts, max_iter=2000, ftol=1e-15, gtol=1e-13)
            if e_opt < best_e:
                best_e = e_opt
                best_pts = pts_opt
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _project_to_sphere(_fibonacci_spiral())
        best_e = _log_energy(best_pts)
    
    # Check time
    if time.time() > deadline - 25.0:
        # Quick polish
        try:
            polished = _polish(best_pts, time_budget=min(10.0, deadline - time.time() - 5.0))
            e_pol = _log_energy(polished)
            if e_pol < best_e:
                best_pts = polished
                best_e = e_pol
        except Exception:
            pass
        return _project_to_sphere(best_pts)
    
    # Simulated annealing from best point
    try:
        sa_pts, sa_e = _simulated_annealing(best_pts, time_budget=min(30.0, deadline - time.time() - 20.0), rng=rng)
        if sa_e < best_e:
            best_e = sa_e
            best_pts = sa_pts
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 15.0:
        # Quick polish
        try:
            polished = _polish(best_pts, time_budget=min(8.0, deadline - time.time() - 5.0))
            e_pol = _log_energy(polished)
            if e_pol < best_e:
                best_pts = polished
                best_e = e_pol
        except Exception:
            pass
        return _project_to_sphere(best_pts)
    
    # Final L-BFGS refinement
    try:
        pts_opt, e_opt, _ = _optimize_lbfgs(best_pts, max_iter=3000, ftol=1e-16, gtol=1e-14)
        if e_opt < best_e:
            best_e = e_opt
            best_pts = pts_opt
    except Exception:
        pass
    
    # High-precision polish
    if time.time() < deadline - 5.0:
        try:
            polished = _polish(best_pts, time_budget=min(10.0, deadline - time.time() - 3.0), gtol=1e-15)
            e_pol = _log_energy(polished)
            if e_pol < best_e:
                best_pts = polished
                best_e = e_pol
        except Exception:
            pass
    
    # Ensure validity
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    dot = best_pts @ best_pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, 1)
    min_dist_sq = np.min(dist_sq[iu])
    if min_dist_sq < 1e-18:  # distance < 1e-9
        best_pts += rng.standard_normal((N, 3)) * 1e-6
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
