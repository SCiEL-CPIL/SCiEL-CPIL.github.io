"""Rewritten program: minimise logarithmic energy of 150 points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
# Deadline guard: evaluator kills after 150s; keep a safety margin.
DEADLINE = 140.0  # seconds


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(pts):
    """Vectorized log energy (less precise but much faster)."""
    # pts: (N, 3)
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    return np.sum(np.log(1.0 / d))


def _log_energy_grad(pts):
    """Compute log energy and gradient using vectorized ops."""
    # pts: (N, 3)
    sq = np.sum(pts * pts, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)

    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = np.sum(np.log(1.0 / d))

    # Gradient: d/dx_i log(1/|x_i - x_j|) = -(x_i - x_j) / |x_i - x_j|^2
    # Sum over j != i
    # For each pair (i,j), contribution to grad[i] is -(x_i - x_j)/d^2, to grad[j] is (x_i - x_j)/d^2
    grad = np.zeros_like(pts)
    # Vectorized: for all pairs
    diff = pts[:, None, :] - pts[None, :, :]  # (N, N, 3)
    inv_d2 = 1.0 / dist_sq  # (N, N)
    # grad[i] = sum_j (x_j - x_i) / |x_i - x_j|^2 = -sum_j (x_i - x_j)/d^2
    # Actually: d/dx_i log(1/d) = -d/dx_i log(d) = -d/dx_i (1/2 log(d^2)) = -(x_i - x_j)/d^2
    # So grad[i] = sum_{j!=i} -(x_i - x_j)/d_ij^2
    # = sum_{j!=i} (x_j - x_i)/d_ij^2
    # diff[i,j] = x_i - x_j, so we want -diff * inv_d2 summed over j
    grad = -np.sum(diff * inv_d2[:, :, None], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_start():
    """Golden-angle Fibonacci spiral on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(seed):
    """Random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _optimize_lbfgs(pts0, max_time, maxiter=5000, ftol=1e-15, gtol=1e-10, maxfun=10000):
    """Run L-BFGS-B optimization from a starting configuration."""
    start_time = time.time()
    
    def objective_and_grad(x):
        pts = x.reshape(N, 3)
        e, g = _log_energy_grad(pts)
        return e, g.flatten()
    
    try:
        res = minimize(
            objective_and_grad,
            pts0.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxfun}
        )
        best_pts = _project_to_sphere(res.x.reshape(N, 3))
        e_val = _log_energy_fast(best_pts)
    except Exception:
        best_pts = pts0
        e_val = _log_energy_fast(pts0)
    
    elapsed = time.time() - start_time
    return best_pts, e_val, elapsed


def _optimize_adam(pts_init, max_time, lr0=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer with tangent-space projection for sphere constraint."""
    pts = pts_init.copy()
    energy = _log_energy_fast(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    stagnation = 0
    
    while time.time() < max_time:
        # Compute gradient
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-12)
        
        # Force F_i = sum_j (x_i - x_j) / dist_sq_ij
        forces = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        
        # Project to tangent space
        dot_products = np.sum(forces * pts, axis=1)
        grad = forces - dot_products[:, np.newaxis] * pts
        
        # Adam update
        t += 1
        lr = lr0 * (0.999 ** t)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        new_pts = pts - lr * update
        norms = np.sqrt(np.sum(new_pts**2, axis=1))
        new_pts = new_pts / norms[:, np.newaxis]
        
        new_energy = _log_energy_fast(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > 100:
                break
    
    return best_pts, best_energy


def construct():
    global DEADLINE
    start_time = time.time()
    
    # Start with Fibonacci spiral
    pts_fib = _fibonacci_start()
    current_energy = _log_energy_fast(pts_fib)
    best_pts = pts_fib.copy()
    best_energy = current_energy
    
    # Phase 1: L-BFGS-B optimization from Fibonacci start (fast, high precision)
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 15:
        pts_opt, e_opt, elapsed = _optimize_lbfgs(pts_fib, time.time() + remaining - 10)
        if time.time() - start_time < DEADLINE - 5:
            e_precise = _log_energy(pts_opt)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_opt
    
    # Phase 2: Adam optimization for local refinement and escape from local minima
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 20:
        pts_adam, e_adam = _optimize_adam(best_pts, time.time() + remaining - 10)
        if time.time() - start_time < DEADLINE - 5:
            e_precise = _log_energy(pts_adam)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_adam
    
    # Phase 3: Multiple random restarts with quick local optimization
    rng = np.random.RandomState(42)
    n_restarts = 15
    for i in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        pts_rand = _random_start(i + 1)
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 10:
            break
        
        # Quick L-BFGS-B from random start
        try:
            pts_new, e_new, _ = _optimize_lbfgs(pts_rand, time.time() + remaining - 5, maxiter=1000, ftol=1e-12, gtol=1e-8, maxfun=2000)
            e_precise = _log_energy(pts_new)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_new
        except Exception:
            continue
    
    # Final polish: high-precision L-BFGS-B with tighter tolerances
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        pts_polish, e_polish, _ = _optimize_lbfgs(best_pts, time.time() + remaining - 3, maxiter=2000, ftol=1e-18, gtol=1e-12, maxfun=5000)
        if time.time() - start_time < DEADLINE - 2:
            e_final = _log_energy(pts_polish)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = pts_polish
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(best_pts)):
        return _fibonacci_start()
    
    return _project_to_sphere(best_pts)
