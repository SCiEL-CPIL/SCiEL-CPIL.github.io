"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    # log(1/d) = -0.5 * log(d_sq)
    d_sq = np.clip(d_sq, 1e-30, None)
    terms = [-0.5 * math.log(di) for di in d_sq]
    return math.fsum(terms)


def _fast_energy(pts):
    """Fast numpy-based energy for inner loops (not used for final scoring)."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    d_sq = dist_sq[iu]
    d_sq = np.clip(d_sq, 1e-30, None)
    return float(-0.5 * np.sum(np.log(d_sq)))


def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _energy_and_grad(x):
    """Compute energy and gradient for L-BFGS. x is (N,3) flattened."""
    pts = x.reshape(N, 3)
    # Project to sphere for correct gradient
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.clip(dist_sq, 1e-15, None)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i<j} log(d_ij^2)
    iu = np.triu_indices(N, k=1)
    d_sq_ij = dist_sq[iu]
    energy = -0.5 * np.sum(np.log(d_sq_ij))
    
    # Vectorized gradient computation
    # grad[i] = sum_{j!=i} (pts[j] - pts[i]) / dist_sq[i,j]
    # = sum_j pts[j]/dist_sq[i,j] - pts[i] * sum_j 1/dist_sq[i,j]
    # But dist_sq[i,i] = 0, so we need to handle that
    
    # 1/dist_sq with diagonal set to 0
    inv_d_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-15
    inv_d_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad[i] = sum_j inv_d_sq[i,j] * (pts[j] - pts[i])
    #         = sum_j inv_d_sq[i,j] * pts[j] - pts[i] * sum_j inv_d_sq[i,j]
    row_sums = np.sum(inv_d_sq, axis=1, keepdims=True)  # (N, 1)
    weighted_sum = inv_d_sq @ pts  # (N, 3)
    grad = weighted_sum - pts * row_sums
    
    # Project gradient to tangent space of sphere
    dots = np.sum(grad * pts, axis=1, keepdims=True)  # (N, 1)
    grad = grad - dots * pts
    
    return energy, grad.flatten()


def _optimize_lbfgs(start_pts, deadline, maxiter=10000, ftol=1e-12, gtol=1e-8):
    """Run L-BFGS optimization on the sphere."""
    x0 = start_pts.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        e, _ = _energy_and_grad(x)
        return e
    
    def jac(x):
        if time.time() > deadline:
            return np.ones(N * 3) * 1e6
        _, g = _energy_and_grad(x)
        return g
    
    try:
        res = minimize(objective, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'maxfun': maxiter * 10,
                                'ftol': ftol, 'gtol': gtol})
        result = _project_to_sphere(res.x.reshape(N, 3))
        return result
    except Exception:
        return None


def _anneal_optimize(start_pts, deadline):
    """Simulated annealing style optimization using fast energy."""
    pts = start_pts.copy()
    energy = _fast_energy(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    T = 0.1
    cooling = 0.995
    step_size = 0.05
    
    max_iter = 10000
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random perturbation
        idx = np.random.randint(N)
        noise = np.random.randn(3) * step_size
        new_pts = pts.copy()
        new_pts[idx] += noise
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_energy = _fast_energy(new_pts)
        delta_e = new_energy - energy
        
        if delta_e < 0 or np.random.random() < math.exp(-delta_e / max(T, 1e-10)):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        T *= cooling
        step_size = max(0.003, step_size * 0.997)
    
    return best_pts


def construct():
    """Main entry point: generate many candidates, optimize, keep best."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    fib_pts = _initial_fibonacci()
    fib_energy = _log_energy(fib_pts)
    best_pts = fib_pts.copy()
    best_energy = fib_energy
    
    # Try multiple diverse initializations with L-BFGS
    np.random.seed(12345)
    
    # Generate diverse starting points
    starts = []
    
    # Fibonacci spiral
    starts.append(fib_pts)
    
    # Rotated Fibonacci spirals
    for angle in [0.5, 1.0, 1.5, 2.0]:
        rot_pts = fib_pts.copy()
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        starts.append(rot_pts @ rot.T)
    
    # Random points
    for i in range(6):
        random_pts = np.random.randn(N, 3)
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        starts.append(random_pts)
    
    # Optimize each starting point
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time() - 15
        if remaining < 5:
            break
        
        maxiter = min(5000, int(remaining * 100))
        result = _optimize_lbfgs(start, deadline - 15, maxiter=maxiter)
        if result is not None:
            energy = _log_energy(result)
            if energy < best_energy:
                best_energy = energy
                best_pts = result
    
    # Annealing refinement
    if time.time() < deadline - 15:
        refined = _anneal_optimize(best_pts, deadline - 10)
        refined_energy = _log_energy(refined)
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_pts = refined
    
    # Final L-BFGS polish with tighter tolerances
    if time.time() < deadline - 8:
        result = _optimize_lbfgs(best_pts, deadline - 5, maxiter=3000, 
                                  ftol=1e-14, gtol=1e-10)
        if result is not None:
            energy = _log_energy(result)
            if energy < best_energy:
                best_energy = energy
                best_pts = result
    
    # Second polish if time remains
    if time.time() < deadline - 3:
        result = _optimize_lbfgs(best_pts, deadline - 1, maxiter=2000,
                                  ftol=1e-15, gtol=1e-12)
        if result is not None:
            energy = _log_energy(result)
            if energy < best_energy:
                best_energy = energy
                best_pts = result
    
    # Final normalization and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
