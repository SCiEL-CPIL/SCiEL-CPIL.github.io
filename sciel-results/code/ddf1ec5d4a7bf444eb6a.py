"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Uses dot-product trick for speed.
    """
    pts = pts.reshape(-1, 3)
    n = pts.shape[0]
    
    # Compute pairwise dot products
    dot = pts @ pts.T  # (n, n)
    
    # Squared distances: |x_i - x_j|^2 = 2 - 2*dot_ij for unit vectors
    diff_sq = 2.0 - 2.0 * dot
    # Clamp to avoid negative values from floating point
    diff_sq = np.maximum(diff_sq, 1e-15)
    np.fill_diagonal(diff_sq, 1.0)
    
    # Energy: sum over i<j of -log(sqrt(diff_sq)) = -0.5 * sum_{i<j} log(diff_sq)
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(diff_sq[iu]))
    
    # Gradient computation
    # inv_dist_sq = 1 / |x_i - x_j|^2
    inv_dist_sq = np.zeros_like(diff_sq)
    np.fill_diagonal(inv_dist_sq, 0)
    np.divide(1.0, diff_sq, out=inv_dist_sq, where=diff_sq > 1e-15)
    
    # grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # = sum_j x_j * inv_dist_sq[i,j] - x_i * sum_j inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = inv_dist_sq @ pts - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _energy_fast(pts):
    """Fast energy computation without gradient."""
    pts = pts.reshape(-1, 3)
    n = pts.shape[0]
    dot = pts @ pts.T
    diff_sq = 2.0 - 2.0 * dot
    diff_sq = np.maximum(diff_sq, 1e-15)
    np.fill_diagonal(diff_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(diff_sq[iu]))


def _project_and_normalize(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci lattice points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _simulated_annealing(pts, time_budget):
    """
    Simulated Annealing on the sphere using random angular perturbations.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy = _energy_fast(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    # Annealing parameters
    T_start = 2.0
    T_end = 0.005
    
    rng = np.random.RandomState(42)
    
    # Adaptive step size
    step_size = 0.3
    accept_count = 0
    total_count = 0
    window = 100
    
    while time.time() - start_time < time_budget:
        # Generate a random perturbation: move one or a few points
        # Choose how many points to move
        if rng.rand() < 0.7:
            # Move single point
            idx = rng.randint(N)
            axis = rng.randn(3)
            axis /= np.linalg.norm(axis)
            angle = rng.randn() * step_size
            c = np.cos(angle)
            s = np.sin(angle)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
            
            new_pts = current_pts.copy()
            new_pts[idx] = new_pts[idx] @ R.T
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        else:
            # Move two points in opposite directions
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            axis = rng.randn(3)
            axis /= np.linalg.norm(axis)
            angle = rng.randn() * step_size * 0.5
            c = np.cos(angle)
            s = np.sin(angle)
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
            R_inv = np.eye(3) - s * K + (1 - c) * (K @ K)
            
            new_pts = current_pts.copy()
            new_pts[idx1] = new_pts[idx1] @ R.T
            new_pts[idx2] = new_pts[idx2] @ R_inv.T
            new_pts[idx1] = new_pts[idx1] / np.linalg.norm(new_pts[idx1])
            new_pts[idx2] = new_pts[idx2] / np.linalg.norm(new_pts[idx2])
        
        new_energy = _energy_fast(new_pts)
        delta_E = new_energy - current_energy
        
        # Temperature based on progress
        t_frac = min(1.0, (time.time() - start_time) / time_budget)
        T = T_start * (T_end / T_start) ** t_frac
        
        accepted = False
        if delta_E < 0 or (T > 0 and rng.rand() < math.exp(-delta_E / T)):
            current_pts = new_pts
            current_energy = new_energy
            accepted = True
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        
        # Adaptive step size
        accept_count += accepted
        total_count += 1
        if total_count % window == 0:
            rate = accept_count / window
            if rate > 0.6:
                step_size *= 1.1
            elif rate < 0.2:
                step_size *= 0.9
            step_size = np.clip(step_size, 0.01, 2.0)
            accept_count = 0
            total_count = 0
    
    return best_pts, best_energy


def _lbfgs_refine(pts, time_budget):
    """
    Refine points using L-BFGS-B with tangent space gradients.
    """
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_budget:
            return float('inf')
        pts = _project_and_normalize(x.reshape(-1, 3))
        return _energy_fast(pts)
    
    def grad(x):
        if time.time() - start_time > time_budget:
            return np.zeros_like(x)
        pts = _project_and_normalize(x.reshape(-1, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient onto tangent space
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g_tangent.flatten()
    
    x0 = pts.flatten()
    
    res = minimize(
        objective, x0, jac=grad, method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
    )
    
    cand = _project_and_normalize(res.x.reshape(-1, 3))
    cand_e = _energy_fast(cand)
    
    return cand, cand_e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start from Fibonacci spiral
    x0 = _fibonacci_sphere(N)
    
    best_pts = x0.copy()
    best_energy = _energy_fast(best_pts)
    
    # Phase 1: L-BFGS-B refinement from Fibonacci start
    if time.time() < deadline - 40:
        try:
            cand, cand_e = _lbfgs_refine(best_pts, time_budget=25.0)
            if cand_e < best_energy:
                best_pts = cand
                best_energy = cand_e
        except Exception:
            pass
    
    # Phase 2: Simulated Annealing for global exploration
    sa_budget = min(50.0, max(10.0, deadline - time.time() - 30.0))
    if sa_budget > 10:
        try:
            sa_pts, sa_e = _simulated_annealing(best_pts, sa_budget)
            if sa_e < best_energy:
                best_pts = sa_pts
                best_energy = sa_e
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish
    if time.time() < deadline - 5:
        try:
            remaining = deadline - time.time() - 5
            if remaining > 5:
                cand, cand_e = _lbfgs_refine(best_pts, time_budget=remaining)
                if cand_e < best_energy:
                    best_pts = cand
                    best_energy = cand_e
        except Exception:
            pass
    
    # Final validation
    result = _project_and_normalize(best_pts.reshape(-1, 3))
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                result[i] += np.random.normal(0, 1e-8, 3)
                result[i] /= np.linalg.norm(result[i])
    
    # Ensure non-finite values are handled
    if not np.all(np.isfinite(result)):
        result = _fibonacci_sphere(N)
    
    return result
# EVOLVE-BLOCK-END
