"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = points.shape[0]
    G = points @ points.T
    dist_sq = 2.0 - 2.0 * G
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(n, k=1)
    d_sq_upper = dist_sq[iu]
    
    # Clip to avoid log(0) or negative due to floating point errors
    d_sq_upper = np.maximum(d_sq_upper, 1e-15)
    terms = -0.5 * np.log(d_sq_upper)
    
    return math.fsum(terms)


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1.0, norms)
    return points / norms


def _langevin_dynamics_init(N, steps=2000, T_start=0.5, T_end=1e-4, rng_seed=42):
    """
    Use Langevin dynamics to find a good initial configuration.
    This is a stochastic global search method.
    """
    rng = np.random.RandomState(rng_seed)
    # Random initialization
    pts = rng.randn(N, 3)
    pts = _project_to_sphere(pts)
    
    # Precompute indices for pairs
    iu = np.triu_indices(N, k=1)
    
    for step in range(steps):
        # Linear cooling schedule
        T = T_start * (T_end / T_start) ** (step / steps)
        
        # Compute forces
        # Force on i is sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
        # Vectorized:
        # dist_sq[i,j] = |x_i - x_j|^2
        # We need sum over j for each i.
        
        # Efficient force calculation
        # G = pts @ pts.T
        # dist_sq = 2 - 2*G
        # Force_i = sum_j (x_i - x_j) / dist_sq[i,j]
        # Note: dist_sq[i,j] = |x_i|^2 + |x_j|^2 - 2 x_i.x_j = 2 - 2 G_ij
        
        G = pts @ pts.T
        dist_sq = 2.0 - 2.0 * G
        np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero on diagonal, will mask out
        
        # Vector = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  (N, N, 3)
        # This is memory heavy for N=100? 100*100*3 = 30000 floats. Fine.
        vecs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        
        # Force = sum_j vecs[i,j] / dist_sq[i,j]
        # We want to sum over axis 1 (j)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Divide by dist_sq, careful with 0s on diagonal (masked out)
        # dist_sq is (N, N)
        # vecs is (N, N, 3)
        # We can do: vecs / dist_sq[..., np.newaxis]
        # But diagonal is 0/0. We set dist_sq diag to 1, so it's 0/1 = 0.
        # Then we sum over axis 1.
        
        forces = np.sum(vecs / dist_sq[..., np.newaxis], axis=1)
        
        # Update
        # x_new = x + alpha * F + sqrt(2 * alpha * T) * noise
        alpha = 0.01
        noise = rng.randn(N, 3) * np.sqrt(2 * alpha * T)
        pts = pts + alpha * forces + noise
        
        # Project to sphere
        pts = _project_to_sphere(pts)
        
    return pts


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    
    # 1. Generate multiple candidates using Langevin Dynamics with different seeds
    candidates = []
    for seed in range(5):
        pts = _langevin_dynamics_init(N, steps=1000, T_start=0.2, T_end=1e-5, rng_seed=seed + 100)
        candidates.append(pts)
        
    # Also include a Fibonacci spiral as a fallback
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_pts)
    
    # 2. Refine each candidate using L-BFGS-B
    best_pts = None
    best_energy = float('inf')
    
    def objective(x_flat):
        pts = _project_to_sphere(x_flat.reshape(-1, 3))
        return _log_energy(pts)
        
    for init_pts in candidates:
        x0_flat = init_pts.flatten()
        
        # Optimize
        res = minimize(
            objective,
            x0_flat,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_pts = _project_to_sphere(res.x.reshape(-1, 3))
            
    # 3. Final high-precision refinement on the best point
    if best_pts is not None:
        x_best_flat = best_pts.flatten()
        res_final = minimize(
            objective,
            x_best_flat,
            method='L-BFGS-B',
            options={'maxiter': 20000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
        )
        if res_final.fun < best_energy:
            best_pts = _project_to_sphere(res_final.x.reshape(-1, 3))
            best_energy = res_final.fun

    # Verify validity
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    norms = np.linalg.norm(best_pts, axis=1)
    assert np.all(norms > 1e-9)
    
    # Check for close points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge
                diff = best_pts[j] - best_pts[i]
                if np.linalg.norm(diff) < 1e-15:
                    diff = np.random.randn(3)
                best_pts[j] = best_pts[j] + 1e-8 * diff / np.linalg.norm(diff)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])

    return best_pts
# EVOLVE-BLOCK-END
