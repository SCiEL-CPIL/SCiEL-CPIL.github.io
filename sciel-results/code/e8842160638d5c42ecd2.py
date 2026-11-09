"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances."""
    # pts: (N, 3)
    n_pts = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    # Get upper triangle indices
    iu = np.triu_indices(n_pts, k=1)
    d_sq = dist_sq[iu]
    # Clip to avoid log(0) and ensure positive
    d_sq = np.clip(d_sq, 1e-30, None)
    d = np.sqrt(d_sq)
    # Sum log(1/d) = -sum(log(d))
    # Using math.fsum for precision
    terms = [-math.log(di) for di in d]
    return math.fsum(terms)


def _log_energy_and_tangent_grad(pts):
    """Compute log(1/d) energy and tangential gradient on the sphere."""
    # pts: (N, 3)
    n_pts = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid zero distances
    dist_sq = np.clip(dist_sq, 1e-30, None)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(n_pts, k=1)
    d = dist[iu]
    energy = -math.fsum([math.log(di) for di in d])
    
    # Gradient in R^3: grad_i = sum_{j != i} (p_j - p_i) / d_ij^2
    # Note: d_ij^2 = dist_sq[i,j]
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = ~np.eye(n_pts, dtype=bool)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad_i = sum_j inv_dist_sq[i,j] * (p_j - p_i)
    # = sum_j inv_dist_sq[i,j] * p_j - p_i * sum_j inv_dist_sq[i,j]
    
    weighted_sum = inv_dist_sq @ pts
    row_sums = inv_dist_sq.sum(axis=1)
    
    # Euclidean gradient
    grad_r3 = weighted_sum - pts * row_sums[:, None]
    
    # Project to tangent space: grad_tan = grad_r3 - (grad_r3 · p_i) p_i
    # Since pts are on unit sphere, p_i · p_i = 1
    dot = np.sum(grad_r3 * pts, axis=1, keepdims=True)
    grad_tan = grad_r3 - dot * pts
    
    return energy, grad_tan


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    return pts / norms


def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_grid():
    """Structured grid-based initialization on sphere."""
    # Use a modified Fibonacci with better distribution
    i = np.arange(N)
    # Use a more even distribution
    z = 1.0 - 2.0 * (i + 0.5) / N
    # Use golden angle with a different phase
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + 0.3)
    r = np.sqrt(np.clip(1.0 - z * z, 0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_clustered_random(rng, n_clusters=10, points_per_cluster=10):
    """Generate random points with some clustering to test robustness."""
    pts = []
    for c in range(n_clusters):
        # Random center
        center = rng.standard_normal(3)
        center = center / np.linalg.norm(center)
        for _ in range(points_per_cluster):
            # Small perturbation around center
            p = center + rng.standard_normal(3) * 0.3
            p = p / np.linalg.norm(p)
            pts.append(p)
    return np.array(pts)


def _optimize_sphere(x0_flat, maxiter=500, ftol=1e-12, gtol=1e-10, deadline=None):
    """Optimize on sphere using L-BFGS-B with tangential gradient.
    
    We parameterize by the 3N coordinates. The objective function projects
    to the sphere. The gradient passed is the tangential gradient.
    """
    def objective(x):
        pts = x.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_tangent_grad(pts)
        return e, g.flatten()
    
    try:
        # Check time if deadline is provided
        if deadline and time.time() > deadline:
            return None, None

        result = minimize(
            objective,
            x0_flat,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxfun': maxiter * 10}
        )
        
        if result.fun is not None and np.isfinite(result.fun):
            result_pts = _project_to_sphere(result.x.reshape(N, 3))
            return result_pts, result.fun
    except Exception:
        pass
    
    return None, None


def _local_refine(pts, rng, n_trials=20, step_sizes=None, deadline=None):
    """Try small random perturbations to find local improvements."""
    if step_sizes is None:
        step_sizes = [0.01, 0.005, 0.001, 0.0005]
    
    best = pts.copy()
    best_e = _log_energy(pts)
    
    for step in step_sizes:
        for _ in range(n_trials):
            if deadline and time.time() > deadline:
                break
            # Random tangent perturbation
            noise = rng.standard_normal((N, 3))
            # Project to tangent space
            dot = np.sum(pts * noise, axis=1, keepdims=True)
            tangent = noise - dot * pts
            # Normalize
            norms = np.linalg.norm(tangent, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-15, None)
            tangent = tangent / norms
            
            candidate = pts + step * tangent
            candidate = _project_to_sphere(candidate)
            
            e = _log_energy(candidate)
            if e < best_e:
                best = candidate
                best_e = e
    
    return best, best_e


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    # 120s limit, leave 15s safety margin
    deadline = start_time + 105
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    inits = [
        _initial_fibonacci(),
        _initial_grid(),
        rng.standard_normal((N, 3)),
        _initial_clustered_random(rng, 10, 10),
        _initial_clustered_random(rng, 5, 20),
    ]
    # Normalize all
    inits = [_project_to_sphere(p) for p in inits]
    
    best_pts = inits[0].copy()
    best_energy = _log_energy(best_pts)
    
    # Optimize from each initial configuration
    for i, init_pts in enumerate(inits):
        if time.time() >= deadline - 20:
            break
        
        init_e = _log_energy(init_pts)
        if init_e < best_energy:
            best_pts = init_pts.copy()
            best_energy = init_e
        
        # Use more iterations for better initial configurations
        maxiter = 400 if i < 2 else 250
        result_pts, result_energy = _optimize_sphere(
            init_pts.flatten(), 
            maxiter=maxiter, 
            ftol=1e-13, 
            gtol=1e-11, 
            deadline=deadline
        )
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy
    
    # Local refinement
    if time.time() < deadline - 10:
        refined_pts, refined_e = _local_refine(
            best_pts, rng, n_trials=30, 
            step_sizes=[0.02, 0.01, 0.005, 0.002, 0.001],
            deadline=deadline
        )
        if refined_e < best_energy:
            best_pts = refined_pts
            best_energy = refined_e
    
    # Final polish with very tight tolerances
    if time.time() < deadline - 5:
        x0 = best_pts.flatten()
        result_pts, result_energy = _optimize_sphere(
            x0, maxiter=300, ftol=1e-15, gtol=1e-14, deadline=deadline
        )
        if result_pts is not None:
            actual_energy = _log_energy(result_pts)
            if actual_energy < best_energy:
                best_pts = result_pts
                best_energy = actual_energy
    
    # One more local refinement pass
    if time.time() < deadline - 2:
        refined_pts, refined_e = _local_refine(
            best_pts, rng, n_trials=15,
            step_sizes=[0.005, 0.002, 0.001, 0.0005],
            deadline=deadline
        )
        if refined_e < best_energy:
            best_pts = refined_pts
            best_energy = refined_e

    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
