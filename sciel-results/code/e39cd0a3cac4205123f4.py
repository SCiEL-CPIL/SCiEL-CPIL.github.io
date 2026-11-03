"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the packing radius for a set of points."""
    if len(points) != N:
        return 0.0
    # Check finite
    if not np.all(np.isfinite(points)):
        return 0.0
    # Compute pairwise distances
    # For 300 points, this is 300*299/2 = 44850 pairs, manageable
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to boundary
    radii = np.sqrt(np.sum(points * points, axis=1))
    min_boundary = 1.0 - np.max(radii)
    return min(min_pair_dist / 2.0, min_boundary)


def _objective(points_flat):
    """Negative of the packing radius for minimization."""
    points = points_flat.reshape(N, 2)
    return -_compute_score(points)


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    # Start with hexagonal lattice initialization
    points = _hexagonal_init()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Try to improve with local optimization
    # Use a multi-start approach with perturbation
    rng = np.random.RandomState(42)
    
    n_attempts = 0
    while time.time() < deadline:
        n_attempts += 1
        # Create a perturbed version
        if n_attempts == 1:
            current = best_points.copy()
        else:
            # Perturb from best
            noise_scale = 0.02 * (1.0 - 0.5 * n_attempts / 50.0)
            noise_scale = max(noise_scale, 0.005)
            current = best_points + rng.randn(N, 2) * noise_scale
        
        # Optimize
        res = minimize(
            _objective,
            current.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.success or res.fun < -best_score:
            new_points = res.x.reshape(N, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Final polish: try a few more iterations from the best
    while time.time() < deadline - 5:
        res = minimize(
            _objective,
            best_points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-10, 'maxls': 50}
        )
        new_points = res.x.reshape(N, 2)
        new_score = _compute_score(new_points)
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
        else:
            break
    
    return best_points


def _hexagonal_init():
    """Create initial hexagonal lattice packing."""
    # Try different spacings to find one that fits N points
    for spacing in [0.11, 0.105, 0.10, 0.095, 0.09, 0.085, 0.08]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by radius and take first N
            idx = np.argsort(radius[mask], kind='stable')[:N]
            return points[idx]
    
    # Fallback: random points
    rng = np.random.RandomState(0)
    points = rng.randn(N, 2)
    radii = np.linalg.norm(points, axis=1)
    points = points / radii[:, np.newaxis] * 0.9
    return points


# EVOLVE-BLOCK-END
