"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time
import random

def _compute_score(centers):
    """Compute the score: min of min pairwise distance/2 and min (1 - |c_i|)."""
    n = len(centers)
    if n < 2:
        return 0.0
    # Check finite
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check shape
    if centers.shape != (n, 2):
        return 0.0
    
    # Compute pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0, output_type='set')
    if not pairs:
        min_dist = 2.0  # No pairs means all points are far apart, but that's unlikely
    else:
        min_dist = float('inf')
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
        if min_dist == float('inf'):
            # Fallback: compute all pairwise distances
            from scipy.spatial.distance import pdist
            dists = pdist(centers)
            min_dist = np.min(dists) if len(dists) > 0 else 2.0
    
    # Compute distance to boundary
    radii = np.linalg.norm(centers, axis=1)
    min_boundary = 1.0 - np.max(radii)
    
    if min_boundary < 0:
        return 0.0
    
    return min(min_dist / 2.0, min_boundary)


def _initial_hexagonal(n_points):
    """Generate an initial hexagonal packing."""
    # Use a hexagonal lattice
    points = []
    # Try different spacings
    for spacing in np.linspace(0.05, 0.3, 50):
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= n_points:
            # Select the n_points closest to origin
            radii = np.linalg.norm(points, axis=1)
            idx = np.argsort(radii, kind='stable')[:n_points]
            return points[idx]
    # Fallback: random
    return np.random.randn(n_points, 2) * 0.5


def _relaxation(centers, max_iter=5000, lr=0.01, deadline=None):
    """
    Perform a relaxation to maximize the minimum distance.
    Uses a gradient-free approach with pairwise repulsion.
    """
    centers = centers.copy()
    n = len(centers)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        tree = cKDTree(centers)
        pairs = tree.query_pairs(1.0, output_type='set')
        
        for i, j in pairs:
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Add small random perturbation
                d = np.random.randn(2) * 1e-10
                dist = np.linalg.norm(d)
            # Repulsive force, stronger when closer
            force_mag = 1.0 / (dist ** 2)
            forces[i] += force_mag * d / dist
            forces[j] -= force_mag * d / dist
        
        # Boundary repulsion
        radii = np.linalg.norm(centers, axis=1)
        for i in range(n):
            if radii[i] > 0.9:
                # Push towards center
                forces[i] -= (centers[i] / radii[i]) * (radii[i] - 0.9) ** 2
        
        # Update
        centers += lr * forces
        
        # Project to valid region
        radii = np.linalg.norm(centers, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            centers[mask] = centers[mask] / radii[mask] * 1.0
    
    return centers


def _perturb_and_reoptimize(centers, deadline):
    """Perturb a few points and re-optimize."""
    centers = centers.copy()
    n = len(centers)
    
    # Perturb a random subset
    n_perturb = max(1, n // 5)
    idx = np.random.choice(n, n_perturb, replace=False)
    centers[idx] += np.random.randn(n_perturb, 2) * 0.05
    
    # Re-optimize
    centers = _relaxation(centers, max_iter=2000, lr=0.005, deadline=deadline)
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial hexagonal packing
    centers = _initial_hexagonal(N)
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Main optimization loop
    while time.time() < deadline:
        # Try relaxation from current best
        relaxed = _relaxation(best_centers, max_iter=3000, lr=0.005, deadline=deadline)
        score = _compute_score(relaxed)
        if score > best_score:
            best_score = score
            best_centers = relaxed.copy()
        
        # Try perturbation and re-optimization
        if time.time() < deadline - 10.0:
            perturbed = _perturb_and_reoptimize(best_centers, deadline)
            score = _compute_score(perturbed)
            if score > best_score:
                best_score = score
                best_centers = perturbed.copy()
        
        # Check if we're close to the known best
        if best_score >= 0.1258:
            break
    
    return best_centers
# EVOLVE-BLOCK-END
