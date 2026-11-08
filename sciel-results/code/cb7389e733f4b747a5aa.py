"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import sys

DEADLINE = 130.0  # safety margin before 150s kill


def _voronoi_inradii(centers, W, H):
    """Compute the inradius of each circle given centers in a W x H rectangle."""
    n = len(centers)
    if n == 0:
        return np.array([])
    
    # Build Voronoi diagram
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback: use distance to nearest neighbor / 2
        from scipy.spatial.distance import cdist
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        return 0.5 * np.min(dists, axis=1)
    
    radii = np.zeros(n)
    for i in range(n):
        # Get region vertices for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if len(region) == 0 or -1 in region:
            # Point is on the boundary; inradius is distance to nearest edge
            x, y = centers[i]
            radii[i] = min(x, y, W - x, H - y)
            continue
        
        # Compute inradius as minimum distance from center to each edge
        min_dist = np.inf
        for j in range(len(region)):
            v1 = vor.vertices[region[j]]
            v2 = vor.vertices[region[(j + 1) % len(region)]]
            
            # Distance from point to line segment v1-v2
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length_sq = dx * dx + dy * dy
            if length_sq < 1e-12:
                dist = np.sqrt((centers[i][0] - v1[0])**2 + (centers[i][1] - v1[1])**2)
            else:
                t = np.clip(((centers[i][0] - v1[0]) * dx + (centers[i][1] - v1[1]) * dy) / length_sq, 0, 1)
                proj_x = v1[0] + t * dx
                proj_y = v1[1] + t * dy
                dist = np.sqrt((centers[i][0] - proj_x)**2 + (centers[i][1] - proj_y)**2)
            min_dist = min(min_dist, dist)
        
        # Also check distance to rectangle boundaries
        min_dist = min(min_dist, centers[i][0], centers[i][1], W - centers[i][0], H - centers[i][1])
        radii[i] = min_dist
    
    return radii


def _objective(params, W, H):
    """Negative sum of inradii (we minimize negative sum to maximize sum)."""
    centers = params.reshape(-1, 2)
    
    # Clamp to rectangle
    centers = np.clip(centers, 1e-6, [W - 1e-6, H - 1e-6])
    
    radii = _voronoi_inradii(centers, W, H)
    
    # Check for overlaps (safety)
    from scipy.spatial.distance import cdist
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            min_sep = radii[i] + radii[j]
            if dists[i, j] < min_sep - 1e-6:
                overlap_penalty += (min_sep - dists[i, j]) ** 2
    
    return -(np.sum(radii)) + 10.0 * overlap_penalty


def _repulsion_forces(centers, W, H, radii, k=10.0):
    """Compute repulsion forces between circles."""
    n = len(centers)
    forces = np.zeros_like(centers)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 1e-8:
                d = np.array([1e-8, 0])
                dist = 1e-8
            min_sep = radii[i] + radii[j]
            if dist < min_sep:
                force = k * (min_sep - dist) / dist
                forces[i] += force * d
                forces[j] -= force * d
    
    # Wall repulsion
    for i in range(n):
        for axis, wall in [(0, 0), (0, W), (1, 0), (1, H)]:
            dist_to_wall = centers[i][axis] if wall == 0 else wall - centers[i][axis]
            if dist_to_wall < radii[i]:
                force = k * (radii[i] - dist_to_wall)
                if wall == 0:
                    forces[i][axis] += force
                else:
                    forces[i][axis] -= force
    
    return forces


def _optimize_from_start(centers0, W, H, time_budget):
    """Run L-BFGS-B optimization from a given starting configuration."""
    n = len(centers0)
    bounds = [(1e-6, W - 1e-6)] * n + [(1e-6, H - 1e-6)] * n
    
    best_params = centers0.flatten().copy()
    best_obj = _objective(best_params, W, H)
    
    start_time = time.time()
    try:
        result = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 2000}
        )
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    # Local refinement with repulsion
    centers = best_params.reshape(-1, 2)
    for step in range(50):
        if time.time() - start_time > time_budget:
            break
        radii = _voronoi_inradii(centers, W, H)
        forces = _repulsion_forces(centers, W, H, radii)
        centers = centers + 0.01 * forces
        centers = np.clip(centers, 1e-6, [W - 1e-6, H - 1e-6])
    
    # Final L-BFGS-B polish
    try:
        result = minimize(
            _objective,
            centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        if result.fun < best_obj:
            best_params = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    return best_params.reshape(-1, 2), best_obj


def construct():
    global DEADLINE
    start_time = time.time()
    best_score = 0.0
    best_result = None
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
    
    for aspect in aspect_ratios:
        if time.time() - start_time > DEADLINE:
            break
        
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        # Multiple restarts per aspect ratio
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            time_budget = min(10.0, DEADLINE - (time.time() - start_time) - 5)
            if time_budget < 2:
                break
            
            if r == 0:
                # Grid start
                side = int(np.ceil(np.sqrt(N)))
                centers0 = np.array([
                    [(i + 0.5) / side * W, (j + 0.5) / side * H]
                    for i in range(side) for j in range(side)
                ][:N])
            else:
                # Random start
                np.random.seed(42 + r * 1000 + int(aspect * 100))
                centers0 = np.random.uniform(0.1, [W - 0.1, H - 0.1], size=(N, 2))
            
            centers, obj = _optimize_from_start(centers0, W, H, time_budget)
            
            # Validate and compute score
            radii = _voronoi_inradii(centers, W, H)
            
            # Check validity
            from scipy.spatial.distance import cdist
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, np.inf)
            
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if dists[i, j] < radii[i] + radii[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
                if centers[i][0] < radii[i] - 1e-6 or centers[i][0] > W - radii[i] + 1e-6:
                    valid = False
                    break
                if centers[i][1] < radii[i] - 1e-6 or centers[i][1] > H - radii[i] + 1e-6:
                    valid = False
                    break
                if not valid:
                    break
            
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([centers, radii])
    
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END

