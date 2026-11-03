"""Improved program using Voronoi-based local search and L-BFGS polishing."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, Delaunay

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _get_voronoi_centroids(centers):
    """
    Compute the centroids of the Voronoi cells for each point,
    clipped to the square [-0.5, 0.5]^2.
    """
    centroids = centers.copy()
    try:
        vor = Voronoi(centers)
        # Map point indices to region indices
        point_to_region = vor.point_region
        # Get regions
        regions = vor.regions
        
        # Square bounds
        min_x, min_y = -0.5, -0.5
        max_x, max_y = 0.5, 0.5
        
        for i in range(N):
            region_idx = point_to_region[i]
            if region_idx == -1:
                continue
            
            region = regions[region_idx]
            if len(region) == 0:
                continue
            
            # Get vertices of the cell
            verts = vor.vertices[region]
            
            # Clip the polygon to the square
            # Simple clipping: keep only vertices inside, and add intersection points?
            # For simplicity and speed, we can just use the vertices that are inside.
            # This might bias the centroid if the cell is large and clipped.
            # A more robust way: Use the intersection of the convex hull of vertices + square corners?
            # Let's just use the vertices inside the square. If the cell extends outside, 
            # the centroid of the *visible* part is a reasonable approximation for the target.
            
            mask = (verts[:, 0] >= min_x) & (verts[:, 0] <= max_x) & \
                   (verts[:, 1] >= min_y) & (verts[:, 1] <= max_y)
            
            if np.sum(mask) < 3:
                # If too few vertices inside, fallback to center or incenter approx
                # Just keep current position or move towards center of square?
                # Better: move towards the center of the square if on boundary?
                # For now, skip update for this point
                continue
                
            clipped_verts = verts[mask]
            centroid = np.mean(clipped_verts, axis=0)
            centroids[i] = centroid
            
    except Exception:
        # If Voronoi fails, return original centers (no update)
        pass
        
    return centroids

def _optimize_voronoi_local_search(start_centers, steps=100, alpha=0.5):
    """
    Perform local search by moving centers towards their Voronoi cell centroids.
    """
    centers = start_centers.copy()
    for _ in range(steps):
        centroids = _get_voronoi_centroids(centers)
        # Move towards centroids
        centers = centers + alpha * (centroids - centers)
        # Clip to square
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers

def _polish_with_lbfgs(centers):
    """
    Polish the centers using L-BFGS with a soft-min objective.
    """
    def objective(params, k):
        c = params.reshape(N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_terms = dists / 2.0
        wall_dists = 0.5 - np.abs(c)
        all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
        all_terms = all_terms[np.isfinite(all_terms)]
        if len(all_terms) == 0:
            return 0.0
        m = np.min(all_terms)
        shifted = all_terms - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use a high k for sharpness
    k = 1000
    
    try:
        result = minimize(
            objective, 
            centers.reshape(-1), 
            args=(k,),
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        return result.x.reshape(N, 2)
    except Exception:
        return centers

def construct():
    # Initialize with Hexagonal Grid
    s = 0.15
    points = []
    range_x = np.arange(-0.8, 0.8, s)
    range_y = np.arange(-0.8, 0.8, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    
    if len(points) < N:
        # Fallback to random if grid is too small (shouldn't happen)
        np.random.seed(42)
        points = np.random.uniform(-0.49, 0.49, (N, 2))
    else:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
        
    # Step 1: Voronoi Local Search
    # Run multiple rounds with different alphas or just one long run
    centers = _optimize_voronoi_local_search(points, steps=100, alpha=0.5)
    
    # Step 2: Polish with L-BFGS
    centers = _polish_with_lbfgs(centers)
    
    # Final Safety Clip
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers
