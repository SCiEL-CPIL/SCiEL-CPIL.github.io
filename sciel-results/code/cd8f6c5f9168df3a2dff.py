import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_relax(pts, grid_pts, max_iters=200, deadline=None):
    """
    Lloyd's algorithm (Voronoi relaxation) on the sphere.
    Moves each point to the centroid of its Voronoi cell (approximated by grid points).
    """
    pts = pts.copy()
    G = len(grid_pts)
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        # 1. Assign each grid point to its nearest configuration point
        # dots[i, j] = pts[i] @ grid_pts[j]
        dots = pts @ grid_pts.T
        # For each grid point j, find the index of the nearest point i
        nearest_idx = np.argmax(dots, axis=0)
        
        # 2. Compute centroids
        # For each point i, sum the grid points assigned to it
        # We use a scatter-add approach or bincount equivalent for vectors
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        
        # Efficient scatter add using np.add.at
        np.add.at(new_pts, nearest_idx, grid_pts)
        np.add.at(counts, nearest_idx, 1)
        
        # Avoid division by zero for isolated points (shouldn't happen with dense grid)
        valid_counts = counts > 0
        new_pts[valid_counts] /= counts[valid_counts, np.newaxis]
        
        # Normalize new points to unit sphere
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts[valid_counts] /= norms[valid_counts]
        
        # For points with no assigned grid points (rare), keep old position
        new_pts[~valid_counts] = pts[~valid_counts]
        
        # 3. Damping to ensure stability
        # Move a fraction of the way towards the centroid
        alpha = 0.5
        pts = (1 - alpha) * pts + alpha * new_pts
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
    return pts

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Vectors for the triangle edges
        AB = B - A
        AC = C - A
        
        # Area of triangle: 0.5 * || AB x AC ||
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Handle degenerate facets
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
            
        # Normal vector for each facet
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward (away from origin)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: distance from pole to any vertex (e.g., A)
        # cos(theta) = P @ A
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        return np.max(ang_radii)
    except Exception:
        return np.pi

def _local_polish(pts, deadline):
    """
    Perform high-iteration count of facet-based local optimization.
    Moves the worst facet's vertices towards its pole.
    """
    best_pts = pts.copy()
    best_r = _compute_hull_radius(pts)
    
    # Adaptive step size
    step = 0.02
    no_improve_count = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            M = len(faces)
            
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if not np.any(valid):
                break
                
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            ang_radii = np.arccos(cos_theta)
            
            curr_r = np.max(ang_radii)
            
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 20:
                    step *= 0.95
                    no_improve_count = 0
                    if step < 1e-5:
                        break
            
            # Find worst facet
            worst_idx = np.argmax(ang_radii)
            v_indices = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            for v in v_indices:
                dir_vec = pole - pts[v]
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    dir_vec /= dir_norm
                else:
                    dir_vec = 0.0
                cand[v] = pts[v] + step * dir_vec
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
                
            # Check improvement
            cand_r = _compute_hull_radius(cand)
            if cand_r < curr_r:
                pts = cand
            else:
                # Try smaller step if stuck
                step *= 0.8
                
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 145.0  # Safety margin
    rng = np.random.default_rng(42)
    
    # Create a dense spherical grid for the Lloyd's algorithm
    # G=5000 is dense enough for good Voronoi approximation
    G = 5000
    i = np.arange(G) + 0.5
    z = 1.0 - 2.0 * i / G
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial configuration
    pts = _initial_fib()
    
    # Phase 1: Lloyd's Algorithm (Voronoi Relaxation)
    # This is deterministic and converges to a local minimum of the covering radius
    # much faster than SA for this type of problem.
    pts = _lloyd_relax(pts, grid_pts, max_iters=100, deadline=deadline - 10.0)
    
    # Phase 2: Local Polish using exact hull metric
    # This refines the Lloyd's result to the precision of the convex hull
    if time.time() < deadline - 2.0:
        pts, best_r = _local_polish(pts, deadline)
        
    # Final normalization
    pts = np.array(pts, dtype=float)
    pts /= (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        return _initial_fib()
        
    return pts
