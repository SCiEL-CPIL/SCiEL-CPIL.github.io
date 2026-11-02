"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area_normalized(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    
    if hull_area <= 0:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Triangle area = 0.5 * |cross product|
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _negative_score(pts):
    """Objective function to minimize (negative of score)."""
    score = _min_triangle_area_normalized(pts)
    return -score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute triangle indices
    tri_indices = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                tri_indices.append((i, j, k))
    tri_indices = np.array(tri_indices)
    n_tris = len(tri_indices)
    
    def fast_score(pts):
        """Fast computation of min triangle area / hull area."""
        # Convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            return 0.0
        if hull_area <= 0:
            return 0.0
        
        # Vectorized triangle area computation
        i_idx = tri_indices[:, 0]
        j_idx = tri_indices[:, 1]
        k_idx = tri_indices[:, 2]
        
        p_i = pts[i_idx]
        p_j = pts[j_idx]
        p_k = pts[k_idx]
        
        # Cross product magnitude
        cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
                (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area
    
    def neg_score_fast(pts):
        return -fast_score(pts)
    
    # Strategy 1: Fermat spiral initial guess
    def fermat_spiral(n, scale=1.0):
        i = np.arange(n)
        r = scale * np.sqrt(i / n)
        theta = i * 2.399963229728653  # Golden angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack((x, y))
        pts -= np.mean(pts, axis=0)
        return pts
    
    # Strategy 2: Regular polygon + center
    def regular_polygon_plus_center(n):
        # Try different decompositions: center + regular polygons
        best_pts = None
        best_score = 0.0
        
        for center_count in range(1, 3):
            remaining = n - center_count
            # Try to make remaining a regular polygon
            if remaining < 3:
                continue
            
            pts = np.zeros((n, 2))
            # Center points at origin
            for i in range(center_count):
                pts[i] = [0.0, 0.0]
            
            # Regular polygon for remaining
            angles = 2 * np.pi * np.arange(remaining) / remaining
            r = 1.0
            for i in range(remaining):
                pts[center_count + i] = [r * np.cos(angles[i]), r * np.sin(angles[i])]
            
            score = fast_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        
        if best_pts is None:
            best_pts = fermat_spiral(n)
        
        return best_pts
    
    # Generate diverse initial points
    initial_points = []
    
    # Fermat spiral
    initial_points.append(fermat_spiral(N))
    
    # Regular polygon + center
    initial_points.append(regular_polygon_plus_center(N))
    
    # Random points in unit circle
    rng = random.Random(42)
    for _ in range(5):
        pts = np.array([[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(N)])
        pts -= np.mean(pts, axis=0)
        # Normalize to unit area
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area > 0:
                pts *= np.sqrt(1.0 / area)
        except Exception:
            pass
        initial_points.append(pts)
    
    # Evaluate initial points
    best_pts = None
    best_score = 0.0
    
    for pts in initial_points:
        score = fast_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Local optimization using Powell (derivative-free, robust for non-smooth)
    for pts_init in initial_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                neg_score_fast,
                pts_init.flatten(),
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                opt_pts = result.x.reshape(N, 2)
                opt_score = fast_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Additional refinement with L-BFGS-B on best found
    if best_pts is not None and time.time() < deadline:
        try:
            result = minimize(
                neg_score_fast,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if result.fun < 0:
                opt_pts = result.x.reshape(N, 2)
                opt_score = fast_score(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Final multi-start with small perturbations if time remains
    if time.time() < deadline - 20:
        for _ in range(10):
            if time.time() > deadline:
                break
            
            perturbed = best_pts + 0.01 * np.random.randn(N, 2)
            try:
                result = minimize(
                    neg_score_fast,
                    perturbed.flatten(),
                    method='Powell',
                    options={'maxiter': 30, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                if result.fun < 0:
                    opt_pts = result.x.reshape(N, 2)
                    opt_score = fast_score(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # Ensure points are well-formed
    if best_pts is None:
        best_pts = fermat_spiral(N)
    
    # Normalize: center at origin
    best_pts -= np.mean(best_pts, axis=0)
    
    return best_pts

# EVOLVE-BLOCK-END
