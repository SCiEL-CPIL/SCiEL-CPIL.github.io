import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations, product

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            # Check for collinearity or degenerate hull
            if np.allclose(pts[:, 0], pts[:, 0][0]) or np.allclose(pts[:, 1], pts[:, 1][0]):
                return 0.0
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _generate_ring_configuration(radii, offsets, counts):
        """
        Generate points from ring parameters.
        radii: list of radii for each ring
        offsets: list of angular offsets for each ring
        counts: list of number of points in each ring
        """
        points = []
        for r, off, n in zip(radii, offsets, counts):
            if n == 0:
                continue
            if n == 1:
                # Single point: place at angle off
                points.append([r * np.cos(off), r * np.sin(off)])
            else:
                angles = np.linspace(off, 2 * np.pi + off, n, endpoint=False)
                xs = r * np.cos(angles)
                ys = r * np.sin(angles)
                ring_pts = np.column_stack([xs, ys])
                points.append(ring_pts)
        
        if not points:
            return np.zeros((N, 2))
            
        pts = np.vstack(points)
        # Ensure exactly N points (should be by construction of counts)
        if len(pts) != N:
            # Fallback if logic error in counts
            return np.vstack([_regular_polygon(N, 1.0)])
        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset, 2 * np.pi + offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _objective_from_params(params, counts):
        """
        Objective function for optimization of ring parameters.
        params: [r1, r2, ..., rk, off1, off2, ..., offk]
        counts: list of point counts for each ring
        """
        k = len(counts)
        radii = params[:k]
        offsets = params[k:2*k]
        
        # Constrain radii to be positive and sorted (outer to inner)
        # We optimize radii in [0, 1] and sort them? 
        # Better: assume radii are given in order outer-to-inner, but allow any order?
        # To maintain symmetry and avoid swapping rings, we fix the order: 
        # ring 0 is outermost, ring k-1 is innermost.
        # So r1 > r2 > ... > rk.
        # We can enforce this by using softplus or just letting the optimizer handle it,
        # but sorting breaks differentiability.
        # Instead, we optimize radii and take absolute values? No, that loses sign info.
        # Let's just optimize radii in [0, 2] and normalize.
        
        # Ensure radii are non-negative
        radii = np.abs(radii)
        
        # If radii are not sorted, the "ring" structure is ambiguous.
        # However, since we are just generating points, the labels don't matter for the score.
        # But for optimization stability, it's better if the geometry is consistent.
        # Let's just generate points. The normalization handles scale.
        
        pts = _generate_ring_configuration(radii, offsets, counts)
        pts = _normalize_points(pts)
        
        # Compute score
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        
        # Add a penalty if points are too close to each other (degenerate)
        # Compute min distance between distinct points
        # This is expensive, so we do it only if min_area is very small?
        # For now, just return -min_area.
        
        # To help optimization, add a small regularization to keep points spread out?
        # No, the min triangle area handles that.
        
        return -min_area

    def _optimize_configuration(counts, initial_radii, initial_offsets, max_iter=50):
        """
        Optimize a specific ring configuration.
        """
        k = len(counts)
        initial_params = np.concatenate([initial_radii, initial_offsets])
        
        # Define bounds
        # Radii: [0, 2] (will be normalized)
        # Offsets: [0, 2*pi]
        bounds = [(0, 2)] * k + [(0, 2 * np.pi)] * k
        
        try:
            res = opt.minimize(
                lambda x: _objective_from_params(x, counts),
                initial_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            best_params = res.x
            best_score = -res.fun
        except Exception:
            # Fallback to initial guess
            best_params = initial_params
            best_score = -_objective_from_params(initial_params, counts)
            
        pts = _generate_ring_configuration(best_params[:k], best_params[k:], counts)
        pts = _normalize_points(pts)
        return pts, best_score

    def _refine_full_points(points, max_iter=100):
        """
        Final local refinement using coordinate descent on the full 2D coordinates.
        """
        pts = _normalize_points(points.copy())
        best_score = compute_score(pts)
        
        # Simple local search: move one point at a time
        step_sizes = [0.05, 0.02, 0.01, 0.005]
        
        for _ in range(max_iter):
            if time.time() > start_time + time_limit - 5:
                break
            improved = False
            for i in range(N):
                # Try moving point i in 8 directions
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        for step in step_sizes:
                            trial = pts.copy()
                            trial[i, 0] += dx * step
                            trial[i, 1] += dy * step
                            trial_norm = _normalize_points(trial)
                            s = compute_score(trial_norm)
                            if s > best_score + 1e-8:
                                pts = trial_norm
                                best_score = s
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return pts, best_score

    # Define diverse structural configurations
    # Each config is a list of counts for concentric rings
    # We assume the first ring is outermost.
    configurations = [
        # Single ring (regular 13-gon)
        [13],
        # 2 rings
        [12, 1], [11, 2], [10, 3], [9, 4], [8, 5], [7, 6], [6, 7], [5, 8], [4, 9], [3, 10], [2, 11], [1, 12],
        # 3 rings
        [10, 2, 1], [9, 3, 1], [8, 4, 1], [7, 5, 1], [6, 6, 1],
        [9, 2, 2], [8, 3, 2], [7, 4, 2], [6, 5, 2],
        [8, 2, 3], [7, 3, 3], [6, 4, 3], [5, 5, 3],
        [7, 2, 4], [6, 3, 4], [5, 4, 4],
        [6, 2, 5], [5, 3, 5], [4, 4, 5],
        [5, 2, 6], [4, 3, 6], [3, 4, 6],
        [4, 2, 7], [3, 3, 7], [2, 4, 7],
        [3, 2, 8], [2, 3, 8],
        [2, 2, 9],
        [1, 2, 10],
        # 4 rings
        [9, 2, 1, 1], [8, 3, 1, 1], [7, 4, 1, 1], [6, 5, 1, 1],
        [8, 2, 2, 1], [7, 3, 2, 1], [6, 4, 2, 1],
        [7, 2, 3, 1], [6, 3, 3, 1],
        [6, 2, 4, 1], [5, 3, 4, 1],
        [5, 2, 5, 1],
        [7, 2, 2, 2], [6, 3, 2, 2],
        [6, 2, 3, 2], [5, 3, 3, 2],
        [5, 2, 4, 2],
        [6, 2, 2, 3], [5, 3, 2, 3],
        [5, 2, 3, 3],
        [4, 2, 4, 3],
        [7, 2, 2, 2], # duplicate check? No, 7+2+2+2=13
        [6, 2, 2, 3], # duplicate
        # 5 rings
        [8, 2, 1, 1, 1], [7, 3, 1, 1, 1], [6, 4, 1, 1, 1],
        [7, 2, 2, 1, 1], [6, 3, 2, 1, 1],
        [6, 2, 3, 1, 1], [5, 3, 3, 1, 1],
        [5, 2, 4, 1, 1],
        [6, 2, 2, 2, 1], [5, 3, 2, 2, 1],
        [5, 2, 3, 2, 1],
        [4, 2, 4, 2, 1],
        [5, 2, 2, 3, 1],
        [4, 2, 3, 3, 1],
        [5, 2, 2, 2, 2],
        # 6 rings
        [7, 2, 1, 1, 1, 1], [6, 3, 1, 1, 1, 1],
        [6, 2, 2, 1, 1, 1], [5, 3, 2, 1, 1, 1],
        [5, 2, 3, 1, 1, 1],
        [4, 2, 4, 1, 1, 1],
        [6, 2, 2, 2, 1, 1],
        [5, 2, 3, 2, 1, 1],
        [5, 2, 2, 3, 1, 1],
        [4, 2, 4, 2, 1, 1],
        [5, 2, 2, 2, 2, 1],
        [4, 2, 3, 3, 2, 1],
        # 7 rings
        [6, 2, 1, 1, 1, 1, 1], [5, 3, 1, 1, 1, 1, 1],
        [5, 2, 2, 1, 1, 1, 1], [4, 3, 2, 1, 1, 1, 1],
        [4, 2, 3, 1, 1, 1, 1],
        [3, 2, 4, 1, 1, 1, 1],
        [5, 2, 2, 2, 1, 1, 1],
        [4, 2, 3, 2, 1, 1, 1],
        [4, 2, 2, 3, 1, 1, 1],
        [3, 2, 4, 2, 1, 1, 1],
        [4, 2, 2, 2, 2, 1, 1],
        [3, 2, 3, 3, 2, 1, 1],
        # 8 rings
        [5, 2, 1, 1, 1, 1, 1, 1], [4, 3, 1, 1, 1, 1, 1, 1],
        [4, 2, 2, 1, 1, 1, 1, 1], [3, 3, 2, 1, 1, 1, 1, 1],
        [3, 2, 3, 1, 1, 1, 1, 1],
        [2, 2, 4, 1, 1, 1, 1, 1],
        [4, 2, 2, 2, 1, 1, 1, 1],
        [3, 2, 3, 2, 1, 1, 1, 1],
        [3, 2, 2, 3, 1, 1, 1, 1],
        [2, 2, 4, 2, 1, 1, 1, 1],
        [3, 2, 2, 2, 2, 1, 1, 1],
        [2, 2, 3, 3, 2, 1, 1, 1],
        # 9 rings
        [4, 2, 1, 1, 1, 1, 1, 1, 1], [3, 3, 1, 1, 1, 1, 1, 1, 1],
        [3, 2, 2, 1, 1, 1, 1, 1, 1], [2, 3, 2, 1, 1, 1, 1, 1, 1],
        [2, 2, 3, 1, 1, 1, 1, 1, 1],
        [1, 2, 4, 1, 1, 1, 1, 1, 1],
        [3, 2, 2, 2, 1, 1, 1, 1, 1],
        [2, 2, 3, 2, 1, 1, 1, 1, 1],
        [2, 2, 2, 3, 1, 1, 1, 1, 1],
        [1, 2, 4, 2, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 1, 1, 1, 1],
        [1, 2, 3, 3, 2, 1, 1, 1, 1],
        # 10 rings
        [3, 2, 1, 1, 1, 1, 1, 1, 1, 1], [2, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1, 1, 1], [1, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 4, 1, 1, 1, 1, 1, 1, 1], # 0 not allowed?
        [2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 2, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 3, 1, 1, 1, 1, 1, 1],
        [0, 2, 4, 2, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1, 1, 1, 1, 1],
        [0, 2, 3, 3, 2, 1, 1, 1, 1, 1],
        # 11 rings
        [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1], [0, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
        [0, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1],
        # 12 rings
        [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1],
        # 13 rings
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    # Filter out invalid configurations (sum != 13 or any count < 1)
    valid_configs = []
    for cfg in configurations:
        if sum(cfg) == N and all(c >= 1 for c in cfg):
            # Remove duplicates
            if cfg not in valid_configs:
                valid_configs.append(cfg)

    # Sort configurations by complexity (fewer rings first, then larger outer rings)
    # This ensures we find a good baseline quickly.
    valid_configs.sort(key=lambda c: (len(c), -c[0]))

    best_score = -float('inf')
    best_points = None

    # Initial guess for radii: evenly spaced
    # Initial guess for offsets: 0
    
    for cfg in valid_configs:
        if time.time() > start_time + time_limit - 10:
            break
            
        k = len(cfg)
        # Initial radii: 1.0, 0.8, 0.6, ...
        initial_radii = np.linspace(1.0, 0.1, k)
        initial_offsets = np.zeros(k)
        
        # Optimize
        pts, score = _optimize_configuration(cfg, initial_radii, initial_offsets, max_iter=30)
        
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
        # If we have time, refine the best points found so far
        if time.time() < start_time + time_limit - 20:
            refined_pts, refined_score = _refine_full_points(pts, max_iter=20)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()

    # Final global refinement of the best points found
    if best_points is not None:
        if time.time() < start_time + time_limit - 5:
            final_pts, final_score = _refine_full_points(best_points, max_iter=100)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()

    # Fallback
    if best_points is None:
        best_points = _normalize_points(_regular_polygon(N, 1.0))

    return best_points
