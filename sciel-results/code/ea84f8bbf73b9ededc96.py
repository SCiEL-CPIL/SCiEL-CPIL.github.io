import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    def _min_triangle_area(points):
        """Compute the minimum area of any triangle formed by 3 points."""
        n = len(points)
        min_area = float('inf')
        # Vectorized computation for speed
        # i, j, k are indices
        for i in range(n):
            for j in range(i + 1, n):
                v1 = points[j] - points[i]
                # Compute cross product magnitude for all k > j
                # v2 = points[k] - points[i]
                # cross = v1[0]*v2[1] - v1[1]*v2[0]
                v2s = points[j+1:] - points[i]
                cross = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
                areas = 0.5 * np.abs(cross)
                if len(areas) > 0:
                    local_min = np.min(areas)
                    if local_min < min_area:
                        min_area = local_min
        return min_area

    def _convex_hull_area(points):
        """Compute the area of the convex hull of the points."""
        if len(points) < 3:
            return 0.0
        try:
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is area
        except Exception:
            return 0.0

    def _score(points):
        """Compute the score: min triangle area / convex hull area."""
        hull_area = _convex_hull_area(points)
        if hull_area <= 1e-12:
            return 0.0
        min_tri = _min_triangle_area(points)
        return min_tri / hull_area

    def _build_symmetric_points(r_inner, r_outer, phase):
        """
        Build 13 points from 3 concentric regular polygons.
        Structure: 1 center, 6 inner, 6 outer.
        """
        points = []
        # Center
        points.append([0.0, 0.0])
        
        # Inner hexagon (6 points)
        angles_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        for a in angles_inner:
            points.append([r_inner * np.cos(a), r_inner * np.sin(a)])
            
        # Outer hexagon (6 points) with phase shift
        angles_outer = np.linspace(phase, 2 * np.pi + phase, 6, endpoint=False)
        for a in angles_outer:
            points.append([r_outer * np.cos(a), r_outer * np.sin(a)])
            
        return np.array(points)

    def _build_symmetric_points_5_7(r_inner, r_outer, phase):
        """
        Alternative structure: 1 center, 5 inner, 7 outer.
        """
        points = []
        points.append([0.0, 0.0])
        
        # Inner pentagon (5 points)
        angles_inner = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        for a in angles_inner:
            points.append([r_inner * np.cos(a), r_inner * np.sin(a)])
            
        # Outer heptagon (7 points) with phase shift
        angles_outer = np.linspace(phase, 2 * np.pi + phase, 7, endpoint=False)
        for a in angles_outer:
            points.append([r_outer * np.cos(a), r_outer * np.sin(a)])
            
        return np.array(points)

    def _neg_score_full(points_flat):
        """Negative score for full 26-dim optimization."""
        points = points_flat.reshape(-1, 2)
        return -_score(points)

    def _optimize_full_space(start_points, max_iter=200):
        """
        Refine a configuration using Nelder-Mead on the full 26-dimensional space.
        """
        try:
            # Nelder-Mead is better for non-smooth functions like min-triangle-area
            res = opt.minimize(
                _neg_score_full,
                start_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 5000}
            )
            if res.fun < 0:
                return res.x.reshape(-1, 2)
            return start_points
        except Exception:
            return start_points

    best_points = None
    best_score = -float('inf')
    
    # Strategy 1: Symmetric 1-6-6 configuration
    # Parameters: r_inner, r_outer, phase
    # r_inner in [0.1, 0.8], r_outer in [0.9, 1.5], phase in [0, pi/3]
    bounds_66 = [(0.1, 0.8), (0.9, 1.5), (0, np.pi/3)]
    
    if time.time() < start_time + time_limit - 20:
        try:
            # Use Differential Evolution for global search in reduced space
            def obj_66(params):
                r_in, r_out, ph = params
                pts = _build_symmetric_points(r_in, r_out, ph)
                return -_score(pts)
            
            de_res_66 = opt.differential_evolution(
                obj_66, bounds_66, seed=42, maxiter=50, tol=1e-4, polish=True, update_interval=5
            )
            
            if de_res_66.fun < 0:
                r_in, r_out, ph = de_res_66.x
                pts = _build_symmetric_points(r_in, r_out, ph)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    # Strategy 2: Symmetric 1-5-7 configuration
    bounds_57 = [(0.1, 0.8), (0.9, 1.5), (0, 2*np.pi/7)]
    
    if time.time() < start_time + time_limit - 20:
        try:
            def obj_57(params):
                r_in, r_out, ph = params
                pts = _build_symmetric_points_5_7(r_in, r_out, ph)
                return -_score(pts)
            
            de_res_57 = opt.differential_evolution(
                obj_57, bounds_57, seed=43, maxiter=50, tol=1e-4, polish=True, update_interval=5
            )
            
            if de_res_57.fun < 0:
                r_in, r_out, ph = de_res_57.x
                pts = _build_symmetric_points_5_7(r_in, r_out, ph)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
        except Exception:
            pass

    # Strategy 3: Full-space refinement of the best symmetric candidate
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            # Refine in full space to potentially break symmetry and improve
            refined = _optimize_full_space(best_points, max_iter=300)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
        except Exception:
            pass

    # Strategy 4: If still time, try a few random full-space restarts with local polish
    if time.time() < start_time + time_limit - 10:
        np.random.seed(42)
        for i in range(3):
            if time.time() > start_time + time_limit - 10:
                break
            
            # Create a random perturbed version of the best known good structure
            # Or just a random convex-ish set
            # Let's try perturbing the best symmetric one
            if best_points is not None:
                noise = np.random.normal(0, 0.1, best_points.shape)
                perturbed = best_points + noise
            else:
                # Fallback: regular 13-gon
                angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
                perturbed = np.column_stack([np.cos(angles), np.sin(angles)])
            
            s = _score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed.copy()
            
            # Quick local polish
            if time.time() < start_time + time_limit - 5:
                refined = _optimize_full_space(perturbed, max_iter=100)
                s = _score(refined)
                if s > best_score:
                    best_score = s
                    best_points = refined.copy()

    # Final safety check
    if best_points is None:
        # Fallback to regular 13-gon
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
