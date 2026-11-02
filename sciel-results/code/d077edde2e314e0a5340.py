"""Rewritten program for Heilbronn convex regions n=14.
Strategy: Log-barrier constrained optimization in Cartesian coordinates with
convexity enforcement, combined with multiple optimization methods and restarts.
"""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.spatial import ConvexHull
from itertools import combinations
from scipy.optimize import LinearConstraint


def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_n_gon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = 2 * np.pi * np.arange(n) / n
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


def _is_strictly_convex(points, tol=1e-10):
    """Check if points form a strictly convex polygon (in order)."""
    n = len(points)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cross < tol:
            return False
    return True


def _convexity_penalty(points, barrier_width=0.01):
    """Compute a penalty for non-convexity using log-barrier style.
    Returns 0 if strictly convex, positive otherwise.
    """
    n = len(points)
    penalty = 0.0
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if cross < barrier_width:
            # Log barrier: -log(cross - epsilon) for cross > epsilon, else large penalty
            if cross > 1e-12:
                penalty += -np.log(cross / barrier_width) * 0.1
            else:
                penalty += 100.0
    return penalty


def _soft_min_triangle_area(points, temperature=0.01):
    """Compute a smooth approximation of the minimum triangle area.
    Uses soft-min: -temperature * log(sum(exp(-area_i / temperature)))
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    # Soft min
    min_val = np.min(areas)
    # Use log-sum-exp trick for numerical stability
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_val + temperature * log_sum
    return soft_min


def _cartesian_to_flat(points):
    """Flatten 2D points to 1D array."""
    return points.flatten()


def _flat_to_cartesian(flat):
    """Reshape 1D array to 2D points."""
    return flat.reshape(-1, 2)


def _objective_cartesian(flat, temperature=0.01, barrier_width=0.01):
    """Objective function: negative score with convexity penalty."""
    points = _flat_to_cartesian(flat)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e6
    
    min_area = _min_triangle_area(points)
    score = min_area / hull_area
    
    # Add convexity penalty
    penalty = _convexity_penalty(points, barrier_width)
    
    return -score + penalty


def _objective_cartesian_smooth(flat, temperature=0.01, barrier_width=0.01):
    """Objective function with smooth min for better gradients."""
    points = _flat_to_cartesian(flat)
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 1e6
    
    soft_min_area = _soft_min_triangle_area(points, temperature)
    score = soft_min_area / hull_area
    
    penalty = _convexity_penalty(points, barrier_width)
    
    return -score + penalty


def _angular_parameterization(params):
    """Convert angular/radial parameters to points.
    params: [r1, r2, ..., rN, theta1, theta2, ..., thetaN]
    Points are ordered by angle to ensure convexity.
    """
    n = N
    radii = params[:n]
    thetas = params[n:]
    
    # Sort by angle to maintain convex ordering
    order = np.argsort(thetas)
    radii_sorted = radii[order]
    thetas_sorted = thetas[order]
    
    # Make thetas sorted and in [0, 2*pi)
    thetas_sorted = np.sort(thetas) % (2 * np.pi)
    # Re-sort radii according to sorted thetas
    thetas_original_sorted = np.sort(thetas % (2 * np.pi))
    # Map back
    radii_final = np.zeros(n)
    for i in range(n):
        idx = np.where(np.isclose(thetas % (2 * np.pi), thetas_original_sorted[i]))[0]
        if len(idx) > 0:
            radii_final[i] = radii[idx[0]]
    
    xs = radii_final * np.cos(thetas_original_sorted)
    ys = radii_final * np.sin(thetas_original_sorted)
    return np.column_stack([xs, ys])


def _polar_to_cartesian_ordered(params):
    """Convert polar parameters to ordered convex points.
    params: [r1, r2, ..., rN, dtheta1, dtheta2, ..., dthetaN]
    where dtheta_i are angular spacings that sum to 2*pi.
    """
    n = N
    radii = params[:n]
    dthetas = params[n:]
    
    # Ensure dthetas are positive and sum to 2*pi
    dthetas = np.abs(dthetas) + 1e-6
    dthetas = dthetas / np.sum(dthetas) * 2 * np.pi
    
    # Cumulative angles
    angles = np.cumsum(np.concatenate([[0], dthetas[:-1]]))
    
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    return np.column_stack([xs, ys])


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 135s deadline, 15s safety margin
    
    # Start with regular 14-gon
    initial_points = _regular_n_gon(N, radius=1.0)
    best_points = initial_points.copy()
    best_score = _score(best_points)
    
    # Method 1: Differential evolution in Cartesian coordinates
    try:
        if time.time() < deadline - 30:
            def _de_objective(flat):
                points = _flat_to_cartesian(flat)
                # Check convexity
                if not _is_strictly_convex(points, tol=1e-8):
                    return 1e6
                hull_area = _convex_hull_area(points)
                if hull_area < 1e-12:
                    return 1e6
                min_area = _min_triangle_area(points)
                return -(min_area / hull_area)
            
            bounds = [(-2, 2)] * (2 * N)
            
            result = differential_evolution(
                _de_objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                tol=1e-5,
                polish=False,
                update_interval=5
            )
            
            if result.fun < -best_score:
                candidate = _flat_to_cartesian(result.x)
                if _is_strictly_convex(candidate, tol=1e-8):
                    best_points = candidate
                    best_score = -result.fun
    except Exception:
        pass
    
    # Method 2: Local refinement with L-BFGS-B
    try:
        if time.time() < deadline - 20:
            # Use the best points found so far
            flat_init = _cartesian_to_flat(best_points)
            
            def _lbfgs_objective(flat):
                points = _flat_to_cartesian(flat)
                if not _is_strictly_convex(points, tol=1e-8):
                    return 1e6
                hull_area = _convex_hull_area(points)
                if hull_area < 1e-12:
                    return 1e6
                min_area = _min_triangle_area(points)
                return -(min_area / hull_area)
            
            bounds = [(-3, 3)] * (2 * N)
            
            result = minimize(
                _lbfgs_objective,
                flat_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < -best_score:
                candidate = _flat_to_cartesian(result.x)
                if _is_strictly_convex(candidate, tol=1e-8):
                    best_points = candidate
                    best_score = -result.fun
    except Exception:
        pass
    
    # Method 3: Nelder-Mead on polar parameters (angular spacings)
    try:
        if time.time() < deadline - 15:
            # Initialize with regular polygon in polar form
            angles_init = 2 * np.pi * np.arange(N) / N
            dthetas_init = np.diff(np.concatenate([angles_init, [angles_init[0] + 2 * np.pi]]))
            radii_init = np.ones(N)
            polar_init = np.concatenate([radii_init, dthetas_init])
            
            def _polar_objective(params):
                points = _polar_to_cartesian_ordered(params)
                if not _is_strictly_convex(points, tol=1e-8):
                    return 1e6
                hull_area = _convex_hull_area(points)
                if hull_area < 1e-12:
                    return 1e6
                min_area = _min_triangle_area(points)
                return -(min_area / hull_area)
            
            result = minimize(
                _polar_objective,
                polar_init,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < -best_score:
                candidate = _polar_to_cartesian_ordered(result.x)
                if _is_strictly_convex(candidate, tol=1e-8):
                    best_points = candidate
                    best_score = -result.fun
    except Exception:
        pass
    
    # Method 4: Multiple random restarts with local optimization
    try:
        if time.time() < deadline - 10:
            rng = np.random.RandomState(123)
            
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                
                # Generate random convex polygon
                # Random angles (sorted) and random radii
                angles = np.sort(rng.uniform(0, 2 * np.pi, N))
                # Ensure minimum angular separation
                min_sep = 2 * np.pi / N * 0.5
                for i in range(N):
                    if angles[(i + 1) % N] - angles[i] < min_sep:
                        angles[(i + 1) % N] = angles[i] + min_sep
                angles = angles % (2 * np.pi)
                angles = np.sort(angles)
                
                radii = rng.uniform(0.5, 1.5, N)
                
                xs = radii * np.cos(angles)
                ys = radii * np.sin(angles)
                points = np.column_stack([xs, ys])
                
                if not _is_strictly_convex(points, tol=1e-8):
                    continue
                
                flat_init = _cartesian_to_flat(points)
                
                def _restart_objective(flat):
                    pts = _flat_to_cartesian(flat)
                    if not _is_strictly_convex(pts, tol=1e-8):
                        return 1e6
                    hull_area = _convex_hull_area(pts)
                    if hull_area < 1e-12:
                        return 1e6
                    min_area = _min_triangle_area(pts)
                    return -(min_area / hull_area)
                
                result = minimize(
                    _restart_objective,
                    flat_init,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                
                if result.fun < -best_score:
                    candidate = _flat_to_cartesian(result.x)
                    if _is_strictly_convex(candidate, tol=1e-8):
                        best_points = candidate
                        best_score = -result.fun
    except Exception:
        pass
    
    # Method 5: Final polish with smooth objective
    try:
        if time.time() < deadline - 5:
            flat_init = _cartesian_to_flat(best_points)
            
            result = minimize(
                _objective_cartesian_smooth,
                flat_init,
                method='L-BFGS-B',
                bounds=[(-3, 3)] * (2 * N),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < -best_score:
                candidate = _flat_to_cartesian(result.x)
                if _is_strictly_convex(candidate, tol=1e-8):
                    best_points = candidate
                    best_score = -result.fun
    except Exception:
        pass
    
    # Final validation
    if _convex_hull_area(best_points) < 1e-10 or not _is_strictly_convex(best_points, tol=1e-8):
        best_points = _regular_n_gon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
