"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _convex_hull_area(points):
    """Compute the area of the convex hull of the given points using Andrew's monotone chain."""
    n = len(points)
    if n < 3:
        return 0.0
    
    pts = sorted(points, key=lambda p: (p[0], p[1]))
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    area = abs(area) * 0.5
    return area


def _score(points):
    """Compute the score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-15:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _regular_polygon_displaced(r_displaced, angle_offset, n_displaced=1):
    """
    Generate points as a regular 13-gon with one point displaced radially inward.
    r_displaced: radial distance of the displaced point (0 to 1)
    angle_offset: angular rotation of the entire configuration
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + angle_offset
    points = np.zeros((N, 2))
    
    for i in range(N):
        if i == 0:  # Displace the first point radially inward
            points[i] = [r_displaced * np.cos(angles[i]), r_displaced * np.sin(angles[i])]
        else:
            points[i] = [np.cos(angles[i]), np.sin(angles[i])]
    
    return points


def _neg_score(params):
    """Negative score for minimization."""
    r_displaced, angle_offset = params
    points = _regular_polygon_displaced(r_displaced, angle_offset)
    return -_score(points)


def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = _score(best_points)
    
    # Try displacing one point radially inward with various angles
    # Use Nelder-Mead to optimize (r_displaced, angle_offset)
    
    # Initial guess: r=0.8, angle=0
    initial_guess = np.array([0.8, 0.0])
    
    try:
        result = minimize(_neg_score, initial_guess, method='Nelder-Mead',
                         options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000})
        
        if result.success or -result.fun > best_score:
            r_opt, angle_opt = result.x
            candidate_points = _regular_polygon_displaced(r_opt, angle_opt)
            candidate_score = _score(candidate_points)
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
    except Exception:
        pass
    
    # Try multiple starting points for robustness
    if time.time() < deadline:
        for r_start in [0.5, 0.7, 0.9, 1.0]:
            for angle_start in [0.0, np.pi/13, 2*np.pi/13]:
                if time.time() > deadline - 5:
                    break
                initial_guess = np.array([r_start, angle_start])
                try:
                    result = minimize(_neg_score, initial_guess, method='Nelder-Mead',
                                     options={'xatol': 1e-5, 'fatol': 1e-7, 'maxiter': 500})
                    
                    if -result.fun > best_score:
                        r_opt, angle_opt = result.x
                        candidate_points = _regular_polygon_displaced(r_opt, angle_opt)
                        candidate_score = _score(candidate_points)
                        if candidate_score > best_score:
                            best_points = candidate_points
                            best_score = candidate_score
                except Exception:
                    pass
            if time.time() > deadline - 5:
                break
    
    return best_points
# EVOLVE-BLOCK-END
