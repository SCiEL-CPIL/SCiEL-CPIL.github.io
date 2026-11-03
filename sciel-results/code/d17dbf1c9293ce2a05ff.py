import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

APOTHEM = math.sqrt(3) / 2.0

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts)

def solve_lp_for_theta(pts, theta):
    """
    Solves the LP for a fixed rotation theta.
    Returns R (inradius of the container hexagon needed to enclose pts)
    and the center (cx, cy).
    """
    n_pts = len(pts)
    # Normals for the container hexagon
    # The sides of the hexagon are at angles theta + k*pi/3
    # The normals are at theta + pi/6 + k*pi/3
    angles = theta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    A_ub_list = []
    b_ub_list = []
    
    for i in range(n_pts):
        px, py = pts[i]
        for k in range(6):
            nx, ny = normals[k]
            # (px - cx) * nx + (py - cy) * ny <= R
            # -cx*nx - cy*ny - R <= -px*nx - py*ny
            A_ub_list.append([-nx, -ny, -1.0])
            b_ub_list.append(-px * nx - py * ny)
            
            # (px - cx) * nx + (py - cy) * ny >= -R
            # cx*nx + cy*ny - R <= px*nx + py*ny
            A_ub_list.append([nx, ny, -1.0])
            b_ub_list.append(px * nx + py * ny)
            
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(np.array([0.0, 0.0, 1.0]), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            return res.x[2], (res.x[0], res.x[1])
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def optimize_rotation(pts):
    """
    Finds the theta that minimizes R for the given set of points.
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf'), 0.0, None

    def objective(theta):
        R, center = solve_lp_for_theta(pts, theta)
        return R

    # Coarse grid search
    best_R = float('inf')
    best_theta = 0.0
    best_center = None
    
    # 60 steps in [0, pi/6]
    for i in range(61):
        theta = i * (math.pi / 6.0) / 60.0
        R, center = objective(theta)
        if R < best_R:
            best_R = R
            best_theta = theta
            best_center = center
            
    # Refine
    try:
        res = minimize_scalar(objective, bounds=(0, math.pi / 6.0), method='bounded',
                              options={'maxiter': 100, 'xatol': 1e-10})
        if res.fun < best_R:
            best_R = res.fun
            best_theta = res.x
            _, best_center = solve_lp_for_theta(pts, best_theta)
    except Exception:
        pass
        
    return best_R, best_theta, best_center

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate Candidates
    candidates = []
    
    # Base: Flower (1 center + 6 neighbors)
    base_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # Ring 2 positions
    ring2_axial = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2), (-2, 1),
        (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -2), (2, -1)
    ]
    
    # Candidate 1: Flower + 4 specific symmetric corners
    # Let's test all combinations of 4 from Ring 2
    # To limit computation, we can filter by heuristic:
    # Prefer combinations that are "balanced"
    
    # We will test a subset of combinations.
    # Total combos: C(12,4) = 495.
    # This is small enough to test all.
    
    best_global_R = float('inf')
    best_global_theta = 0.0
    best_global_center = None
    best_global_pts = None
    
    # Pre-compute cartesian for ring 2
    ring2_pts = axial_to_cartesian(ring2_axial)
    base_pts = axial_to_cartesian(base_axial)
    
    # We will iterate through combinations
    # To speed up, we can skip if time is running out
    for i in range(len(ring2_axial)):
        for j in range(i+1, len(ring2_axial)):
            for k in range(j+1, len(ring2_axial)):
                for l in range(k+1, len(ring2_axial)):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    extra_axial = [ring2_axial[i], ring2_axial[j], ring2_axial[k], ring2_axial[l]]
                    axial_list = base_axial + extra_axial
                    pts = np.vstack([base_pts, ring2_pts[[i, j, k, l]]])
                    
                    # Heuristic check: bounding box of centers
                    # If the bbox is too large, skip
                    # Actually, just run the LP
                    
                    R, theta, center = optimize_rotation(pts)
                    if R < best_global_R:
                        best_global_R = R
                        best_global_theta = theta
                        best_global_center = center
                        best_global_pts = pts

    # If no candidate found (should not happen), fallback to base flower + random
    if best_global_pts is None:
        axial_list = base_axial + [ring2_axial[0], ring2_axial[2], ring2_axial[4], ring2_axial[6]]
        best_global_pts = axial_to_cartesian(axial_list)
        best_global_R, best_global_theta, best_global_center = optimize_rotation(best_global_pts)

    # Calculate S
    # S = 2R/sqrt(3) + 1
    S = (2.0 * best_global_R) / math.sqrt(3) + 1.0
    
    # Prepare output
    # Inner hexagons: (x, y, angle)
    # All inner hexagons have angle 0 (aligned with grid)
    # The container is rotated by best_global_theta
    # Wait, the problem asks for inner (x,y,angle) and outer center, side, rotation.
    # If we fixed inner at angle 0, and rotated container by theta,
    # then in the global frame:
    # Inner hexagons are at (x,y) with angle 0.
    # Outer hexagon is centered at (cx, cy) with rotation theta.
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), 0.0])
        
    center = [float(best_global_center[0]), float(best_global_center[1])]
    side = float(S)
    rotation = float(math.degrees(best_global_theta))
    
    return inner, center, side, rotation

if __name__ == '__main__':
    construct()
