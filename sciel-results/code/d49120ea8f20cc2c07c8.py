# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area_and_indices(pts):
    """
    Calculate the minimum triangle area and the indices of the vertices forming it.
    """
    n = len(pts)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    
    # For N=13, there are 286 triangles. A simple loop is fast enough.
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                area = 0.5 * abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _hull_area(pts):
    """
    Calculate the area of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """
    Calculate the score: A_min / A_hull.
    """
    a_min, _ = _min_triangle_area_and_indices(pts)
    a_hull = _hull_area(pts)
    if a_hull == 0:
        return 0.0
    return a_min / a_hull

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest
    triangle area to the convex hull area.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit

    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Local Search with Simulated Annealing and Gradient Ascent on bottleneck
    current_pts = pts.copy()
    current_score = best_score
    
    temp = 0.1  # Initial temperature for random perturbations
    cooling = 0.999
    min_temp = 1e-4
    step_size = 0.01  # Base step size for gradient moves
    
    no_improvement_count = 0
    max_no_improvement = 2000
    
    while time.time() < deadline:
        # 1. Identify the bottleneck triangle
        a_min, (i, j, k) = _min_triangle_area_and_indices(current_pts)
        
        # 2. Try to expand this specific triangle using gradient information
        # Area = 0.5 * s * (xi(yj-yk) + xj(yk-yi) + xk(yi-yj))
        xi, yi = current_pts[i]
        xj, yj = current_pts[j]
        xk, yk = current_pts[k]
        
        # Determine orientation sign
        cross = (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi)
        s = 1.0 if cross > 0 else -1.0
        
        # Gradients for increasing the area of triangle (i,j,k)
        # dA/dxi = 0.5 * s * (yj - yk)
        # dA/dyi = 0.5 * s * (xk - xj)
        gi = 0.5 * s * np.array([yj - yk, xk - xj])
        gj = 0.5 * s * np.array([yk - yi, xi - xk])
        gk = 0.5 * s * np.array([yi - yj, xj - xi])
        
        # Normalize gradients to have unit length for consistent step size
        norm_gi = np.linalg.norm(gi)
        norm_gj = np.linalg.norm(gj)
        norm_gk = np.linalg.norm(gk)
        
        if norm_gi > 1e-9: gi /= norm_gi
        if norm_gj > 1e-9: gj /= norm_gj
        if norm_gk > 1e-9: gk /= norm_gk
        
        # Propose move: Move vertices in direction of gradient
        new_pts = current_pts.copy()
        new_pts[i] += step_size * gi
        new_pts[j] += step_size * gj
        new_pts[k] += step_size * gk
        
        new_score = _score(new_pts)
        
        # Acceptance logic
        accepted = False
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            accepted = True
        else:
            # Metropolis acceptance for random exploration
            delta = new_score - current_score
            if np.random.random() < np.exp(delta / temp):
                current_pts = new_pts
                current_score = new_score
                accepted = True
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Occasionally inject random noise to escape local minima
        if no_improvement_count > max_no_improvement or np.random.random() < 0.01:
            # Random perturbation of all points
            noise = np.random.normal(0, 0.05, current_pts.shape)
            current_pts = current_pts + noise
            current_score = _score(current_pts)
            no_improvement_count = 0
            
            # If score dropped significantly, revert to best
            if current_score < best_score * 0.5:
                current_pts = best_pts.copy()
                current_score = best_score
        
        # Cooling
        temp *= cooling
        if temp < min_temp:
            temp = 0.05 # Reheat slightly to continue exploration
            # Also, maybe try a different step size
            step_size = np.random.uniform(0.005, 0.02)

    return best_pts
# EVOLVE-BLOCK-END
