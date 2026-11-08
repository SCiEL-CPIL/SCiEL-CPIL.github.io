"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _min_triangle_area_ratio(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    # Compute convex hull area
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        # Degenerate case
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = pts[i], pts[j], pts[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area

def _neg_score(pts_flat):
    """Negative of the score for scipy minimize (which minimizes)."""
    pts = pts_flat.reshape(-1, 2)
    return -_min_triangle_area_ratio(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Initial guess: regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    best_pts = x0.reshape(-1, 2)
    best_score = _min_triangle_area_ratio(best_pts)
    
    # Try a few random initializations and optimize from each
    np.random.seed(42)
    
    n_evals = 0
    for init_idx in range(5):
        if time.time() > deadline - 10.0:
            break
        
        if init_idx == 0:
            init = x0.copy()
        else:
            # Perturbed regular polygon or random
            if init_idx <= 2:
                angles_pert = 2 * np.pi * np.arange(N) / N + np.random.randn(N) * 0.1
                init = np.column_stack([np.cos(angles_pert), np.sin(angles_pert)]).flatten()
            else:
                # Random points in unit circle
                r = np.sqrt(np.random.rand(N))
                theta = 2 * np.pi * np.random.rand(N)
                init = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).flatten()
        
        # Use Nelder-Mead for optimization
        try:
            result = minimize(_neg_score, init, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            n_evals += 1
            if result.fun < 0:  # Valid optimization
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_pts = result.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Final result
    return best_pts
# EVOLVE-BLOCK-END
