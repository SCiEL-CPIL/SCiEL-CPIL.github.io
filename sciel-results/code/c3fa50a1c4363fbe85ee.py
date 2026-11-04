"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1.0

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    
    # Vectorized calculation of all triangle areas
    # i, j, k are distinct indices
    # We can generate all combinations
    indices = np.arange(n)
    # Create meshgrid of indices
    # This creates arrays of shape (n, n, n) which is 14^3 = 2744, manageable
    ii, jj, kk = np.meshgrid(indices, indices, indices, indexing='ij')
    # Filter where i < j < k to avoid duplicates and self-intersections
    mask = (ii < jj) & (jj < kk)
    ii = ii[mask]
    jj = jj[mask]
    kk = kk[mask]
    
    p1 = points[ii]
    p2 = points[jj]
    p3 = points[kk]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy: Simulated Annealing with multiple diverse initializations.
    """
    best_points = None
    best_score = -np.inf
    
    # Initializations
    inits = []
    
    # 1. Regular 14-gon
    angles = 2 * np.pi * np.arange(N) / N
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Jittered Grid 4x4 (remove 2 corners)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    # Remove (0,0) and (1,1) to make 14 points? Or random removal.
    # Let's just take first 14
    inits.append(grid[:14] + np.random.RandomState(1).rand(14, 2) * 0.1)
    
    # 3. Random
    rng = np.random.RandomState(42)
    inits.append(rng.rand(N, 2))
    
    # 4. Two rings
    outer_angles = 2 * np.pi * np.arange(7) / 7
    inner_angles = 2 * np.pi * np.arange(7) / 7 + np.pi / 7
    outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    inits.append(np.vstack([outer, inner]))
    
    for init_points in inits:
        # Simulated Annealing
        current_points = init_points.copy()
        current_score = _score(current_points)
        
        T = 0.1
        T_min = 1e-6
        cooling = 0.995
        step_size = 0.1
        
        for i in range(20000):
            # Perturb a random point
            idx = np.random.randint(N)
            noise = np.random.randn(2) * step_size
            new_points = current_points.copy()
            new_points[idx] = current_points[idx] + noise
            
            new_score = _score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
            else:
                # Accept with probability exp((new - old) / T)
                # Note: We want to maximize, so delta = new - old
                delta = new_score - current_score
                if np.random.rand() < np.exp(delta / T):
                    current_points = new_points
                    current_score = new_score
            
            T *= cooling
            if T < T_min:
                break
        
        # Local refinement with Nelder-Mead on the flattened array
        # This might be slow, but we only do it for the best few
        # Let's just do a few hill climbing steps if score is good
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_points.copy()
            
    return best_points
# EVOLVE-BLOCK-END
