import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRI = len(IDX_I)

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _hexagonal_points(params):
    """
    Generate 14 points from a 2x7 hexagonal lattice structure.
    params: array of shape (21,)
    - params[0:2]: Basis vector 1 (u)
    - params[2:4]: Basis vector 2 (v)
    - params[4:21]: 7 offsets for the 7 interior points (2 per point)
    """
    u = params[0:2]
    v = params[2:4]
    offsets = params[4:21].reshape(7, 2)
    
    # Grid coordinates for 2x7
    # i in {0, 1}, j in {0, ..., 6}
    # Point (i, j) is at i*u + j*v + offset[i*7 + j]
    points = np.zeros((14, 2))
    for i in range(2):
        for j in range(7):
            idx = i * 7 + j
            points[idx] = i * u + j * v + offsets[idx]
            
    return points

def _objective(params):
    """
    Maximize min triangle area / hull area.
    We minimize - (min_area / hull_area).
    To make it smooth, we can use a soft-min approximation if needed, 
    but for the initial structural change, let's try direct optimization 
    with a robust method or just standard minimization on the smooth 
    surrogate if we implement one. 
    However, min is non-differentiable. 
    Let's use a log-sum-exp soft min on the negative areas?
    Actually, maximizing the minimum is equivalent to minimizing the maximum of -areas.
    Let's use a soft-min approximation for the area to make it differentiable for gradient methods,
    or stick to Nelder-Mead which handles non-differentiable functions.
    Given the 21D space, Nelder-Mead is viable but might be slow.
    Let's try SLSQP or BFGS with a smoothed objective.
    
    Soft min approximation: -log(sum(exp(-areas/T))) * T
    For small T, this approximates min(areas).
    """
    pts = _hexagonal_points(params)
    areas = _all_areas(pts)
    hull_area = _compute_hull_area(pts)
    
    if hull_area < 1e-10:
        return 1e10 # Penalize degenerate hulls
        
    # We want to maximize min(areas) / hull_area.
    # Let's maximize min(areas) subject to hull_area being reasonable, 
    # or directly optimize the ratio.
    # The ratio is tricky because hull_area changes.
    # Let's just maximize min(areas) - lambda * variance? No.
    # Let's maximize min(areas) / hull_area.
    
    min_area = np.min(areas)
    return -min_area / hull_area

def _objective_smooth(params, T=0.01):
    """
    Smoothed objective: -soft_min(areas) / hull_area
    soft_min(areas) = -T * log(sum(exp(-areas/T)))
    """
    pts = _hexagonal_points(params)
    areas = _all_areas(pts)
    hull_area = _compute_hull_area(pts)
    
    if hull_area < 1e-10:
        return 1e10
        
    # Clip areas to avoid overflow in exp
    areas_clipped = np.clip(areas, 1e-8, None)
    soft_min = -T * np.log(np.sum(np.exp(-areas_clipped / T)))
    
    return -soft_min / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _elapsed():
        return time.time() - start_time

    # Initial guess for hexagonal lattice
    # Regular hexagon side length s.
    # Basis vectors:
    # u = (s, 0)
    # v = (s/2, s*sqrt(3)/2)
    # For a 2x7 grid, this forms a parallelogram-ish shape.
    # We want to maximize min triangle area.
    # In a regular triangular lattice, the minimum triangle area is the area of the smallest triangle,
    # which is the area of the lattice cell / 2? 
    # Area of lattice cell = |u x v| = s^2 * sqrt(3)/2.
    # The triangles formed by 3 lattice points can have areas that are multiples of this.
    # The smallest non-zero area is the cell area / 2? No, the cell is a parallelogram.
    # The triangle formed by 3 adjacent lattice points (forming a small equilateral triangle) 
    # has area = (sqrt(3)/4) * s^2.
    # The hull area of the 2x7 grid will be roughly 7 * cell_area + some boundary effects.
    
    s = 1.0
    u0 = np.array([s, 0.0])
    v0 = np.array([s/2.0, s * np.sqrt(3)/2.0])
    
    # Offsets should be small to start with, allowing the lattice to distort.
    # Or start with 0 and let optimization move them.
    offsets0 = np.zeros((7, 2))
    
    params0 = np.concatenate([u0, v0, offsets0.flatten()])
    
    # Also try a few other initializations
    # 1. Slightly distorted hexagon
    s2 = 0.8
    u1 = np.array([s2, 0.1])
    v1 = np.array([s2/2.0, s2 * np.sqrt(3)/2.0])
    params1 = np.concatenate([u1, v1, np.zeros(14)])
    
    # 2. Random small offsets
    np.random.seed(123)
    offsets_rand = np.random.uniform(-0.1, 0.1, (7, 2))
    params2 = np.concatenate([u0, v0, offsets_rand.flatten()])
    
    initial_params_list = [params0, params1, params2]
    
    # Optimization loop
    for p0 in initial_params_list:
        if _elapsed() > deadline - 30.0:
            break
            
        # Try Nelder-Mead first as it doesn't require gradients and is robust
        try:
            res = minimize(_objective, p0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < 1e9:
                pts = _hexagonal_points(res.x)
                score = _score(pts)
                update_best(pts, score)
                
                # Polish with BFGS using smoothed objective
                if _elapsed() < deadline - 10.0:
                    # Use a smaller T for polishing
                    res2 = minimize(_objective_smooth, res.x, method='BFGS', 
                                    options={'maxiter': 100, 'gtol': 1e-8})
                    if res2.fun < 1e9:
                        pts2 = _hexagonal_points(res2.x)
                        score2 = _score(pts2)
                        update_best(pts2, score2)
                        
        except Exception:
            pass

    # Fallback if hexagonal structure fails or is worse than a simple polygon
    if best_points is None or best_score < 30.0:
         angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
         fallback_pts = np.column_stack((np.cos(angles), np.sin(angles)))
         fallback_score = _score(fallback_pts)
         update_best(fallback_pts, fallback_score)

    # Final check
    if best_points is None:
         angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
         best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points

if __name__ == "__main__":
    # Simple self-test
    pts = construct()
    print("Score:", _score(pts))
