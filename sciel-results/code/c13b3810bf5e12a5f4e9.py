import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 14

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _generate_ring_points(params, n_out=7, n_in=7):
    """
    Generate points from ring parameters [r, theta].
    Outer ring: n_out points on unit circle.
    Inner ring: n_in points on circle of radius r, rotated by theta.
    """
    r, theta = params
    
    angles_out = np.linspace(0, 2*np.pi, n_out, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = theta + np.linspace(0, 2*np.pi, n_in, endpoint=False)
    inner_pts = r * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    return np.vstack([outer_pts, inner_pts])

def _neg_score_ring(params):
    """
    Negative score for a 2-ring configuration defined by [r, theta].
    Used for differential_evolution.
    """
    try:
        points = _generate_ring_points(params)
        return -_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin for 120s limit
    
    best_score = 0.0
    best_points = None
    
    # 1. Global Search: Differential Evolution on ring parameters (r, theta)
    # We assume the optimal structure is two concentric regular heptagons.
    # r is the radius of the inner heptagon.
    # theta is the rotational offset. Due to 7-fold symmetry, theta in [0, 2pi/7] is sufficient.
    
    bounds = [(0.3, 0.8), (0.0, 2*np.pi/7)]
    
    # We can also try other splits like (8,6) but (7,7) is the most symmetric and likely optimal for N=14.
    # Let's stick to (7,7) to maximize time for DE and Polish.
    
    try:
        # Use a reasonable population and iterations
        # 14 points is small, so _score is fast.
        res = differential_evolution(
            _neg_score_ring,
            bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            polish=True
        )
        
        if res.success:
            best_params = res.x
            best_points = _generate_ring_points(best_params)
            best_score = -res.fun
        else:
            # If DE fails, use the best found
            best_params = res.x
            best_points = _generate_ring_points(best_params)
            best_score = -res.fun
            
    except Exception:
        # Fallback: Regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)

    # 2. Local Refinement: Nelder-Mead on full 28D space
    # This allows breaking the perfect ring symmetry if it improves the minimum triangle area.
    if best_points is not None and time.time() < deadline - 10:
        
        def objective_full(x):
            if time.time() > deadline - 2:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return -_score(p)
            
        try:
            # Initialize NM with the best ring configuration
            x0 = best_points.flatten()
            
            res_nm = minimize(
                objective_full, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < -best_score: # i.e., score improved
                new_pts = res_nm.x.reshape(N, 2)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_points = new_pts
                    
        except TimeoutError:
            pass
        except Exception:
            pass

    # 3. Final Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points
