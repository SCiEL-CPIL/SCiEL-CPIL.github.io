import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 14

def get_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Min Triangle Area (Vectorized)
    # Generate all unique triples (i, j, k) with i < j < k
    # For N=14, number of triples is C(14,3) = 364. This is small enough to precompute indices.
    # However, get_score is called many times, so precomputing indices once is better if possible,
    # but we can't easily do that inside the function without global state or passing them.
    # Let's just compute them efficiently.
    
    # Precomputed indices for N=14
    # i < j < k
    # We can generate these once at module level or use a cached approach.
    # Since N is fixed, let's just use itertools or a simple loop to build the array once.
    
    # To avoid regenerating indices every call, we can use a global variable initialized once.
    global _triples_idx
    if not hasattr(get_score, '_triples_idx'):
        i_idx = []
        j_idx = []
        k_idx = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        get_score._triples_idx = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    
    i_flat, j_flat, k_flat = get_score._triples_idx
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We keep a safety margin.
    deadline = start_time + 145.0

    # Objective function for minimizers (maximize score -> minimize -score)
    def objective(flat_points):
        pts = flat_points.reshape(N, 2)
        return -get_score(pts)

    best_points = None
    best_score = 0.0

    # Strategy: Multiple local optimizations starting from diverse initializations.
    # 1. Regular Polygon
    # 2. Perturbed Regular Polygons
    # 3. Random Points (filtered for convexity/quality)
    # 4. Grid-like structures?
    
    # Since Powell is known to be effective for this type of smooth but non-convex objective,
    # we will use Powell's method primarily.
    
    initializations = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    initializations.append(np.column_stack((x, y)))
    
    # 2. Perturbed Regular Polygons
    rng = np.random.default_rng(42)
    for i in range(20):
        noise = rng.normal(0, 0.1, (N, 2))
        pts = initializations[0] + noise
        # Ensure it's not degenerate
        try:
            hull = ConvexHull(pts)
            if hull.volume > 1e-3:
                initializations.append(pts)
        except:
            pass

    # 3. Random Points in Unit Disk (or Square)
    # We want points to be well-distributed.
    for i in range(50):
        # Sample from unit square [-1, 1]x[-1, 1]
        pts = rng.uniform(-1, 1, (N, 2))
        try:
            hull = ConvexHull(pts)
            if hull.volume > 1e-3:
                initializations.append(pts)
        except:
            pass

    # 4. "Spiral" or other structured patterns?
    # Let's stick to perturbations and randoms for now, as Powell is strong.

    for init_pts in initializations:
        if time.time() > deadline - 5.0:
            break
            
        try:
            # Use Powell's method for local refinement
            # Powell is derivative-free and good for this
            res = minimize(
                objective, 
                init_pts.flatten(), 
                method='Powell', 
                options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000, 'maxfev': 20000}
            )
            
            if res.success or res.fun < -best_score:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
                    
        except Exception:
            pass

    # If we have time, do a final fine-tuning with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            res_nm = minimize(
                objective, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 50000}
            )
            if res_nm.fun < -best_score:
                best_score = -res_nm.fun
                best_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        best_points = np.column_stack((x, y))

    return best_points
