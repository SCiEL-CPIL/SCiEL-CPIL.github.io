import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def get_points_from_uv(u, v):
    """
    Generate 8 points from parameters u, v using D4 symmetry.
    Points are (±u, ±v) and (±v, ±u).
    """
    pts = np.array([
        [u, v], [u, -v], [-u, v], [-u, -v],
        [v, u], [v, -u], [-v, u], [-v, -u]
    ])
    return pts

def score_from_points(pts):
    """
    Calculate the score for a set of points.
    1. Check distinctness.
    2. Calculate diameter.
    3. Scale to diameter 1 (for internal calculation, then scale back for score formula).
    4. Calculate product of distances.
    5. Return score.
    """
    n = len(pts)
    if n != 8:
        return -np.inf
    
    # Check distinctness
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-6:
                return -np.inf
            if d < min_dist:
                min_dist = d
        
    # Calculate diameter
    max_dist = 0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d > max_dist:
                max_dist = d
                
    if max_dist < 1e-6:
        return -np.inf
        
    # Scale to diameter 1
    scaled_pts = pts / max_dist
    
    # Calculate product of distances (log sum for stability)
    log_prod = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                d = np.linalg.norm(scaled_pts[i] - scaled_pts[j])
                if d < 1e-15:
                    return -np.inf
                log_prod += np.log(d)
                
    # Score formula:
    # Delta_1 = prod d_ij (diameter 1)
    # Score = 2^32 * Delta_1
    # log(Score) = 32 * ln(2) + log_prod
    log_score = 32 * np.log(2) + log_prod
    return np.exp(log_score)

def objective_uv(params):
    """
    Objective function for differential_evolution on (u, v).
    Returns negative score (to be minimized).
    """
    u, v = params
    # Ensure non-negative
    u = abs(u)
    v = abs(v)
    
    pts = get_points_from_uv(u, v)
    score = score_from_points(pts)
    if not np.isfinite(score):
        return 1e10
    return -score

def objective_16d(params):
    """
    Objective function for 16D optimization.
    params: 16 floats [x0, y0, x1, y1, ..., x7, y7]
    """
    pts = params.reshape(8, 2)
    score = score_from_points(pts)
    if not np.isfinite(score):
        return 1e10
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 1.0
    best_points = None
    
    # 1. D4 Symmetric Search (2D)
    # Bounds for u, v. Since diameter is normalized, max coordinate is 1.
    # Actually, if u=1, v=0, dist is 2. Scaled down.
    # So u, v in [0, 1] is sufficient.
    
    try:
        bounds_uv = [(0, 1), (0, 1)]
        result_uv = differential_evolution(
            objective_uv, 
            bounds_uv, 
            maxiter=100, 
            popsize=15, 
            tol=1e-8, 
            seed=42
        )
        
        if result_uv.success and result_uv.fun < 0:
            score_uv = -result_uv.fun
            if score_uv > best_score:
                best_score = score_uv
                u, v = result_uv.x
                best_points = get_points_from_uv(u, v)
                
    except Exception as e:
        pass
        
    if time.time() > deadline:
        if best_points is not None:
            return best_points.tolist()
        # Fallback to regular octagon
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
        
    # 2. Refine D4 solution with local optimizer
    if best_points is not None:
        # Extract u, v from best_points? 
        # It's easier to just use result_uv.x if available.
        # But if we found it via grid or other means, we need to recover u,v.
        # For now, assume result_uv.x is the best.
        if 'result_uv' in locals() and result_uv.success:
            x0_uv = result_uv.x
            res_refine = minimize(
                objective_uv, 
                x0_uv, 
                method='Powell', 
                tol=1e-12, 
                options={'maxiter': 1000}
            )
            if res_refine.fun < 0:
                score_ref = -res_refine.fun
                if score_ref > best_score:
                    best_score = score_ref
                    u, v = res_refine.x
                    best_points = get_points_from_uv(u, v)

    if time.time() > deadline:
        if best_points is not None:
            return best_points.tolist()
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    # 3. General 16D Search
    # Start from the best D4 configuration or regular octagon
    if best_points is not None:
        initial_16d = best_points.flatten()
    else:
        from math import cos, sin, pi
        oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        initial_16d = oct_pts.flatten()
        
    # Perform a local refinement on the 16D space to break symmetry if needed
    # Using Powell or Nelder-Mead
    try:
        res_16d = minimize(
            objective_16d, 
            initial_16d, 
            method='Powell', 
            tol=1e-10, 
            options={'maxiter': 2000}
        )
        if res_16d.fun < 0:
            score_16d = -res_16d.fun
            if score_16d > best_score:
                best_score = score_16d
                best_points = res_16d.x.reshape(8, 2)
    except Exception as e:
        pass

    if time.time() > deadline:
        if best_points is not None:
            return best_points.tolist()
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]

    # 4. Run a short 16D Differential Evolution to explore non-D4 regions
    # Only if time permits
    if time.time() < deadline - 30:
        try:
            bounds_16d = [(-1, 1) for _ in range(16)]
            # Use a small population to keep it fast
            res_de_16d = differential_evolution(
                objective_16d, 
                bounds_16d, 
                maxiter=20, 
                popsize=10, 
                tol=1e-6, 
                seed=42,
                init='sobol'
            )
            if res_de_16d.fun < 0:
                score_de = -res_de_16d.fun
                if score_de > best_score:
                    best_score = score_de
                    best_points = res_de_16d.x.reshape(8, 2)
                    
                    # Refine the DE result
                    res_ref_16d = minimize(
                        objective_16d, 
                        res_de_16d.x, 
                        method='Powell', 
                        tol=1e-10, 
                        options={'maxiter': 500}
                    )
                    if res_ref_16d.fun < 0:
                        score_ref_de = -res_ref_16d.fun
                        if score_ref_de > best_score:
                            best_score = score_ref_de
                            best_points = res_ref_16d.x.reshape(8, 2)
        except Exception as e:
            pass

    if best_points is not None:
        # Final check: ensure coordinates are within limits
        # The scaling in score_from_points handles diameter, but the output points
        # should be the raw points that achieve the score.
        # The task says "scaling the set to diameter 2".
        # My score_from_points scales to diameter 1 internally to compute the score.
        # The returned points should be the ones that, when scaled to diameter 2, give the score.
        # Actually, the score is invariant under scaling.
        # So we can return the points as they are (diameter 1 or whatever).
        # However, the prompt says "construct() -> 8 distinct points ... |coordinate| < 10^6".
        # It doesn't require the returned points to have diameter 2.
        # It says "score is ... after scaling the set to diameter 2".
        # So returning points with diameter 1 is fine, as long as they are distinct.
        
        # One detail: if the optimal configuration has diameter 1 in my internal calc,
        # the returned points will have diameter 1.
        # This is valid.
        
        return best_points.tolist()
    else:
        from math import cos, sin, pi
        return [[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]
