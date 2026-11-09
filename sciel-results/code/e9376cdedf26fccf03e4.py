import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Solves the Heilbronn problem for 13 points in a convex region using
    convex optimization (Linear Programming).
    
    Strategy:
    1. Fix the convex hull to be a specific polygon (e.g., 13-gon) to normalize area.
    2. The problem of maximizing the minimum triangle area is non-convex in general,
       but we can use a bisection search on the minimum area threshold delta.
    3. For a given delta, we check if there exists a configuration of 13 points
       within the hull such that all triangle areas >= delta.
       This feasibility problem can be formulated as a Linear Program if we fix
       the positions of some points or use specific structural assumptions.
       
       However, a full LP for free points is complex. 
       Instead, we use a hybrid:
       - We assume the optimal configuration has points on the boundary of the convex hull 
         or arranged in a specific grid-like structure.
       
       Actually, a more robust convex approach for small N:
       We can use the fact that for a fixed set of points, the min triangle area is 
       the min over all triples. 
       
       Let's use a different convex strategy:
       We maximize the minimum area by solving a sequence of LPs.
       But LPs require linear constraints. 
       Triangle area is linear in coordinates for fixed other two points? No, it's 
       |det|. |det| >= delta is not linear. It is a SOCP (Second Order Cone Program).
       
       Since we only have standard scipy (linprog), we cannot solve SOCP directly.
       
       Alternative Strategy: 
       Use a "Fixed Hull" approach. 
       Assume the convex hull is a regular 13-gon. 
       We want to place 13 points. 
       If we place points ON the hull, the problem is simpler? 
       
       Let's try a different material strategy: 
       **Gradient Ascent with Subgradients**.
       The function f(P) = min_triangle_area(P) / hull_area(P) is differentiable almost everywhere.
       We can compute the subgradient with respect to the points that form the minimum triangle.
       We perform subgradient ascent.
       
       This is different from the current "expand min triangle" heuristic because it uses 
       formal subgradient information from all active constraints (if we approximate the min 
       as a smooth min) or just the single min triangle gradient, and we normalize by hull area 
       properly.
       
       Wait, the prompt asks for a strategy MATERIALLY DIFFERENT from "refining the current program".
       The current program uses `_expand_min_triangle` (heuristic move) and `_refine_local` (simple loop).
       
       New Strategy: **Nelder-Mead Simplex Optimization** on the vector of 26 coordinates.
       Nelder-Mead is a derivative-free optimization method that is robust for non-smooth functions 
       like min(triangle areas). It explores the space more intelligently than simple coordinate 
       expansion by reflecting, expanding, and contracting the simplex of candidate points.
       
       Why is this different?
       1. Current: Heuristic "push apart" specific points.
       2. New: Systematic simplex method that adapts step size and direction based on function evaluations 
          of the entire configuration.
       3. It handles the non-convexity and non-smoothness (min function) better than ad-hoc moves.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10 min budget, 10s safety margin

    def score(points):
        """
        Compute the score: min triangle area / convex hull area.
        Returns negative score for invalid to maximize.
        """
        if points.shape != (13, 2):
            return -np.inf
        if not np.all(np.isfinite(points)):
            return -np.inf
        
        # Calculate min triangle area
        min_area = float('inf')
        # Vectorized calculation of all 286 triangle areas
        # i, j, k indices
        # To be fast, we can use broadcasting but 13 is small so loops are fine, 
        # but let's try a semi-vectorized approach for speed in Nelder-Mead (many evals)
        
        # Actually, for Nelder-Mead, speed is critical. 
        # 286 triangles * 26 coords. 
        # Let's write a fast loop in Python, it's usually fast enough for 13 points.
        
        # Precompute indices
        indices = np.array(list(itertools.combinations(range(13), 3)))
        i, j, k = indices.T
        p1 = points[i]
        p2 = points[j]
        p3 = points[k]
        
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        # Cross product in 2D: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
        # Area = 0.5 * |cross|
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        # Calculate Hull Area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return -np.inf

        if hull_area <= 1e-10:
            return -np.inf

        return min_area / hull_area

    # Initial configurations
    # 1. Regular Polygon
    theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    init_poly = np.column_stack((np.cos(theta), np.sin(theta)))
    
    # 2. Halton
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_points = np.zeros((13, 2))
    for i in range(13):
        halton_points[i, 0] = halton(i + 1, 2)
        halton_points[i, 1] = halton(i + 1, 3)
    init_halton = (halton_points - 0.5) * 2.0

    # 3. Random
    rng = np.random.default_rng(42)
    init_rand = rng.uniform(-1, 1, (13, 2))
    
    inits = [init_poly, init_halton, init_rand]
    
    best_score = -np.inf
    best_points = None
    
    from scipy.optimize import minimize
    
    # Nelder-Mead settings
    # We want to maximize score, so we minimize -score
    def neg_score(points_flat):
        pts = points_flat.reshape(13, 2)
        return -score(pts)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Nelder-Mead
        # x0 is flattened
        x0 = init.flatten()
        
        # Options for Nelder-Mead
        # maxiter: limit iterations
        # xatol, fatol: tolerances
        
        # We need to be careful with time. 
        # Each evaluation takes ~1-2ms (ConvexHull is the bottleneck).
        # Nelder-Mead might do 1000-5000 evaluations.
        # 5000 * 2ms = 10s. 
        # We have 590s. So we can run many instances or one long one.
        
        try:
            res = minimize(
                neg_score, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-10, 
                    'adaptive': True
                }
            )
            
            res_pts = res.x.reshape(13, 2)
            res_score = -res.fun
            
            if res_score > best_score:
                best_score = res_score
                best_points = res_pts
                
        except Exception as e:
            pass
        
        # If we have time, do a second pass with a different init or refinement
        if time.time() < deadline - 60:
             # Try to perturb the best found so far and run NM again
             if best_points is not None:
                 perturbed = best_points + rng.normal(0, 0.1, (13, 2))
                 x0_pert = perturbed.flatten()
                 try:
                     res2 = minimize(
                         neg_score, 
                         x0_pert, 
                         method='Nelder-Mead', 
                         options={
                             'maxiter': 1000, 
                             'xatol': 1e-9, 
                             'fatol': 1e-11, 
                             'adaptive': True
                         }
                     )
                     res_pts2 = res2.x.reshape(13, 2)
                     res_score2 = -res2.fun
                     if res_score2 > best_score:
                         best_score = res_score2
                         best_points = res_pts2
                 except Exception:
                     pass

    # Fallback if NM failed or didn't run
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = init_poly
        
    return best_points

if __name__ == "__main__":
    # Quick test
    pts = heilbronn_convex13()
    print("Shape:", pts.shape)
    print("Score:", score(pts))
