import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Pre-compute all triangle index combinations
    tri_indices = list(combinations(range(N), 3))
    tri_indices = np.array(tri_indices)  # shape (364, 3)
    
    def compute_score(pts):
        """Compute min triangle area / convex hull area"""
        # Convex Hull Area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume  # For 2D, .volume is area
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
        
        # Vectorized min triangle area computation
        # For each triangle (i,j,k), area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
        pi = pts[tri_indices[:, 0]]  # shape (364, 2)
        pj = pts[tri_indices[:, 1]]  # shape (364, 2)
        pk = pts[tri_indices[:, 2]]  # shape (364, 2)
        
        v1 = pj - pi  # shape (364, 2)
        v2 = pk - pi  # shape (364, 2)
        
        # Cross product: v1[0]*v2[1] - v1[1]*v2[0]
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        score = min_area / hull_area
        return score
    
    def neg_score(flat_pts):
        pts = flat_pts.reshape(N, 2)
        # Check for degeneracy
        if np.allclose(pts, pts[0]):
            return 0.0
        return -compute_score(pts)

    # Define bounds for the points. 
    # Since the score is normalized by the hull area, the absolute scale doesn't matter much, 
    # but we need a bounded domain for DE. 
    # A box of [-1, 1] x [-1, 1] is a reasonable starting space.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_score_val = 0
    best_result = None
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    initial_score = compute_score(x0.reshape(N, 2))
    if initial_score > best_score_val:
        best_score_val = initial_score
        best_result = x0.copy()
        
    # Use Differential Evolution for global search
    # We need to be careful with the number of iterations and population size to fit in the time limit.
    # DE is stochastic, so we set a fixed seed for reproducibility if needed, but random is fine.
    
    try:
        if time.time() < deadline - 30:  # Reserve 30s for local polish
            de_result = differential_evolution(
                neg_score,
                bounds,
                maxiter=200,  # Number of generations
                popsize=15,    # Population size
                tol=1e-6,
                seed=42,
                polish=False,   # We will do our own polish
                init='sobol'    # Use Sobol sequence for better initial spread
            )
            
            if de_result.fun < 0:
                score_val = -de_result.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_result = de_result.x.copy()
    except Exception:
        pass
    
    # Local polish using Powell or Nelder-Mead on the best result
    if best_result is not None and time.time() < deadline - 10:
        try:
            # Try Powell first
            res_polish = minimize(neg_score, best_result, method='Powell',
                                  options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
            score_val = -res_polish.fun
            if score_val > best_score_val:
                best_score_val = score_val
                best_result = res_polish.x.copy()
                
            # Try Nelder-Mead on the best result so far
            if time.time() < deadline - 5:
                res_nm = minimize(neg_score, best_result, method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
                score_val = -res_nm.fun
                if score_val > best_score_val:
                    best_score_val = score_val
                    best_result = res_nm.x.copy()
                    
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to regular polygon if something went wrong
        best_result = x0
        
    return best_result.reshape(N, 2)
