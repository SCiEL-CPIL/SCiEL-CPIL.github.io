import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
        
    n = points.shape[0]
    
    # Calculate minimum triangle area
    min_area = np.inf
    
    # Loop over all triples
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v2 = pk - pi
                # Cross product magnitude gives 2 * area
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if not np.isfinite(min_area) or min_area <= 0:
        return 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area <= 0:
        return 0.0

    return min_area / hull_area

def _calculate_score_fast(points):
    """
    Fast vectorized score calculation. Returns (score, min_area, hull_area).
    """
    if not np.all(np.isfinite(points)):
        return 0.0, 0.0, 0.0
        
    n = points.shape[0]
    
    # Vectorized minimum triangle area calculation
    # For each pair (i,j), compute cross products with all k > j
    min_area = np.inf
    
    # Use a more efficient approach: for each point i, compute areas with all pairs (j,k) where j,k > i
    for i in range(n):
        pi = points[i]
        rest = points[i+1:]
        nr = len(rest)
        if nr < 2:
            continue
        # Compute vectors from pi to all other points
        vecs = rest - pi  # shape (nr, 2)
        # Compute cross products for all pairs
        # For each pair (a, b) where a < b in rest
        for a in range(nr):
            v1 = vecs[a]
            for b in range(a + 1, nr):
                v2 = vecs[b]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    
    if not np.isfinite(min_area) or min_area <= 0:
        return 0.0, 0.0, 0.0

    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0, 0.0, 0.0
        
    if hull_area <= 0:
        return 0.0, 0.0, 0.0

    return min_area / hull_area, min_area, hull_area

def _negative_score(flat_points):
    points = flat_points.reshape(N, 2)
    return -_calculate_score(points)

def _affine_points(params):
    """
    Generate 14 points from a regular 14-gon transformed by an affine matrix.
    params: [a, b, c, d, e, f] representing the affine transform:
        x' = a*x + c*y + e
        y' = b*x + d*y + f
    """
    angles = 2 * np.pi * np.arange(N) / N
    base = np.column_stack((np.cos(angles), np.sin(angles)))
    
    a, b, c, d, e, f = params
    
    # Apply affine transformation
    x = a * base[:, 0] + c * base[:, 1] + e
    y = b * base[:, 0] + d * base[:, 1] + f
    
    return np.column_stack((x, y))

def _negative_affine_score(params):
    points = _affine_points(params)
    return -_calculate_score(points)

def construct():
    start_time = time.time()
    timeout = 135.0  # Safety margin for 150s limit
    
    # 1. Initialize with Regular Polygon
    angles = 2 * np.pi * np.arange(N) / N
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = _calculate_score(init_points)
    best_points = init_points.copy()
    
    # 2. Affine Transformation Optimization
    # Start with identity transform and optimize the 6 affine parameters
    # This preserves the angular structure while allowing shape adaptation
    if time.time() - start_time < timeout - 60:
        try:
            # Initial affine params: identity (scale=1, no rotation, no shear, no translation)
            # a=1, b=0, c=0, d=1, e=0, f=0
            affine_bounds = [
                (0.5, 2.0),  # a: x-scale
                (-1.0, 1.0), # b: y from x (rotation/shear)
                (-1.0, 1.0), # c: x from y (rotation/shear)
                (0.5, 2.0),  # d: y-scale
                (-2.0, 2.0), # e: x translation
                (-2.0, 2.0)  # f: y translation
            ]
            
            # Use differential evolution on affine parameters
            res_affine_de = differential_evolution(
                _negative_affine_score,
                affine_bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                polish=False,
                popsize=10
            )
            
            if res_affine_de.x is not None:
                candidate_points = _affine_points(res_affine_de.x)
                cand_score = _calculate_score(candidate_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate_points.copy()
        except Exception:
            pass
    
    # 3. Full Point Optimization with Differential Evolution
    if time.time() - start_time < timeout - 40:
        bounds = [(-2, 2)] * (2 * N)
        
        def de_callback(xk, convergence):
            nonlocal best_score, best_points
            nonlocal start_time, timeout
            if time.time() - start_time > timeout:
                raise RuntimeError("Timeout")
            # Evaluate current best from DE
            pts = xk.reshape(N, 2)
            s = _calculate_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                
        try:
            res_de = differential_evolution(
                _negative_score, 
                bounds, 
                seed=42, 
                maxiter=100, 
                tol=1e-8, 
                polish=False,
                callback=de_callback,
                popsize=12
            )
        except RuntimeError:
            pass
        except Exception:
            pass

    # 4. Local Refinement with Powell
    if time.time() - start_time < timeout - 25:
        try:
            res_powell = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Powell', 
                options={'maxiter': 300, 'disp': False, 'xtol': 1e-9, 'ftol': 1e-9}
            )
            if res_powell.x is not None:
                candidate_points = res_powell.x.reshape(N, 2)
                cand_score = _calculate_score(candidate_points)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = candidate_points.copy()
        except Exception:
            pass

    # 5. Further Refinement with Nelder-Mead
    if time.time() - start_time < timeout - 15:
        try:
            res_nm = minimize(
                _negative_score, 
                best_points.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'disp': False, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res_nm.x is not None:
                candidate_points2 = res_nm.x.reshape(N, 2)
                cand_score2 = _calculate_score(candidate_points2)
                if cand_score2 > best_score:
                    best_score = cand_score2
                    best_points = candidate_points2.copy()
        except Exception:
            pass

    # 6. Final fine-tuning with small perturbations
    if time.time() - start_time < timeout - 5:
        for _ in range(5):
            if time.time() - start_time > timeout - 5:
                break
            try:
                # Try small random perturbations
                perturbation_scale = 0.01
                for attempt in range(20):
                    perturbed = best_points + perturbation_scale * np.random.randn(N, 2)
                    perturbed_score = _calculate_score(perturbed)
                    if perturbed_score > best_score:
                        best_score = perturbed_score
                        best_points = perturbed.copy()
                        perturbation_scale *= 0.5
                    else:
                        perturbation_scale *= 0.7
            except Exception:
                pass

    return best_points
