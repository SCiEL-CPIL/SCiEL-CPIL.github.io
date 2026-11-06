import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

_N = 8
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _score(pts):
    """
    Calculate the score for a set of points.
    Score = (Product of all ordered pair distances) / (Diameter^N * N^N) ?
    Task says: "Delta/n^n, where Delta is the product over ordered pairs of distances after scaling the set to diameter 2"
    Let d_ij be distances.
    Scale factor k = 2 / D_max.
    Scaled distances d'_ij = k * d_ij.
    Delta = prod_{i!=j} d'_ij.
    Score = Delta / 8^8.
    """
    pts = np.asarray(pts, dtype=float).reshape(_N, 2)
    # Calculate all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    
    # Diameter
    dia = d.max()
    if dia < 1e-12:
        return 0.0
    
    # Scale to diameter 2
    # d_scaled = d * (2.0 / dia)
    # Product of all off-diagonal elements of d_scaled
    # log(Delta) = sum_{i!=j} log(d_scaled_ij)
    # log(d_scaled_ij) = log(d_ij) + log(2.0) - log(dia)
    
    # Set diagonal to 1 to avoid log(0)
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1.0)
    
    log_prod_scaled = np.log(d_safe).sum() + _N * (_N - 1) * (math.log(2.0) - math.log(dia))
    
    # Score = exp(log_prod_scaled) / (8^8)
    # log(Score) = log_prod_scaled - 8 * log(8)
    log_score = log_prod_scaled - _N * math.log(_N)
    
    # Return actual score, not log, to match "raw value" expectation, 
    # but for optimization we might want to return -log_score to avoid underflow.
    # The evaluator measures "raw value".
    # If log_score is very negative, exp will be 0.
    # If log_score is around 0, exp is around 1.
    # Score 1.0 corresponds to log_score = 0.
    # Score 1.25 corresponds to log_score = log(1.25) ~ 0.22.
    
    return math.exp(log_score)

def _objective_generic(pts_flat):
    pts = pts_flat.reshape(_N, 2)
    # Penalty for duplicates
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e12
    s = _score(pts)
    return -s

def _get_points_D2(params):
    """
    Generate 8 points with D2 symmetry.
    params: [a, b, c, d]
    Points: (±a, ±b) and (±c, ±d)
    """
    a, b, c, d = params
    # Ensure positive to avoid duplicate sets from sign flips, 
    # though the optimizer might find negative values, the geometry is same.
    # We just construct the 8 points.
    p1 = np.array([a, b])
    p2 = np.array([-a, b])
    p3 = np.array([-a, -b])
    p4 = np.array([a, -b])
    p5 = np.array([c, d])
    p6 = np.array([-c, d])
    p7 = np.array([-c, -d])
    p8 = np.array([c, -d])
    return np.array([p1, p2, p3, p4, p5, p6, p7, p8])

def _objective_D2(params):
    pts = _get_points_D2(params)
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-6:
        return 1e12
    s = _score(pts)
    return -s

def _refine_generic(pts, maxiter=2000):
    """
    Refine a full 8-point configuration using L-BFGS-B.
    """
    def obj(x):
        p = x.reshape(_N, 2)
        d = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
            return 1e12
        return -_score(p)
        
    try:
        res = minimize(obj, pts.ravel(), method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12})
        cand = res.x.reshape(_N, 2)
        # Check validity
        d = np.linalg.norm(cand[:, np.newaxis, :] - cand[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, 1e12)
        if d.min() < 1e-6:
             return pts, _score(pts)
        return cand, _score(cand)
    except Exception:
        return pts, _score(pts)

def construct():
    t0 = time.time()
    best_pts = None
    best_score = 0.0
    
    # 1. Optimize D2 symmetric configuration (4 variables)
    # Bounds: a,b,c,d in [0.1, 10]
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]
    
    # Use Differential Evolution for global search in the 4D space
    try:
        res_de = differential_evolution(
            _objective_D2, bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-8,
            polish=True,
            popsize=15
        )
        if res_de.fun < 1e10:
            best_params = res_de.x
            pts = _get_points_D2(best_params)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    except Exception as e:
        pass

    # 2. Refine the D2 result using generic optimizer (breaks symmetry if beneficial)
    if best_pts is not None and _remaining() > 20:
        cand, s = _refine_generic(best_pts, maxiter=1000)
        if s > best_score:
            best_score = s
            best_pts = cand

    # 3. Try multiple random D2 starts and refine
    rng = np.random.default_rng(12345)
    num_random_starts = 20
    for i in range(num_random_starts):
        if _remaining() < 20: break
        
        # Random parameters
        # We can sample from a wide range
        a = rng.uniform(0.5, 5.0)
        b = rng.uniform(0.5, 5.0)
        c = rng.uniform(0.5, 5.0)
        d = rng.uniform(0.5, 5.0)
        
        # Ensure some diversity, maybe swap or scale
        params = [a, b, c, d]
        pts = _get_points_D2(params)
        
        # Quick local refinement
        cand, s = _refine_generic(pts, maxiter=200)
        if s > best_score:
            best_score = s
            best_pts = cand

    # 4. If we have time, try a few generic random starts (16 variables)
    # This is expensive, so only do if time permits.
    if best_pts is not None and _remaining() > 50:
        # Use the best points as a base and add noise
        for i in range(5):
            if _remaining() < 20: break
            noisy = best_pts + rng.normal(0, 0.5, size=(8, 2))
            cand, s = _refine_generic(noisy, maxiter=500)
            if s > best_score:
                best_score = s
                best_pts = cand

    # 5. Fallback to regular octagon if nothing worked
    if best_pts is None:
        best_pts = np.array([[math.cos(2*math.pi*k/8), math.sin(2*math.pi*k/8)] for k in range(8)])
        best_score = _score(best_pts)

    # Final check and cleanup
    best_pts = np.asarray(best_pts, dtype=float).reshape(_N, 2)
    
    # Ensure distinct points
    d = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, 1e12)
    if d.min() < 1e-5:
         # Perturb if too close
         for i in range(_N):
             for j in range(i+1, _N):
                 if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-5:
                     best_pts[j] += [1e-5, 0]
                     
    return [[float(x), float(y)] for (x, y) in best_pts]
