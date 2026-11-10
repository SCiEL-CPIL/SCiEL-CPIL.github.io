"""Program to maximize the product of distances for 8 planar points with fixed diameter."""
import numpy as np
import time
from math import cos, sin, pi, log, sqrt
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Diameter
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    
    # Scale to diameter 2
    scale = 2.0 / diam
    dists_scaled = dists * scale
    
    # Product over ordered pairs (i != j)
    mask = ~np.eye(n, dtype=bool)
    # Avoid log(0) if points coincide
    if np.any(dists_scaled[mask] < 1e-12):
        return 0.0
        
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score_from_params(params):
    """
    Compute negative log score for a C4 symmetric configuration defined by:
    params = [r1, theta2, r2]
    Orbit 1: r1 * exp(i * (0, pi/2, pi, 3pi/2))
    Orbit 2: r2 * exp(i * (theta2, theta2 + pi/2, ...))
    """
    r1 = params[0]
    theta2 = params[1]
    r2 = params[2]
    
    # Generate points
    pts1 = np.array([
        [r1 * cos(0), r1 * sin(0)],
        [r1 * cos(pi/2), r1 * sin(pi/2)],
        [r1 * cos(pi), r1 * sin(pi)],
        [r1 * cos(3*pi/2), r1 * sin(3*pi/2)]
    ])
    
    pts2 = np.array([
        [r2 * cos(theta2), r2 * sin(theta2)],
        [r2 * cos(theta2 + pi/2), r2 * sin(theta2 + pi/2)],
        [r2 * cos(theta2 + pi), r2 * sin(theta2 + pi)],
        [r2 * cos(theta2 + 3*pi/2), r2 * sin(theta2 + 3*pi/2)]
    ])
    
    pts = np.vstack([pts1, pts2])
    
    # Calculate score
    n = 8
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam < 1e-12:
        return 1e10
        
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    
    # Check for zero distances
    if np.any(dists_scaled[mask] < 1e-12):
        return 1e10
        
    log_prod = np.sum(np.log(dists_scaled[mask]))
    neg_log_score = n * log(n) - log_prod
    return neg_log_score

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin (150s limit)
    
    n = 8
    best_score = 0.0
    best_pts = None
    
    def _neg_log_score_16d(pts):
        n = 8
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        diam = np.max(dists)
        if diam < 1e-12:
            return 1e10
        scale = 2.0 / diam
        dists_scaled = dists * scale
        mask = ~np.eye(n, dtype=bool)
        if np.any(dists_scaled[mask] < 1e-12):
            return 1e10
        log_prod = np.sum(np.log(dists_scaled[mask]))
        return n * log(n) - log_prod
    
    def _neg_log_score_from_params(params):
        r1 = params[0]
        theta2 = params[1]
        r2 = params[2]
        
        pts1 = np.array([
            [r1 * cos(0), r1 * sin(0)],
            [r1 * cos(pi/2), r1 * sin(pi/2)],
            [r1 * cos(pi), r1 * sin(pi)],
            [r1 * cos(3*pi/2), r1 * sin(3*pi/2)]
        ])
        pts2 = np.array([
            [r2 * cos(theta2), r2 * sin(theta2)],
            [r2 * cos(theta2 + pi/2), r2 * sin(theta2 + pi/2)],
            [r2 * cos(theta2 + pi), r2 * sin(theta2 + pi)],
            [r2 * cos(theta2 + 3*pi/2), r2 * sin(theta2 + 3*pi/2)]
        ])
        pts = np.vstack([pts1, pts2])
        return _neg_log_score_16d(pts)
    
    def _update_best(candidate_pts):
        nonlocal best_score, best_pts
        sc = _score(candidate_pts)
        if sc > best_score:
            best_score = sc
            best_pts = candidate_pts
        return sc
    
    # 1. Multi-start Powell optimization from diverse initializations
    initializations = []
    
    # Regular octagon
    initializations.append(np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)]))
    
    # Two concentric squares (C4 symmetric) - regular
    r1, theta2, r2 = 1.0, pi/4, 1.0
    pts1 = np.array([[r1 * cos(a), r1 * sin(a)] for a in [0, pi/2, pi, 3*pi/2]])
    pts2 = np.array([[r2 * cos(theta2 + a), r2 * sin(theta2 + a)] for a in [0, pi/2, pi, 3*pi/2]])
    initializations.append(np.vstack([pts1, pts2]))
    
    # Two concentric squares - deformed
    r1, theta2, r2 = 0.8, pi/4, 1.2
    pts1 = np.array([[r1 * cos(a), r1 * sin(a)] for a in [0, pi/2, pi, 3*pi/2]])
    pts2 = np.array([[r2 * cos(theta2 + a), r2 * sin(theta2 + a)] for a in [0, pi/2, pi, 3*pi/2]])
    initializations.append(np.vstack([pts1, pts2]))
    
    # Two concentric squares - another deformation
    r1, theta2, r2 = 1.2, pi/6, 0.8
    pts1 = np.array([[r1 * cos(a), r1 * sin(a)] for a in [0, pi/2, pi, 3*pi/2]])
    pts2 = np.array([[r2 * cos(theta2 + a), r2 * sin(theta2 + a)] for a in [0, pi/2, pi, 3*pi/2]])
    initializations.append(np.vstack([pts1, pts2]))
    
    # Random perturbations of regular octagon
    rng = np.random.RandomState(42)
    for _ in range(4):
        base = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        perturbed = base + rng.normal(0, 0.1, size=(8, 2))
        initializations.append(perturbed)
    
    # Random points
    for _ in range(4):
        random_pts = rng.uniform(-1.5, 1.5, size=(8, 2))
        initializations.append(random_pts)
    
    # Run Powell from each initialization
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 15.0:
            break
        
        try:
            def _obj(x_flat):
                return _neg_log_score_16d(x_flat.reshape(8, 2))
            
            res = minimize(_obj, init_pts.flatten(), method='Powell', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if res.fun < 1e6:
                candidate_pts = res.x.reshape(8, 2)
                _update_best(candidate_pts)
        except Exception as e:
            pass
    
    # 2. Differential Evolution on the full 16D space for broader exploration
    if time.time() < deadline - 25.0:
        bounds = [(-2.0, 2.0)] * 16
        
        def _de_objective(x_flat):
            if time.time() > deadline - 5.0:
                return 1e10
            pts = x_flat.reshape(8, 2)
            return _neg_log_score_16d(pts)
        
        try:
            de_res = differential_evolution(_de_objective, bounds, maxiter=80, popsize=20, 
                                            seed=42, tol=1e-8, polish=False, 
                                            mutation=(0.3, 1.0), recombination=0.8)
            
            if de_res.fun < 1e6:
                candidate_pts = de_res.x.reshape(8, 2)
                _update_best(candidate_pts)
                
                # Polish with Powell
                if time.time() < deadline - 10.0:
                    def _obj(x_flat):
                        return _neg_log_score_16d(x_flat.reshape(8, 2))
                    
                    slsqp_res = minimize(_obj, candidate_pts.flatten(), method='Powell', 
                                         options={'maxiter': 300, 'ftol': 1e-14, 'xtol': 1e-14})
                    if slsqp_res.fun < 1e6:
                        candidate_pts = slsqp_res.x.reshape(8, 2)
                        _update_best(candidate_pts)
        except Exception as e:
            pass
    
    # 3. Final high-precision polish pass on the best result
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            def _obj(x_flat):
                return _neg_log_score_16d(x_flat.reshape(8, 2))
            
            polish_res = minimize(_obj, best_pts.flatten(), method='Powell', 
                                 options={'maxiter': 500, 'ftol': 1e-15, 'xtol': 1e-15})
            if polish_res.fun < 1e6:
                candidate_pts = polish_res.x.reshape(8, 2)
                _update_best(candidate_pts)
        except Exception as e:
            pass
    
    # 4. Fallback: Regular Octagon
    if best_pts is None:
        best_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
        best_score = _score(best_pts)
        
    return best_pts.tolist()
