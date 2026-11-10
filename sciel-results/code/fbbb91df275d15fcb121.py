"""Program to maximize product of distances for 8 planar points at fixed diameter."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import cos, sin, pi, log, sqrt, exp
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta / n^n where Delta is product of ordered-pair distances after scaling to diameter 2."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam == 0:
        return 0.0
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    score = np.exp(log_prod) / (n ** n)
    return score

def _neg_log_score(points):
    """Negative log of the score for minimization."""
    n = len(points)
    pts = np.array(points, dtype=np.float64)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    diam = np.max(dists)
    if diam == 0:
        return 1e10
    scale = 2.0 / diam
    dists_scaled = dists * scale
    mask = ~np.eye(n, dtype=bool)
    log_prod = np.sum(np.log(dists_scaled[mask]))
    return n * log(n) - log_prod

def _two_squares_model(r1, r2, theta):
    """Generate 8 points: 4 on circle of radius r1 at angles 0, pi/2, pi, 3pi/2 + theta,
       4 on circle of radius r2 at angles pi/4, 3pi/4, 5pi/4, 7pi/4 + theta."""
    angles1 = np.array([0, pi/2, pi, 3*pi/2]) + theta
    angles2 = np.array([pi/4, 3*pi/4, 5*pi/4, 7*pi/4]) + theta
    pts1 = np.column_stack([r1 * cos(angles1), r1 * sin(angles1)])
    pts2 = np.column_stack([r2 * cos(angles2), r2 * sin(angles2)])
    return np.vstack([pts1, pts2])

def _two_squares_neglog(r1, r2, theta):
    """Negative log score for two-squares model."""
    pts = _two_squares_model(r1, r2, theta)
    return _neg_log_score(pts)

def _five_param_model(params):
    """5-parameter model: two concentric rotated squares with independent radii, orientations, and vertical offset.
    params = [r1, r2, theta1, theta2, y_offset]
    Square 1: 4 points at angles theta1, theta1+pi/2, theta1+pi, theta1+3pi/2 with radius r1, shifted by (0, y_offset)
    Square 2: 4 points at angles theta2, theta2+pi/2, theta2+pi, theta2+3pi/2 with radius r2, shifted by (0, -y_offset)
    """
    r1, r2, theta1, theta2, y_off = params
    if r1 < 0.01 or r2 < 0.01:
        return None
    
    angles1 = np.array([theta1, theta1 + pi/2, theta1 + pi, theta1 + 3*pi/2])
    angles2 = np.array([theta2, theta2 + pi/2, theta2 + pi, theta2 + 3*pi/2])
    
    pts1 = np.column_stack([r1 * cos(angles1), r1 * sin(angles1) + y_off])
    pts2 = np.column_stack([r2 * cos(angles2), r2 * sin(angles2) - y_off])
    
    return np.vstack([pts1, pts2])

def _five_param_neglog(params):
    """Negative log score for 5-parameter model."""
    pts = _five_param_model(params)
    if pts is None:
        return 1e10
    return _neg_log_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    n = 8
    best_score = 0.0
    best_pts = None
    
    # Regular octagon as baseline
    oct_pts = np.array([[cos(2 * pi * k / 8), sin(2 * pi * k / 8)] for k in range(8)])
    best_score = _score(oct_pts)
    best_pts = oct_pts.copy()
    
    rng = np.random.RandomState(42)
    
    # === PHASE 1: Differential evolution on 5-parameter model ===
    if time.time() < deadline - 80:
        # bounds for [r1, r2, theta1, theta2, y_offset]
        bounds = [(0.1, 3.0), (0.1, 3.0), (0, pi/2), (0, pi/2), (-1.0, 1.0)]
        
        try:
            de_result = differential_evolution(
                _five_param_neglog, bounds,
                seed=42, maxiter=200, popsize=20,
                tol=1e-10, polish=True,
                workers=1
            )
            if de_result.fun < 1e6:
                pts = _five_param_model(de_result.x)
                if pts is not None:
                    sc = _score(pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # === PHASE 2: Local optimization on 5-parameter model from best ===
    if time.time() < deadline - 50:
        # Try to extract 5-param from best_pts if it came from 5-param model
        # For now, just do local optimization on full 16-dim space from best_pts
        
        def objective(x):
            pts_x = x.reshape(n, 2)
            return _neg_log_score(pts_x)
        
        # SLSQP with diameter constraint
        def constraint_diameter(x):
            pts_x = x.reshape(n, 2)
            diff = pts_x[:, np.newaxis, :] - pts_x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            return 2.0 - np.max(dists)
        
        constraints = [{'type': 'ineq', 'fun': constraint_diameter}]
        
        try:
            result = minimize(objective, best_pts.flatten(), method='SLSQP',
                            constraints=constraints, options={'maxiter': 500, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
        
        # L-BFGS-B
        try:
            result = minimize(objective, best_pts.flatten(), method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-14})
            if result.fun < 1e6:
                opt_pts = result.x.reshape(n, 2)
                sc = _score(opt_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # === PHASE 3: Additional DE runs with different seeds on 5-param model ===
    if time.time() < deadline - 30:
        bounds = [(0.1, 3.0), (0.1, 3.0), (0, pi/2), (0, pi/2), (-1.0, 1.0)]
        for seed in [123, 456, 789]:
            if time.time() > deadline - 15:
                break
            try:
                de_result = differential_evolution(
                    _five_param_neglog, bounds,
                    seed=seed, maxiter=100, popsize=15,
                    tol=1e-10, polish=True
                )
                if de_result.fun < 1e6:
                    pts = _five_param_model(de_result.x)
                    if pts is not None:
                        sc = _score(pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = pts.copy()
            except Exception:
                pass
    
    # === PHASE 4: Perturbation-based polish ===
    if time.time() < deadline - 10:
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            noise = rng.randn(n, 2) * 0.05
            perturbed = best_pts + noise
            try:
                result = minimize(objective, perturbed.flatten(), method='SLSQP',
                                constraints=constraints, options={'maxiter': 200, 'ftol': 1e-14})
                if result.fun < 1e6:
                    opt_pts = result.x.reshape(n, 2)
                    sc = _score(opt_pts)
                    if sc > best_score:
                        best_score = sc
                        best_pts = opt_pts.copy()
            except Exception:
                pass
    
    # === PHASE 5: Fine polish ===
    if time.time() < deadline - 3:
        for step_size in [0.01, 0.005, 0.002]:
            if time.time() > deadline - 2:
                break
            for trial in range(30):
                if time.time() > deadline - 2:
                    break
                noise = rng.randn(n, 2) * step_size
                perturbed = best_pts + noise
                try:
                    result = minimize(objective, perturbed.flatten(), method='SLSQP',
                                    constraints=constraints, options={'maxiter': 100, 'ftol': 1e-15})
                    if result.fun < 1e6:
                        opt_pts = result.x.reshape(n, 2)
                        sc = _score(opt_pts)
                        if sc > best_score:
                            best_score = sc
                            best_pts = opt_pts.copy()
                except Exception:
                    pass
    
    return best_pts.tolist()
# EVOLVE-BLOCK-END
