"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        # Scale to diameter 2
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2  # Product over ordered pairs
        return Delta / (n ** n)

    def objective(x):
        """Objective for optimization: minimize negative log score."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        # log(score) = 2*(sum(log(scaled_d)) - n*log(n))
        # scaled_d = d * 2/D
        # log(scaled_d) = log(d) + log(2) - log(D)
        # sum(log(scaled_d)) = sum(log(d)) + 28*log(2) - 28*log(D)
        # log_score = 2 * (sum(log(d)) + 28*log(2) - 28*log(D) - 8*log(8))
        # We minimize -log_score
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Regular octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    best_score = compute_score(reg_pts)
    best_pts = reg_pts.copy()

    def try_candidate(pts):
        nonlocal best_score, best_pts
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # Generate structured initial candidates
    # Two concentric squares.
    # Square 1: angles 0, pi/2, pi, 3pi/2
    # Square 2: angles offset by theta
    
    candidates = []
    
    # 1. Regular Octagon
    candidates.append(reg_pts.copy())
    
    # 2. Two squares, interlaced (theta = pi/4), varying radii
    # We will optimize r1, r2, theta
    for r1 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
        for r2 in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
            for theta in [np.pi/4, 0.0, np.pi/8, 3*np.pi/8]:
                if time.time() > deadline - 20.0:
                    break
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = angles1 + theta
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                pts = np.vstack([pts1, pts2])
                candidates.append(pts)
        
        if time.time() > deadline - 20.0:
            break

    # 3. Random perturbations of regular octagon
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 20.0:
            break
        angles = angles_reg + 0.2 * np.random.randn(n)
        radii = 1.0 + 0.2 * np.random.randn(n)
        radii = np.abs(radii) + 0.1
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)

    # Refine each candidate
    bounds = [(-5, 5)] * 16
    
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline - 5.0:
            break
        
        x0 = init_pts.flatten()
        
        # Quick check if already good
        s_init = compute_score(init_pts)
        if s_init > best_score:
            best_score = s_init
            best_pts = init_pts.copy()

        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                s_ref = compute_score(refined_pts)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined_pts.copy()
                    
                    # If we found a significant improvement, do a tighter refinement
                    if s_ref > 1.001:
                         if time.time() < deadline - 2.0:
                             res2 = minimize(
                                 objective,
                                 res.x,
                                 method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
                             )
                             if np.all(np.isfinite(res2.x)):
                                 refined_pts2 = res2.x.reshape(n, 2)
                                 s_ref2 = compute_score(refined_pts2)
                                 if s_ref2 > best_score:
                                     best_score = s_ref2
                                     best_pts = refined_pts2.copy()
        except Exception:
            pass

    # Ensure distinct points and valid coordinates
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        # Fallback to regular octagon
        best_pts = reg_pts

    # Scale to ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
