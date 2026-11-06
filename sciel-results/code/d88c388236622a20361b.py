import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_score(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-12:
            return 0.0
        D = np.max(d)
        if D < 1e-12:
            return 0.0
        scaled_d = d * (2.0 / D)
        prod_unordered = np.prod(scaled_d)
        Delta = prod_unordered ** 2
        return Delta / (n ** n)

    best_score = 0.0
    best_pts = None

    def try_candidate(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(pts)):
            return 0.0
        s = compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    try_candidate(reg_pts)

    # 2. Two Concentric Squares
    def make_two_squares(r1, r2, theta):
        angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
        angles2 = angles1 + theta
        pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def objective_two_squares(params):
        r1, r2, theta = params
        if r1 < 0.001 or r2 < 0.001:
            return 1e10
        pts = make_two_squares(r1, r2, theta)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Global search in 3D space (r1, r2, theta)
    bounds_ts = [(0.1, 1.5), (0.1, 1.5), (0.0, np.pi/2)]
    
    if time.time() < deadline - 30.0:
        try:
            res_de = differential_evolution(
                objective_two_squares,
                bounds_ts,
                seed=42,
                maxiter=100,
                tol=1e-10,
                polish=False, # We will polish manually
                init='sobol',
                workers=1
            )
            if res_de.success or res_de.fun < 1e9:
                r1_opt, r2_opt, theta_opt = res_de.x
                pts_opt = make_two_squares(r1_opt, r2_opt, theta_opt)
                try_candidate(pts_opt)
        except Exception:
            pass

    # Local refinement from best DE result and some guesses
    if best_pts is not None:
        # Extract r1, r2, theta from best_pts if it looks like two squares
        # For now, just use the DE result and a few manual guesses for L-BFGS-B
        pass

    guesses = [
        [0.8, 1.0, np.pi/4],
        [1.0, 0.8, np.pi/4],
        [0.9, 1.1, np.pi/4],
        [1.1, 0.9, np.pi/4],
        [0.7, 1.2, np.pi/4],
        [1.2, 0.7, np.pi/4],
        [0.95, 1.05, np.pi/4],
        [1.05, 0.95, np.pi/4],
    ]
    
    if res_de.success or res_de.fun < 1e9:
        guesses.append(res_de.x)
        
    for guess in guesses:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                objective_two_squares,
                guess,
                method='L-BFGS-B',
                bounds=bounds_ts,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.success or res.fun < 1e9:
                r1_opt, r2_opt, theta_opt = res.x
                pts_opt = make_two_squares(r1_opt, r2_opt, theta_opt)
                try_candidate(pts_opt)
        except Exception:
            pass

    # 3. General 16D Optimization (Polish)
    def objective_general(x):
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
        log_sum = np.sum(np.log(d))
        # Penalty for D > 2 (we scale later, but keeping D ~ 2 helps optimization)
        # Actually, the score is scale invariant. We can fix D=2 by scaling.
        # But L-BFGS-B needs a bounded domain.
        # Let's just maximize log(prod(d)) - 28*log(D)
        # This is equivalent to maximizing log(prod(d/D))
        return -(log_sum - 28.0 * np.log(D))

    if best_pts is not None:
        # Polish the best two-squares result
        x0 = best_pts.flatten()
        
        # Try a few different initial perturbations for general optimization
        inits = [x0]
        np.random.seed(1)
        for _ in range(5):
            inits.append(x0 + 0.01 * np.random.randn(2*n))
            
        bounds_general = [(-2.0, 2.0)] * (2 * n)

        for init in inits:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    objective_general,
                    init,
                    method='L-BFGS-B',
                    bounds=bounds_general,
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16}
                )
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    try_candidate(refined_pts)
            except Exception:
                pass

        # Final Nelder-Mead polish
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    objective_general,
                    best_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 10000, 'fatol': 1e-15, 'xatol': 1e-15, 'adaptive': True}
                )
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    try_candidate(refined_pts)
            except Exception:
                pass

    # Fallback
    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    # Final validation
    best_pts = np.asarray(best_pts)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts

    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
