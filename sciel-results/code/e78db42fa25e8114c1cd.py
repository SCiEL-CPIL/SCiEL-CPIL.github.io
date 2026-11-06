import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts, dtype=float)
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
        # log_score = 2 * (sum(log(d)) + 28*log(2) - 28*log(D) - 8*log(8))
        # Constant terms don't affect minimization, but we include them for consistency
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    best_score = 0.0
    best_pts = None

    # 1. Structured Search: Two Concentric Squares (t, phi)
    # Outer square radius 1, inner square radius t, rotation phi
    def build_two_squares(t, phi):
        angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
        angles_inner = angles_outer + phi
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def neg_logscore_tp(x):
        t, phi = x
        if t <= 1e-6 or t > 5.0:
            return 1e10
        pts = build_two_squares(t, phi)
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

    # Multi-start for (t, phi)
    t_starts = np.linspace(0.1, 2.0, 10)
    phi_starts = np.linspace(0, np.pi/2, 5)
    
    for t0 in t_starts:
        if time.time() > deadline - 30:
            break
        for phi0 in phi_starts:
            if time.time() > deadline - 30:
                break
            x0 = np.array([t0, phi0])
            try:
                res = minimize(neg_logscore_tp, x0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-13})
                if np.all(np.isfinite(res.x)):
                    t2, phi2 = float(res.x[0]), float(res.x[1])
                    if t2 > 1e-6:
                        cand = build_two_squares(t2, phi2)
                        s2 = compute_score(cand)
                        if s2 > best_score:
                            best_score = s2
                            best_pts = cand
            except Exception:
                pass

    # 2. Global Search: Differential Evolution on 16 coords
    # We fix the center at origin to reduce symmetry breaking? No, let DE handle it.
    # To make DE more effective, we can scale the problem.
    # Let's run DE with a limited budget.
    if time.time() < deadline - 20:
        bounds = [(-2, 2)] * 16
        try:
            # Use a subset of the population to save time
            de_res = differential_evolution(
                objective, 
                bounds, 
                maxiter=50, 
                popsize=10, 
                tol=0.01,
                seed=42,
                polish=False # We will polish later
            )
            if de_res.success or np.isfinite(de_res.fun):
                cand = de_res.x.reshape(n, 2)
                s_de = compute_score(cand)
                if s_de > best_score:
                    best_score = s_de
                    best_pts = cand
        except Exception:
            pass

    # 3. Polish the best candidate
    if best_pts is not None and time.time() < deadline - 5:
        # Local polish with L-BFGS-B
        bounds = [(-10, 10)] * 16
        try:
            res_polish = minimize(
                objective, 
                best_pts.flatten(), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if np.all(np.isfinite(res_polish.x)):
                refined = res_polish.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass
        
        # Second polish pass with Powell for robustness
        if time.time() < deadline - 2:
            try:
                res_powell = minimize(
                    objective, 
                    best_pts.flatten(), 
                    method='Powell',
                    options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                if np.all(np.isfinite(res_powell.x)):
                    refined2 = res_powell.x.reshape(n, 2)
                    s_ref2 = compute_score(refined2)
                    if s_ref2 > best_score:
                        best_score = s_ref2
                        best_pts = refined2
            except Exception:
                pass

    # Fallback to regular octagon if nothing worked
    if best_pts is None:
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
        best_score = compute_score(best_pts)

    # Final validity check
    best_pts = np.asarray(best_pts, dtype=float)
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    if np.min(d) < 1e-6 or not np.all(np.isfinite(best_pts)):
        angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])

    # Ensure coordinate bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
