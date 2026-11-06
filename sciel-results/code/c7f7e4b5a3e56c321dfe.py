import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140.0 

    # Precompute indices for unordered pairs
    iu = np.triu_indices(n, k=1)
    
    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts, dtype=float)
        if not np.all(np.isfinite(pts)):
            return 0.0
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
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
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        D = np.max(d)
        if D < 1e-10:
            return 1e10
        # log(Delta/n^n) = 2 * (sum(log(d_i)) - n*log(D) + n*log(2) - log(n^n))
        # = 2 * (sum(log(d_i)) - n*log(D) + n*log(2) - n*log(n))
        log_score = 2.0 * (np.sum(np.log(d)) - n * np.log(D) + n * np.log(2.0) - n * np.log(n))
        return -log_score

    best_score = 0.0
    best_pts = np.zeros((n, 2))

    # --- Phase 1: Structural Seeds (Fast) ---
    
    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    s_reg = compute_score(reg_pts)
    if s_reg > best_score:
        best_score = s_reg
        best_pts = reg_pts.copy()

    # 2. Two Concentric Squares (Optimized t, phi)
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
        return objective(pts.flatten())

    # Multi-start for two squares
    t_starts = np.linspace(0.2, 2.0, 10)
    phi_starts = np.linspace(0, np.pi/4, 5)
    for t0 in t_starts:
        if time.time() > deadline - 60: break
        for phi0 in phi_starts:
            if time.time() > deadline - 60: break
            try:
                res = minimize(neg_logscore_tp, np.array([t0, phi0]), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-13})
                if np.all(np.isfinite(res.x)):
                    cand = build_two_squares(float(res.x[0]), float(res.x[1]))
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass

    # 3. General 4+4 (Two squares with independent rotations and radii)
    def build_44(r1, r2, p1, p2):
        a1 = np.array([p1, p1 + np.pi/2, p1 + np.pi, p1 + 3.0*np.pi/2])
        a2 = np.array([p2, p2 + np.pi/2, p2 + np.pi, p2 + 3.0*np.pi/2])
        pts1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        return np.vstack([pts1, pts2])

    def neg_logscore_44(x):
        r1, r2, p1, p2 = x
        if r1 <= 1e-6 or r2 <= 1e-6 or r1 > 5.0 or r2 > 5.0:
            return 1e10
        pts = build_44(r1, r2, p1, p2)
        return objective(pts.flatten())

    # Multi-start for 4+4
    for r1_0 in np.linspace(0.3, 1.5, 5):
        if time.time() > deadline - 50: break
        for r2_0 in np.linspace(0.3, 1.5, 5):
            if time.time() > deadline - 50: break
            try:
                res = minimize(neg_logscore_44, np.array([r1_0, r2_0, 0.0, np.pi/4]), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-13})
                if np.all(np.isfinite(res.x)):
                    cand = build_44(float(res.x[0]), float(res.x[1]), float(res.x[2]), float(res.x[3]))
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_pts = cand
            except Exception:
                pass

    # --- Phase 2: Global Search (Differential Evolution) ---
    # This is the structural change: using a global optimizer on the full 16D space.
    # We use a tight time budget for this phase.
    if time.time() < deadline - 30:
        bounds = [(-2.0, 2.0)] * 16
        # We run a few short DE instances
        for seed in [1, 2, 3]:
            if time.time() > deadline - 30: break
            try:
                # popsize=5, maxiter=30 is very fast but might not converge.
                # However, combined with polishing, it can find new basins.
                res_de = differential_evolution(objective, bounds, 
                                                maxiter=30, 
                                                popsize=5, 
                                                seed=seed,
                                                tol=1e-8,
                                                polish=False) # We will polish manually
                if np.all(np.isfinite(res_de.x)):
                    cand = res_de.x.reshape(n, 2)
                    # Quick polish of DE result
                    res_pol = minimize(objective, cand.flatten(), method='L-BFGS-B',
                                       bounds=bounds, options={'maxiter': 200, 'ftol': 1e-14})
                    if np.all(np.isfinite(res_pol.x)):
                        cand_pol = res_pol.x.reshape(n, 2)
                        s = compute_score(cand_pol)
                        if s > best_score:
                            best_score = s
                            best_pts = cand_pol
            except Exception:
                pass

    # --- Phase 3: Final Polish ---
    # Polish the best candidate found so far with high precision
    if time.time() < deadline - 5:
        bounds = [(-5.0, 5.0)] * 16
        try:
            res_final = minimize(objective, best_pts.flatten(), method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12})
            if np.all(np.isfinite(res_final.x)):
                refined = res_final.x.reshape(n, 2)
                s_ref = compute_score(refined)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined
        except Exception:
            pass

    # Final validity check
    best_pts = np.asarray(best_pts, dtype=float)
    if not np.all(np.isfinite(best_pts)):
        best_pts = reg_pts
        
    diffs = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    d = dists[iu]
    if np.min(d) < 1e-6:
        best_pts = reg_pts

    # Ensure coordinate bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()

# EVOLVE-BLOCK-END
