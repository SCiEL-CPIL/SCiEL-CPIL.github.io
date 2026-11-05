import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    n = 8

    def compute_score(pts):
        """Calculate the actual score: Delta / n^n where Delta is product of pairwise distances after scaling to diameter 2."""
        pts = np.array(pts, dtype=np.float64)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        D = np.max(d)
        if D < 1e-9:
            return -np.inf
        scale = 2.0 / D
        d_scaled = d * scale
        d_scaled = np.maximum(d_scaled, 1e-15)
        log_delta = np.sum(np.log(d_scaled))
        delta = np.exp(log_delta)
        score = delta / (n ** n)
        return score

    def neg_log_score_16d(pts_flat):
        """Negative of log(Delta) - 28*log(D) for full 16D point coordinates."""
        if time.time() > deadline:
            return 1e10
        pts = pts_flat.reshape(n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 1e10
        D = np.max(d)
        if D < 1e-9:
            return 1e10
        log_prod = np.sum(np.log(d))
        return 28.0 * np.log(D) - log_prod

    def neg_log_score_polar(params):
        """
        Negative log score for polar coordinates.
        params: [r_2, ..., r_8, theta_2, ..., theta_8]
        Point 1 is fixed at (1, 0) i.e. r_1=1, theta_1=0.
        """
        if time.time() > deadline:
            return 1e10
        r = np.concatenate(([1.0], params[:7]))
        theta = np.concatenate(([0.0], params[7:14]))
        
        # Ensure r is positive
        if np.any(r < 1e-6):
            return 1e10
            
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        iu = np.triu_indices(n, 1)
        d = dists[iu]
        if np.min(d) < 1e-8:
            return 1e10
        D = np.max(d)
        if D < 1e-9:
            return 1e10
        log_prod = np.sum(np.log(d))
        return 28.0 * np.log(D) - log_prod

    best_score = -np.inf
    best_pts = None

    # Baseline: Regular Octagon
    angles_oct = 2 * np.pi * np.arange(8) / 8
    octagon = np.column_stack([np.cos(angles_oct), np.sin(angles_oct)])
    oct_score = compute_score(octagon)
    best_score = oct_score
    best_pts = octagon.copy()

    # Phase 1: Differential Evolution on Polar Coordinates (14D)
    # Fix point 1 at (1,0). Optimize r_2..r_8 and theta_2..theta_8.
    if time.time() < deadline - 60:
        bounds_polar = [(0.01, 5.0)] * 7 + [(0.0, 2 * np.pi)] * 7
        try:
            res_de = differential_evolution(
                neg_log_score_polar,
                bounds_polar,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False
            )
            if res_de.fun < 1e9:
                # Convert best polar params to cartesian
                r_best = np.concatenate(([1.0], res_de.x[:7]))
                theta_best = np.concatenate(([0.0], res_de.x[7:14]))
                cand_pts = np.column_stack([r_best * np.cos(theta_best), r_best * np.sin(theta_best)])
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    # Phase 2: Refinement with L-BFGS-B on Cartesian Coordinates (16D)
    if time.time() < deadline - 10:
        best_flat = best_pts.flatten()
        try:
            res_refine = minimize(
                neg_log_score_16d,
                best_flat,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res_refine.fun < 1e9:
                cand_pts = res_refine.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    # Phase 3: Final Polish with Nelder-Mead (robust to small deviations)
    if time.time() < deadline - 3:
        best_flat = best_pts.flatten()
        try:
            res_final = minimize(
                neg_log_score_16d,
                best_flat,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res_final.fun < 1e9:
                cand_pts = res_final.x.reshape(n, 2)
                cand_score = compute_score(cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = cand_pts
        except Exception:
            pass

    # Final validation
    if best_pts is None:
        angles = 2 * np.pi * np.arange(n) / n
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure distinct points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        angles = 2 * np.pi * np.arange(n) / n
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Ensure coordinates are within bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        scale = 1e6 / max_coord
        best_pts = best_pts * scale

    return [[float(p[0]), float(p[1])] for p in best_pts]
