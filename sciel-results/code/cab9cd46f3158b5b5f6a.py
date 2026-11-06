"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Soft-max power for smooth diameter approximation
    P = 30.0  # Large power approximates max while remaining smooth
    
    def compute_score(pts):
        """Compute the actual score for a set of points."""
        pts = np.asarray(pts, dtype=np.float64)
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
        log_prod = np.sum(np.log(scaled_d))
        # Delta = product over ordered pairs = (product over unordered)^2
        log_delta = 2.0 * log_prod
        score = math.exp(log_delta - n * math.log(n))
        return score
    
    def objective(x):
        """Objective: minimize negative log score using soft-max for diameter."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return 1e10
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-30)
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return 1e10
        
        # Soft-max approximation of diameter
        # dmax_soft = (sum(d^P))^(1/P)
        dP = d ** P
        dmax_soft = (np.sum(dP)) ** (1.0 / P)
        if dmax_soft < 1e-10:
            return 1e10
        
        # log_score = 2*(sum(log(d)) - 28*log(dmax_soft) + 28*log(2)) - 8*log(8)
        # We minimize -log_score
        log_score = 2.0 * (np.sum(np.log(d)) - 28.0 * np.log(dmax_soft) + 28.0 * np.log(2.0) - 8.0 * np.log(8.0))
        return -log_score
    
    def gradient(x):
        """Analytical gradient of the objective."""
        pts = x.reshape(n, 2)
        if not np.all(np.isfinite(pts)):
            return np.zeros(16)
        
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-30)
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        if np.min(d) < 1e-10:
            return np.zeros(16)
        
        dmax_soft = (np.sum(d ** P)) ** (1.0 / P)
        if dmax_soft < 1e-10:
            return np.zeros(16)
        
        # Gradient of log_prod = sum(log(d_ij))
        # d(log d_ij)/d(pts_k) = (pts_k - pts_j)/d_ij^2 for k=i, (pts_j - pts_i)/d_ij^2 for k=j
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(i+1, n):
                d2 = dists[i, j] ** 2
                if d2 < 1e-20:
                    continue
                diff_ij = pts[i] - pts[j]
                # Each unordered pair contributes to ordered product squared
                # log_score has factor 2, so gradient is doubled
                grad[i] += 2.0 * diff_ij / d2
                grad[j] += 2.0 * (-diff_ij) / d2
        
        # Gradient of -28*log(dmax_soft)
        # dmax_soft = (sum(d^P))^(1/P)
        # d(dmax_soft)/d(d_k) = (1/P) * (sum(d^P))^(1/P - 1) * P * d_k^(P-1)
        #                      = dmax_soft^(P-1) / sum(d^P) * d_k^(P-1)
        #                      = d_k^(P-1) / (sum(d^P)^(P-1)/P)
        # Simpler: d(log dmax_soft)/d(d_k) = d_k^(P-1) / sum(d^P) * (1/P) * P
        #         = d_k^(P-1) / sum(d^P)
        # Actually: log(dmax_soft) = (1/P) * log(sum(d^P))
        # d/d(d_k) = (1/P) * (P * d_k^(P-1)) / sum(d^P) = d_k^(P-1) / sum(d^P)
        
        sum_dP = np.sum(d ** P)
        if sum_dP < 1e-30:
            return (-grad).ravel()
        
        # d(log dmax_soft)/d(d_k) for each unordered pair k
        # Then d(d_k)/d(pts_i) = (pts_i - pts_j)/d_k, d(d_k)/d(pts_j) = (pts_j - pts_i)/d_k
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            dk = d[idx]
            if dk < 1e-15:
                continue
            dlog_ddk = (dk ** (P - 1)) / sum_dP
            diff_ij = pts[i] - pts[j]
            # d(log dmax_soft)/d(pts_i) = dlog_ddk * (pts_i - pts_j)/dk
            grad[i] -= 28.0 * dlog_ddk * diff_ij / dk
            grad[j] -= 28.0 * dlog_ddk * (-diff_ij) / dk
        
        # Objective is -log_score, so gradient of objective = -gradient of log_score
        return (-grad).ravel()
    
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
    candidates = []
    
    # 1. Regular Octagon
    candidates.append(reg_pts.copy())
    
    # 2. Two concentric squares with varying radii and angle offset
    for r1 in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        for r2 in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            for theta in [np.pi/4, 0.0, np.pi/8, 3*np.pi/8, np.pi/6, np.pi/3]:
                if time.time() > deadline - 15.0:
                    break
                angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
                angles2 = angles1 + theta
                pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
                pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
                pts = np.vstack([pts1, pts2])
                candidates.append(pts)
        
        if time.time() > deadline - 15.0:
            break
    
    # 3. Interlaced regular octagon variants (perturbed angles)
    np.random.seed(42)
    for trial in range(15):
        if time.time() > deadline - 15.0:
            break
        # Perturb angles and radii
        angles = angles_reg + 0.15 * np.random.randn(n)
        radii = 1.0 + 0.15 * np.random.randn(n)
        radii = np.abs(radii) + 0.3
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 4. Square + 4 points at midpoints of edges or diagonals
    for r_outer in [1.0, 1.2, 1.4]:
        for r_inner in [0.5, 0.7, 0.85]:
            if time.time() > deadline - 15.0:
                break
            angles_outer = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
            angles_inner = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
            pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
            pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
            pts = np.vstack([pts_outer, pts_inner])
            candidates.append(pts)
    
    # Refine each candidate with L-BFGS-B using the soft-max objective
    bounds = [(-5, 5)] * 16
    max_candidates = 80  # Limit total candidates to stay within time
    
    for i, init_pts in enumerate(candidates[:max_candidates]):
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
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if np.all(np.isfinite(res.x)):
                refined_pts = res.x.reshape(n, 2)
                s_ref = compute_score(refined_pts)
                if s_ref > best_score:
                    best_score = s_ref
                    best_pts = refined_pts.copy()
                    
                    # If significant improvement, do tighter refinement
                    if s_ref > 1.001 and time.time() < deadline - 3.0:
                        res2 = minimize(
                            objective,
                            res.x,
                            jac=gradient,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                        )
                        if np.all(np.isfinite(res2.x)):
                            refined_pts2 = res2.x.reshape(n, 2)
                            s_ref2 = compute_score(refined_pts2)
                            if s_ref2 > best_score:
                                best_score = s_ref2
                                best_pts = refined_pts2.copy()
        except Exception:
            pass
    
    # Final polish on the best solution found
    if time.time() < deadline - 2.0:
        try:
            final_res = minimize(
                objective,
                best_pts.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if np.all(np.isfinite(final_res.x)):
                final_pts = final_res.x.reshape(n, 2)
                s_final = compute_score(final_pts)
                if s_final > best_score:
                    best_score = s_final
                    best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Ensure distinct points and valid coordinates
    best_pts = np.asarray(best_pts, dtype=np.float64)
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
    
    return [[float(best_pts[i, 0]), float(best_pts[i, 1])] for i in range(n)]
# EVOLVE-BLOCK-END
