import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

# EVOLVE-BLOCK-START
def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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
        # Product over ordered pairs is (product over unordered)^2
        log_prod_unordered = np.sum(np.log(scaled_d))
        log_delta = 2.0 * log_prod_unordered
        log_score = log_delta - n * np.log(n)
        return math.exp(log_score)

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
        log_score = 2.0 * (np.sum(np.log(d)) + 28.0 * np.log(2.0) - 28.0 * np.log(D) - 8.0 * np.log(8.0))
        return -log_score

    # Regular octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    best_score = compute_score(reg_pts)
    best_pts = reg_pts.copy()

    # Antipodal symmetric parameterization:
    # Point i = (r_i * cos(theta_i), r_i * sin(theta_i))
    # Point i+4 = (-r_i * cos(theta_i), -r_i * sin(theta_i))
    # This gives 4 radii + 4 angles = 8 parameters
    
    def antipodal_objective(params):
        """Objective for antipodal symmetric optimization."""
        if time.time() > deadline:
            return 1e10
        r = params[:4]
        theta = params[4:]
        
        # Build points
        pts = np.zeros((n, 2))
        for i in range(4):
            ri = r[i]
            ti = theta[i]
            pts[i] = [ri * math.cos(ti), ri * math.sin(ti)]
            pts[i + 4] = [-ri * math.cos(ti), -ri * math.sin(ti)]
        
        # Compute distances
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

    def antipodal_to_points(params):
        """Convert antipodal parameters to 8 points."""
        r = params[:4]
        theta = params[4:]
        pts = np.zeros((n, 2))
        for i in range(4):
            ri = r[i]
            ti = theta[i]
            pts[i] = [ri * math.cos(ti), ri * math.sin(ti)]
            pts[i + 4] = [-ri * math.cos(ti), -ri * math.sin(ti)]
        return pts

    # Initial guess: regular octagon in antipodal form
    # Angles: 0, pi/4, pi/2, 3pi/4 with radii 1
    x0_sym = np.array([1.0, 1.0, 1.0, 1.0,
                        0.0, np.pi/4, np.pi/2, 3.0*np.pi/4])
    
    # Try antipodal symmetric optimization with multiple restarts
    np.random.seed(42)
    num_sym_restarts = 20
    
    for i in range(num_sym_restarts):
        if time.time() > deadline - 10:
            break
        
        if i == 0:
            x0 = x0_sym.copy()
        else:
            # Perturb the regular octagon in antipodal space
            noise_r = np.random.normal(0, 0.15, size=4)
            noise_t = np.random.normal(0, 0.2, size=4)
            x0 = x0_sym + np.concatenate([noise_r, noise_t])
        
        try:
            res = minimize(antipodal_objective, x0, method='L-BFGS-B',
                          bounds=[(0.1, 5.0)]*4 + [(-np.pi, np.pi)]*4,
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if np.all(np.isfinite(res.x)):
                cand_pts = antipodal_to_points(res.x)
                s_cand = compute_score(cand_pts)
                if s_cand > best_score:
                    best_score = s_cand
                    best_pts = cand_pts.copy()
                    
                    # Tighter refinement if significant improvement
                    if s_cand > 1.001 and time.time() < deadline - 5:
                        res2 = minimize(antipodal_objective, res.x, method='L-BFGS-B',
                                       bounds=[(0.1, 5.0)]*4 + [(-np.pi, np.pi)]*4,
                                       options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12})
                        if np.all(np.isfinite(res2.x)):
                            cand_pts2 = antipodal_to_points(res2.x)
                            s_cand2 = compute_score(cand_pts2)
                            if s_cand2 > best_score:
                                best_score = s_cand2
                                best_pts = cand_pts2.copy()
        except Exception:
            pass

    # Also try full coordinate optimization from the best antipodal result
    if time.time() < deadline - 15:
        seeds = [best_pts, reg_pts]
        # Add some random perturbations
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            angles = angles_reg + 0.3 * np.random.randn(n)
            radii = 1.0 + 0.2 * np.random.randn(n)
            radii = np.abs(radii) + 0.1
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            seeds.append(pts)

        bounds = [(-5, 5)] * 16

        for init_pts in seeds:
            if time.time() > deadline - 5:
                break
            
            x0 = init_pts.flatten()
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                
                if np.all(np.isfinite(res.x)):
                    refined_pts = res.x.reshape(n, 2)
                    s_ref = compute_score(refined_pts)
                    if s_ref > best_score:
                        best_score = s_ref
                        best_pts = refined_pts.copy()
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
