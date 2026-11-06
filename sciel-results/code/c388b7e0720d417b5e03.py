import time
import math
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    start_time = time.time()
    deadline = start_time + 140.0

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

    def objective_fixed_diameter(x, p_idx, q_idx):
        """
        x: 12 coordinates for the 6 variable points.
        p_idx, q_idx: indices of the fixed diameter points.
        Returns negative sum of log distances.
        """
        # Reconstruct full points array
        pts = np.zeros((n, 2))
        pts[p_idx] = [-1.0, 0.0]
        pts[q_idx] = [1.0, 0.0]
        
        # Fill variable points
        var_indices = [i for i in range(n) if i != p_idx and i != q_idx]
        for i, idx in enumerate(var_indices):
            pts[idx] = x[2*i : 2*i+2]
            
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        if np.min(d) < 1e-12:
            return 1e10
            
        # We want to maximize sum(log(d)), so minimize -sum(log(d))
        # Note: The diameter is fixed at 2 by construction of p, q.
        # We must ensure no other distance exceeds 2 (handled by constraints).
        return -np.sum(np.log(d))

    def constraints_fixed_diameter(x, p_idx, q_idx):
        """
        Constraints: all pairwise distances <= 2.
        Returns list of dicts for SLSQP.
        """
        pts = np.zeros((n, 2))
        pts[p_idx] = [-1.0, 0.0]
        pts[q_idx] = [1.0, 0.0]
        
        var_indices = [i for i in range(n) if i != p_idx and i != q_idx]
        for i, idx in enumerate(var_indices):
            pts[idx] = x[2*i : 2*i+2]
            
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        
        cons = []
        for k in range(len(d)):
            i, j = iu[0][k], iu[1][k]
            # We only need to constrain pairs that are NOT the fixed diameter pair
            # because the fixed pair is exactly 2.
            # Actually, we can constrain all, but the fixed one is equality.
            # SLSQP handles inequalities.
            # d_ij <= 2  =>  2 - d_ij >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j, p_idx=p_idx, q_idx=q_idx: 2.0 - _get_dist(x, p_idx, q_idx, i, j)
            })
        return cons

    def _get_dist(x, p_idx, q_idx, i, j):
        pts = np.zeros((n, 2))
        pts[p_idx] = [-1.0, 0.0]
        pts[q_idx] = [1.0, 0.0]
        var_indices = [k for k in range(n) if k != p_idx and k != q_idx]
        for k, idx in enumerate(var_indices):
            pts[idx] = x[2*k : 2*k+2]
        return np.linalg.norm(pts[i] - pts[j])

    # This constraint formulation is very slow because it rebuilds pts every time.
    # Better to vectorize or pass pts.
    # Let's rewrite objective and constraints to be efficient.

    def make_optimizers(p_idx, q_idx):
        var_indices = [i for i in range(n) if i != p_idx and i != q_idx]
        
        def obj(x):
            pts = np.zeros((n, 2))
            pts[p_idx] = [-1.0, 0.0]
            pts[q_idx] = [1.0, 0.0]
            for i, idx in enumerate(var_indices):
                pts[idx] = x[2*i : 2*i+2]
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            iu = np.triu_indices(n, k=1)
            d = dists[iu]
            if np.min(d) < 1e-12:
                return 1e10
            return -np.sum(np.log(d))
            
        def cons_func(x):
            pts = np.zeros((n, 2))
            pts[p_idx] = [-1.0, 0.0]
            pts[q_idx] = [1.0, 0.0]
            for i, idx in enumerate(var_indices):
                pts[idx] = x[2*i : 2*i+2]
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            iu = np.triu_indices(n, k=1)
            d = dists[iu]
            # Return 2 - d for all pairs
            return 2.0 - d
            
        # SLSQP expects a list of constraint dicts or a single function returning array for 'ineq'
        # If we return an array from 'fun' for 'ineq', it's not directly supported in standard SLSQP dict format.
        # We need to create a list of dicts or use a wrapper.
        # Creating 28 dicts is slow but acceptable for 10-20 runs.
        
        def make_cons_list():
            iu = np.triu_indices(n, k=1)
            cons = []
            for k in range(len(iu[0])):
                i, j = iu[0][k], iu[1][k]
                # Skip the fixed diameter pair constraint? 
                # The fixed pair is exactly 2. 2-2=0. It's on the boundary.
                # It's fine to include it.
                def c_fun(x, i=i, j=j):
                    pts = np.zeros((n, 2))
                    pts[p_idx] = [-1.0, 0.0]
                    pts[q_idx] = [1.0, 0.0]
                    for m, idx in enumerate(var_indices):
                        pts[idx] = x[2*m : 2*m+2]
                    return 2.0 - np.linalg.norm(pts[i] - pts[j])
                cons.append({'type': 'ineq', 'fun': c_fun})
            return cons

        return obj, make_cons_list()

    best_score = 0.0
    best_pts = None

    # Generate initial candidates
    candidates = []

    # 1. Regular Octagon
    angles_reg = np.linspace(0, 2 * np.pi, n, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    candidates.append(reg_pts)

    # 2. Two Squares Family
    # t = R_inner / R_outer, phi = angular offset
    for t in np.linspace(0.2, 1.5, 10):
        for phi in np.linspace(0, np.pi/2, 5):
            angles_outer = np.array([0.0, np.pi/2, np.pi, 3.0*np.pi/2])
            angles_inner = angles_outer + phi
            pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
            pts_inner = t * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
            pts = np.vstack([pts_outer, pts_inner])
            candidates.append(pts)

    # 3. Two Rectangles Family
    # a_o, a_i = aspect ratios (width/height)
    for a_o in [1.0, 1.5, 2.0]:
        for a_i in [0.5, 1.0, 1.5]:
            for phi in [0.0, np.pi/4]:
                # Outer rectangle
                wo, ho = a_o, 1.0
                # Scale to fit in unit circle roughly? No, just define corners
                # Centered at origin
                x_o = wo / 2
                y_o = ho / 2
                pts_outer = np.array([
                    [-x_o, -y_o], [x_o, -y_o], [x_o, y_o], [-x_o, y_o]
                ])
                # Inner rectangle
                wi, hi = a_i, 1.0
                x_i = wi / 2
                y_i = hi / 2
                pts_inner = np.array([
                    [-x_i, -y_i], [x_i, -y_i], [x_i, y_i], [-x_i, y_i]
                ])
                # Rotate inner
                c, s = np.cos(phi), np.sin(phi)
                R = np.array([[c, -s], [s, c]])
                pts_inner = pts_inner @ R.T
                pts = np.vstack([pts_outer, pts_inner])
                candidates.append(pts)

    # Process candidates
    for init_pts in candidates:
        if time.time() > deadline - 20:
            break
            
        # Find diameter pair
        diffs = init_pts[:, np.newaxis, :] - init_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        iu = np.triu_indices(n, k=1)
        d = dists[iu]
        max_idx = np.argmax(d)
        p_idx = iu[0][max_idx]
        q_idx = iu[1][max_idx]
        
        # Transform to fix p, q
        # 1. Translate so midpoint is 0
        mid = (init_pts[p_idx] + init_pts[q_idx]) / 2.0
        pts_trans = init_pts - mid
        
        # 2. Rotate so p is at angle pi, q at 0
        # Vector q - p
        v = pts_trans[q_idx] - pts_trans[p_idx]
        angle = np.arctan2(v[1], v[0])
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])
        pts_rot = pts_trans @ R.T
        
        # 3. Scale so distance pq is 2
        # Current distance is 2 * ||v||
        D_curr = np.linalg.norm(v)
        if D_curr < 1e-6:
            continue
        scale = 2.0 / D_curr
        pts_scaled = pts_rot * scale
        
        # Now p is at (-1, 0) and q is at (1, 0) approximately
        # Check
        # pts_scaled[p_idx] should be [-1, 0]
        # pts_scaled[q_idx] should be [1, 0]
        
        # Extract variable points
        var_indices = [i for i in range(n) if i != p_idx and i != q_idx]
        x0 = np.array([[pts_scaled[idx][0], pts_scaled[idx][1]] for idx in var_indices]).flatten()
        
        # Optimize
        try:
            obj, make_cons = make_optimizers(p_idx, q_idx)
            cons = make_cons()
            
            res = minimize(obj, x0, method='SLSQP', constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.fun < 1e9:
                # Reconstruct points
                res_pts = np.zeros((n, 2))
                res_pts[p_idx] = [-1.0, 0.0]
                res_pts[q_idx] = [1.0, 0.0]
                for i, idx in enumerate(var_indices):
                    res_pts[idx] = res.x[2*i : 2*i+2]
                
                # Check validity
                if np.all(np.isfinite(res_pts)):
                    # Check if any distance > 2 + tol
                    d_res = np.sqrt(np.sum((res_pts[:, np.newaxis, :] - res_pts[np.newaxis, :, :])**2, axis=2))
                    iu_r = np.triu_indices(n, k=1)
                    d_r = d_res[iu_r]
                    if np.max(d_r) <= 2.0 + 1e-6 and np.min(d_r) > 1e-6:
                        # The score calculation normalizes by D, but here D is fixed to 2.
                        # So we just compute the product term.
                        # Score = Delta / 8^8.
                        # Delta = (prod d)^2.
                        # prod d = prod(d_r).
                        prod_d = np.prod(d_r)
                        Delta = prod_d ** 2
                        score = Delta / (8 ** 8)
                        
                        if score > best_score:
                            best_score = score
                            best_pts = res_pts
        except Exception:
            pass

    # Fallback if no valid points found
    if best_pts is None:
        best_pts = reg_pts
        best_score = compute_score(best_pts)

    # Final Polish: Run L-BFGS-B on the best point found, without fixing diameter, 
    # but using a smooth approximation of the max? 
    # Or just use the result from SLSQP.
    
    # Ensure coordinate bounds
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6:
        best_pts = best_pts / max_coord * 1e5

    return best_pts.tolist()
