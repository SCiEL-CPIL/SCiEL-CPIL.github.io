import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(pts):
        min_pair = np.min(pdist(pts)) / 2.0
        min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
        return min(min_pair, min_boundary)

    def radial_layer_init():
        """
        Deterministic radial layer construction.
        We place circles in concentric rings.
        Ring 0: 1 circle at center.
        Ring k (k>=1): m_k circles.
        
        We need sum(m_k) = 300.
        A common efficient packing for large N uses rings with m_k ~ 6k.
        Let's try to fit 300 points into rings.
        1 + 6 + 12 + 18 + ... + 6k = 1 + 6 * k(k+1)/2 = 1 + 3k(k+1).
        For k=10: 1 + 3*10*11 = 331. Too many.
        For k=9: 1 + 3*9*10 = 271. We need 29 more.
        So we can have 9 full rings (271 points) and distribute the remaining 29 
        into a 10th ring or adjust the inner rings.
        
        Actually, a better heuristic for "optimal" radial packing is to solve for 
        the radii R_k and counts m_k such that the minimum distance is maximized.
        
        Constraints:
        1. Distance from center to ring k: R_k.
        2. Distance between ring k and k+1: R_{k+1} - R_k >= 2r.
        3. Distance within ring k: 2 * R_k * sin(pi/m_k) >= 2r  => r <= R_k * sin(pi/m_k).
        4. Boundary: 1 - R_max >= r.
        
        We want to maximize r.
        Let's define the rings.
        We'll use a greedy or analytical approach to determine m_k.
        Given a target r, what is the max number of points we can fit?
        Or, given N=300, find the best configuration.
        
        Let's try a specific configuration:
        Ring 0: 1 point.
        Ring 1: 6 points.
        Ring 2: 12 points.
        Ring 3: 18 points.
        Ring 4: 24 points.
        Ring 5: 30 points.
        Ring 6: 36 points.
        Ring 7: 42 points.
        Ring 8: 48 points.
        Sum so far: 1 + 6*(1+2+3+4+5+6+7+8) = 1 + 6*36 = 217.
        Remaining: 300 - 217 = 83 points.
        Ring 9: 54 points. Sum = 271.
        Remaining: 29 points.
        Ring 10: 29 points.
        
        This creates 11 rings (including center).
        Let's solve for r and radii R_k.
        Variables: R_1, ..., R_10. (R_0 = 0).
        Constraints:
        R_{k+1} - R_k >= 2r
        R_k * sin(pi/m_k) >= r
        1 - R_{10} >= r
        
        To maximize r, we should tighten the constraints.
        Assume R_{k+1} - R_k = 2r for most k? 
        And R_k = r / sin(pi/m_k) for the limiting rings?
        
        This is a linear programming problem if we fix m_k.
        Maximize r
        Subject to:
        R_{k+1} - R_k - 2r >= 0
        R_k - r/sin(pi/m_k) >= 0
        R_{10} + r <= 1
        
        We can solve this LP for a fixed set of m_k.
        Then we can try different distributions of m_k for the last few rings 
        to see if a better r is found.
        """
        
        # Define a function to solve the LP for given ring counts
        def solve_lp(ring_counts):
            # ring_counts[0] is ignored (center), ring_counts[1] is ring 1, etc.
            # Let K = len(ring_counts) - 1
            K = len(ring_counts) - 1
            if K == 0:
                return 0.5, [0]
                
            # Variables: R_1...R_K, r
            # Total variables: K + 1
            # We want to maximize r.
            # Using scipy.optimize.linprog to minimize -r.
            
            c = np.zeros(K + 1)
            c[-1] = -1.0  # Maximize r => Minimize -r
            
            # Inequalities A_ub x <= b_ub
            A_ub = []
            b_ub = []
            
            # 1. R_{k+1} - R_k >= 2r  =>  -R_{k+1} + R_k + 2r <= 0
            for k in range(K - 1):
                row = np.zeros(K + 1)
                row[k] = 1.0      # R_k
                row[k+1] = -1.0   # -R_{k+1}
                row[-1] = 2.0     # 2r
                A_ub.append(row)
                b_ub.append(0.0)
                
            # 2. R_k >= r / sin(pi/m_k)  =>  -R_k + r/sin(pi/m_k) <= 0
            for k in range(K):
                m_k = ring_counts[k+1]
                if m_k <= 0:
                    return 0, [0]*K
                sin_val = np.sin(np.pi / m_k)
                if sin_val < 1e-10:
                    return 0, [0]*K
                inv_sin = 1.0 / sin_val
                row = np.zeros(K + 1)
                row[k] = -1.0
                row[-1] = inv_sin
                A_ub.append(row)
                b_ub.append(0.0)
                
            # 3. R_K + r <= 1
            row = np.zeros(K + 1)
            row[-2] = 1.0  # R_K
            row[-1] = 1.0  # r
            A_ub.append(row)
            b_ub.append(1.0)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            # Bounds: R_k >= 0, r >= 0
            bounds = [(0, None)] * (K + 1)
            
            try:
                res = minimize(lambda x: -x[-1], np.zeros(K+1), method='Nelder-Mead', options={'maxiter': 1000})
                # Actually, Nelder-Mead might be slow or get stuck. 
                # Let's use a simple bisection on r? 
                # Or just use the LP solver if available? SciPy has linprog.
                from scipy.optimize import linprog
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.success:
                    r_opt = res.x[-1]
                    radii = res.x[:-1]
                    return r_opt, radii
                else:
                    return 0, [0]*K
            except Exception:
                return 0, [0]*K

        # Candidate ring distributions
        # Base: 1, 6, 12, ..., 48, 54, 29
        # Let's try a few variations of the last few rings to optimize the "tail"
        # Since the outer rings have larger radius, they can hold more points for the same angular spacing.
        # But the constraint R_k * sin(pi/m_k) >= r is tighter for smaller m_k (larger angle).
        # Actually, for a fixed r, larger m_k allows smaller R_k.
        # We want to minimize the total radial extent sum(2r) + r/sin(pi/m_K).
        
        # Let's just try the standard hex-like growth and a few perturbations.
        base_counts = [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]
        
        best_r = 0
        best_radii = []
        best_counts = base_counts
        
        # Try a few variations
        variations = [
            [1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29],
            [1, 6, 12, 18, 24, 30, 36, 42, 48, 50, 33],
            [1, 6, 12, 18, 24, 30, 36, 42, 48, 48, 35],
            [1, 6, 12, 18, 24, 30, 36, 42, 48, 45, 38],
            [1, 6, 12, 18, 24, 30, 36, 42, 48, 42, 41],
            [1, 6, 12, 18, 24, 30, 36, 42, 48, 40, 43],
        ]
        
        for counts in variations:
            if sum(counts) != N:
                continue
            r_val, rad_vals = solve_lp(counts)
            if r_val > best_r:
                best_r = r_val
                best_radii = rad_vals
                best_counts = counts
        
        # Construct points from best configuration
        pts = []
        # Center
        pts.append([0.0, 0.0])
        
        for k in range(1, len(best_counts)):
            m_k = best_counts[k]
            R_k = best_radii[k-1]
            # Angular phase offset to maximize distance to neighbors in adjacent rings?
            # For now, start with 0 offset. We will refine phases later.
            angles = np.linspace(0, 2*np.pi, m_k, endpoint=False)
            # Add a small offset for odd rings to interleave?
            # Hex lattice has offset 0 for even rows, pi/6 for odd?
            # Let's try offset = 0 for all, then optimize.
            if k % 2 == 0:
                offset = 0
            else:
                offset = np.pi / m_k
            angles += offset
            x = R_k * np.cos(angles)
            y = R_k * np.sin(angles)
            pts.extend(zip(x, y))
            
        pts = np.array(pts)
        
        # Refinement: Optimize phases and radii using L-BFGS-B on the reduced parameter space
        # Parameters: R_1...R_K, phi_1...phi_K (phase offsets for each ring)
        # Total params: 2 * K
        K = len(best_counts) - 1
        params = np.zeros(2 * K)
        for k in range(K):
            params[k] = best_radii[k]
            # Initial phase from construction
            if (k+1) % 2 == 0:
                params[K+k] = 0
            else:
                params[K+k] = np.pi / best_counts[k+1]
                
        def objective(params):
            if time.time() > deadline:
                return 1e10
            radii = params[:K]
            phases = params[K:]
            # Ensure radii are ordered and positive
            if np.any(radii < 0):
                return 1e10
            # Check monotonicity? LP should have handled it, but during optimization it might violate.
            # If R_k < R_{k-1}, it's invalid. We can penalize heavily or clip.
            # Let's clip radii to be increasing.
            for i in range(1, K):
                if radii[i] < radii[i-1]:
                    radii[i] = radii[i-1]
            
            pts = []
            pts.append([0.0, 0.0])
            for k in range(K):
                m_k = best_counts[k+1]
                R_k = radii[k]
                phi = phases[k]
                angles = np.linspace(0, 2*np.pi, m_k, endpoint=False) + phi
                x = R_k * np.cos(angles)
                y = R_k * np.sin(angles)
                pts.extend(zip(x, y))
            pts = np.array(pts)
            
            # Compute radius
            min_pair = np.min(pdist(pts)) / 2.0
            min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
            r = min(min_pair, min_boundary)
            
            # Penalty if points go outside or overlap significantly? 
            # The objective is just -r. But if points overlap, r is negative? 
            # pdist gives positive distances. If points are very close, r is small.
            # If points are outside, min_boundary is negative.
            # So minimizing -r (maximizing r) naturally handles this.
            
            return -r

        # Gradient is hard to compute analytically for this reduced space due to the min operation.
        # We can use numerical gradient or just Nelder-Mead.
        # Given K is small (~10), Nelder-Mead on 20 variables is feasible.
        
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            if res.fun < objective(params):
                best_params = res.x
            else:
                best_params = params
        except Exception:
            best_params = params

        # Reconstruct points with best_params
        radii = best_params[:K]
        phases = best_params[K:]
        # Enforce monotonic radii
        for i in range(1, K):
            if radii[i] < radii[i-1]:
                radii[i] = radii[i-1]
                
        pts = []
        pts.append([0.0, 0.0])
        for k in range(K):
            m_k = best_counts[k+1]
            R_k = radii[k]
            phi = phases[k]
            angles = np.linspace(0, 2*np.pi, m_k, endpoint=False) + phi
            x = R_k * np.cos(angles)
            y = R_k * np.sin(angles)
            pts.extend(zip(x, y))
        pts = np.array(pts)
        
        # Final check: ensure all points are inside
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        mask = r_pts > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / r_pts[mask, np.newaxis] * 0.9999
            
        return pts

    # Step 1: Generate initial radial layer packing
    pts = radial_layer_init()
    
    # Step 2: Compute initial radius
    best_radius = compute_radius(pts)
    best_pts = pts.copy()
    
    # Step 3: Local refinement using full L-BFGS-B on all coordinates
    # This allows breaking the strict radial symmetry if beneficial.
    from scipy.optimize import minimize
    
    def smooth_min(x, alpha=50.0):
        x_clipped = np.clip(x, -100, 100)
        return -np.log(np.sum(np.exp(-alpha * x_clipped))) / alpha

    def neg_smooth_min(x_flat, alpha=50.0):
        if time.time() > deadline:
            return 1e10
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        all_vals = np.concatenate([pair_vals, boundary])
        return -smooth_min(all_vals, alpha)

    def compute_gradient(x_flat, alpha=50.0):
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        all_vals = np.concatenate([pair_vals, boundary])
        x_clipped = np.clip(all_vals, -100, 100)
        exp_vals = np.exp(-alpha * x_clipped)
        weights = exp_vals / np.sum(exp_vals)
        grad = np.zeros_like(x_flat)
        pair_count = len(pair_vals)
        for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
            d = dist[i, j]
            if d < 1e-10:
                continue
            w = weights[idx] * alpha
            grad[2*i] += w * (pts[i, 0] - pts[j, 0]) / (2.0 * d)
            grad[2*i+1] += w * (pts[i, 1] - pts[j, 1]) / (2.0 * d)
            grad[2*j] += w * (pts[j, 0] - pts[i, 0]) / (2.0 * d)
            grad[2*j+1] += w * (pts[j, 1] - pts[i, 1]) / (2.0 * d)
        for i in range(n):
            r = r_pts[i]
            if r < 1e-10:
                continue
            w = weights[pair_count + i] * alpha
            grad[2*i] += w * (-pts[i, 0] / r)
            grad[2*i+1] += w * (-pts[i, 1] / r)
        return grad

    x0 = best_pts.flatten()
    
    if time.time() < deadline - 30:
        result = minimize(
            neg_smooth_min,
            x0,
            jac=compute_gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
        )
        
        if result.fun < neg_smooth_min(x0):
            pts_new = result.x.reshape(-1, 2)
            r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
            mask = r_new > 1.0
            if np.any(mask):
                pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
            r_new_val = compute_radius(pts_new)
            if r_new_val > best_radius:
                best_radius = r_new_val
                best_pts = pts_new.copy()

    # Final validation
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask_out = r_final >= 1.0
    if np.any(mask_out):
        best_pts[mask_out] = best_pts[mask_out] / r_final[mask_out, np.newaxis] * 0.9999
    
    min_dist = np.min(pdist(best_pts))
    if min_dist < 1e-6:
        np.random.seed(123)
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < 1e-6:
                    best_pts[j] += np.random.normal(0, 1e-4, size=2)
    
    return best_pts
