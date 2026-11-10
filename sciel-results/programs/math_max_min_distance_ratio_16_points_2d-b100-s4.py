import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 15.0
    N = 16
    N_PAIRS = N * (N - 1) // 2

    def get_ratio(pts):
        """Calculate d_min / d_max. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return d_min / d_max

    def get_cost_fast(pts):
        """Calculate (d_max / d_min)^2. Lower is better. Optimized for small N."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(pts.shape[0], dtype=bool)
        
        dist_sq_min = np.where(mask, dist_sq, np.inf)
        d_min_sq = np.min(dist_sq_min)
        d_max_sq = np.max(dist_sq)
        
        if d_min_sq <= 1e-12 or d_max_sq <= 1e-12:
            return np.inf
            
        return d_max_sq / d_min_sq

    def generate_initial_candidates():
        """Generate diverse initial configurations."""
        candidates = []
        
        # 1. Circle
        angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
        circle = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(circle)
        
        # 2. Two concentric circles (4 inner, 12 outer)
        inner_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        outer_angles = np.linspace(0, 2*np.pi, 12, endpoint=False) + np.pi/12
        r_inner = 0.5
        r_outer = 1.0
        inner_pts = np.column_stack([r_inner * np.cos(inner_angles), r_inner * np.sin(inner_angles)])
        outer_pts = np.column_stack([r_outer * np.cos(outer_angles), r_outer * np.sin(outer_angles)])
        two_circles = np.vstack([inner_pts, outer_pts])
        candidates.append(two_circles)
        
        # 3. Three concentric circles (2, 6, 8)
        inner_angles = np.linspace(0, 2*np.pi, 2, endpoint=False)
        mid_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        r1, r2, r3 = 0.3, 0.7, 1.0
        pts1 = np.column_stack([r1 * np.cos(inner_angles), r1 * np.sin(inner_angles)])
        pts2 = np.column_stack([r2 * np.cos(mid_angles), r2 * np.sin(mid_angles)])
        pts3 = np.column_stack([r3 * np.cos(outer_angles), r3 * np.sin(outer_angles)])
        three_circles = np.vstack([pts1, pts2, pts3])
        candidates.append(three_circles)
        
        # 4. Hexagonal lattice subset
        a = 1.0
        hex_pts = []
        # Center
        hex_pts.append([0, 0])
        # Ring 1 (6 points)
        for i in range(6):
            angle = i * np.pi / 3
            hex_pts.append([a * np.cos(angle), a * np.sin(angle)])
        # Ring 2 (12 points, take 9)
        ring2 = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + 0.5 * a * j
                y = np.sqrt(3)/2 * a * j
                r = np.sqrt(x*x + y*y)
                if 1.8 * a < r < 2.2 * a:
                    ring2.append([x, y])
        # Sort by angle and take 9
        ring2 = np.array(ring2)
        angles_r2 = np.arctan2(ring2[:,1], ring2[:,0])
        idx = np.argsort(angles_r2)[:9]
        hex_pts = np.array(hex_pts)
        hex_pts = np.vstack([hex_pts, ring2[idx]])
        hex_pts = hex_pts[:16]
        candidates.append(hex_pts)
        
        # 5. Square grid 4x4
        grid = []
        for i in range(4):
            for j in range(4):
                grid.append([i - 1.5, j - 1.5])
        grid = np.array(grid)
        candidates.append(grid)
        
        # 6. Random
        np.random.seed(42)
        random_pts = np.random.randn(16, 2)
        candidates.append(random_pts)
        
        # Normalize all candidates to unit max radius
        for i, c in enumerate(candidates):
            c = c - np.mean(c, axis=0)
            max_r = np.max(np.linalg.norm(c, axis=1))
            if max_r > 0:
                c = c / max_r
            candidates[i] = c
            
        return candidates

    def simulated_annealing(initial_pts, time_budget):
        """
        Fast local search using Simulated Annealing.
        Uses squared distances for speed.
        """
        pts = initial_pts.copy()
        current_cost = get_cost_fast(pts)
        best_pts_sa = pts.copy()
        best_cost_sa = current_cost
        
        # Adaptive temperature
        T = 0.5
        cooling_rate = 0.9999
        min_T = 1e-8
        no_improve_count = 0
        max_no_improve = 20000
        
        t_start = time.time()
        
        while time.time() - t_start < time_budget:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            idx = np.random.randint(0, len(pts))
            noise = np.random.randn(2) * T
            old_pos = pts[idx].copy()
            pts[idx] += noise
            
            new_cost = get_cost_fast(pts)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_pts_sa = pts.copy()
            else:
                delta = new_cost - current_cost
                if T > min_T:
                    if delta > T * 50: 
                        accept = False
                    else:
                        prob = np.exp(-delta / T)
                        accept = np.random.rand() < prob
                else:
                    accept = False
                
                if accept:
                    current_cost = new_cost
                else:
                    pts[idx] = old_pos
                    no_improve_count += 1
                    
            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                pts = best_pts_sa.copy()
                current_cost = best_cost_sa
                T = 0.1
                no_improve_count = 0
                max_no_improve = 10000

        return best_pts_sa, best_cost_sa

    def scp_bisection(pts, time_budget=100.0):
        """
        Sequential Convex Programming with Bisection on minimum distance constraint.
        
        Strategy:
        1. Normalize points so d_max = 1.
        2. Binary search for the largest t such that there exists a configuration 
           with d_min >= t and d_max <= 1.
        3. The subproblem for a fixed t is: Minimize d_max subject to d_ij >= t.
           This is convex if we relax d_max to a smooth surrogate or use L-BFGS-B 
           with a penalty for violations of d_ij >= t.
        
        We will use a penalty method for the constraints in the subproblem.
        Objective: f(x) = d_max(x) + lambda * sum( max(0, t - d_ij(x))^2 )
        """
        pts = pts.copy()
        # Normalize
        d = pdist(pts)
        d_max_init = np.max(d)
        if d_max_init < 1e-10:
            return pts, get_ratio(pts)
        pts = pts / d_max_init
        
        # Find initial feasible t (current d_min)
        d_min_init = np.min(d)
        
        def solve_subproblem(t_target, x0, time_limit):
            """
            Solve: Minimize d_max subject to d_ij >= t_target.
            Using penalty method with L-BFGS-B.
            """
            def obj(x):
                p = x.reshape(N, 2)
                d = pdist(p)
                d_max = np.max(d)
                # Penalty for d_ij < t_target
                viol = np.maximum(0, t_target - d)
                penalty = np.sum(viol**2)
                # We want to minimize d_max, but the penalty forces d_min up.
                # If we can't satisfy d_min >= t_target with d_max <= 1, 
                # the penalty will be high.
                return d_max + 100.0 * penalty
            
            def grad(x):
                p = x.reshape(N, 2)
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dist_mat = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist_mat, 0)
                
                d_max_val = np.max(dist_mat)
                mask_max = dist_mat > d_max_val - 1e-6
                
                grad = np.zeros_like(x)
                
                # Gradient of d_max
                for i in range(N):
                    for j in range(i+1, N):
                        if mask_max[i, j]:
                            d_ij = dist_mat[i, j]
                            if d_ij < 1e-10: continue
                            diff_ij = p[i] - p[j]
                            grad_ij = diff_ij / d_ij
                            grad[2*i] += grad_ij[0]
                            grad[2*i+1] += grad_ij[1]
                            grad[2*j] -= grad_ij[0]
                            grad[2*j+1] -= grad_ij[1]
                        
                        # Gradient of penalty
                        if dist_mat[i, j] < t_target:
                            d_ij = dist_mat[i, j]
                            if d_ij < 1e-10: continue
                            diff_ij = p[i] - p[j]
                            grad_ij = diff_ij / d_ij
                            coeff = -2.0 * (t_target - d_ij) * 100.0
                            grad[2*i] += coeff * grad_ij[0]
                            grad[2*i+1] += coeff * grad_ij[1]
                            grad[2*j] -= coeff * grad_ij[0]
                            grad[2*j+1] -= coeff * grad_ij[1]
                
                return grad
            
            try:
                res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                               options={'maxiter': 50, 'maxls': 20, 'gtol': 1e-5})
                new_pts = res.x.reshape(N, 2)
                # Check feasibility
                d_new = pdist(new_pts)
                d_min_new = np.min(d_new)
                d_max_new = np.max(d_new)
                
                # If feasible (d_min >= t_target - tol) and d_max is reasonable
                if d_min_new >= t_target - 1e-4:
                    # Normalize back to unit max? No, keep scale consistent for bisection.
                    # Actually, we normalized input to d_max=1. The subproblem tries to keep d_max low.
                    # If it succeeds, the ratio is d_min_new / d_max_new.
                    return new_pts, d_min_new / d_max_new if d_max_new > 1e-10 else 0
                else:
                    return None, 0
            except Exception:
                return None, 0

        t_start = time.time()
        
        # Bisection range: [0, 1] (since d_max=1)
        # We know d_min_init is feasible.
        t_low = d_min_init
        t_high = 1.0
        best_pts = pts.copy()
        best_ratio = get_ratio(pts)
        
        # Perform a few bisection steps
        for _ in range(10):
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            t_mid = (t_low + t_high) / 2.0
            # Time per subproblem
            time_per_sub = max(1.0, (time_budget - (time.time() - t_start)) / 2.0)
            
            res_pts, res_ratio = solve_subproblem(t_mid, best_pts.flatten(), time_per_sub)
            
            if res_pts is not None:
                # Feasible, try higher t
                t_low = t_mid
                if res_ratio > best_ratio:
                    best_ratio = res_ratio
                    best_pts = res_pts.copy()
            else:
                # Infeasible, try lower t
                t_high = t_mid
        
        return best_pts, best_ratio

    # Main loop
    best_pts = None
    best_ratio = 0.0

    candidates = generate_initial_candidates()
    for c in candidates:
        if len(c) != 16:
            continue
        ratio = get_ratio(c)
        if ratio > best_ratio:
            best_ratio = ratio
            best_pts = c.copy()

    if best_pts is None:
        best_pts = np.random.randn(16, 2)

    # Phase 1: Simulated Annealing (Global Exploration)
    time_elapsed = time.time() - start_time
    sa_budget = min(300.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 100.0)
    if sa_budget > 10:
        sa_pts, sa_cost = simulated_annealing(best_pts, sa_budget)
        sa_ratio = get_ratio(sa_pts)
        if sa_ratio > best_ratio:
            best_pts = sa_pts
            best_ratio = sa_ratio

    # Phase 2: SCP Bisection (Local Refinement / Structural Optimization)
    time_elapsed = time.time() - start_time
    scp_budget = min(200.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN - 20.0)
    if scp_budget > 10:
        scp_pts, scp_ratio = scp_bisection(best_pts, scp_budget)
        if scp_ratio > best_ratio:
            best_pts = scp_pts
            best_ratio = scp_ratio

    # Phase 3: Final Polish with SA on the SCP result
    time_elapsed = time.time() - start_time
    polish_budget = TIME_LIMIT - time_elapsed - SAFETY_MARGIN
    if polish_budget > 10:
        final_pts, final_cost = simulated_annealing(best_pts, polish_budget)
        final_ratio = get_ratio(final_pts)
        if final_ratio > best_ratio:
            best_pts = final_pts
            best_ratio = final_ratio

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = np.random.randn(16, 2)

    return final_pts
