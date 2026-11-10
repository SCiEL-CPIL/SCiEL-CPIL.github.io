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
    SAFETY_MARGIN = 30.0
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

    def get_ratio_sq(pts):
        """Calculate (d_min / d_max)^2. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def compute_distances(pts):
        """Compute all pairwise distances."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        return dist

    def objective_log_ratio(x):
        """
        Objective: minimize log(d_max) - log(d_min)
        This is equivalent to maximizing log(d_min/d_max).
        """
        pts = x.reshape(N, 2)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-15 or d_max <= 1e-15:
            return 1e10
        return np.log(d_max) - np.log(d_min)

    def objective_soft(x, temperature=0.01):
        """
        Soft objective using softmax/softmin for smooth gradients.
        """
        pts = x.reshape(N, 2)
        d = pdist(pts)
        if np.any(d <= 1e-15):
            return 1e10
        
        # Soft max: -log(sum(exp(-d/temperature))) * temperature
        # This approximates min(d) as temperature -> 0
        # Actually, soft_min = -temperature * log(sum(exp(-d/temperature)))
        # soft_max = temperature * log(sum(exp(d/temperature)))
        
        # We want to minimize soft_max(d) - soft_min(d)
        # But let's stick with log ratio for stability and use a penalty for close pairs
        
        d_min = np.min(d)
        d_max = np.max(d)
        base = np.log(d_max) - np.log(d_min)
        
        # Add penalty for pairs that are too close
        penalty = 0.0
        for di in d:
            if di < d_min * 1.1:
                penalty += (d_min - di)**2
        
        return base + 0.01 * penalty

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

    def sequential_convex_programming(pts, time_budget=200.0):
        """
        SCP: Minimize max distance subject to min distance >= constraint.
        Iteratively tighten the constraint.
        """
        pts = pts.copy()
        t_start = time.time()
        
        # Start with current d_min as constraint
        d = pdist(pts)
        d_min_current = np.min(d)
        d_max_current = np.max(d)
        
        # We want to maximize d_min/d_max.
        # Strategy: Fix d_max to 1 (by scaling). Then maximize d_min.
        # Or: Fix d_min to 1. Then minimize d_max.
        
        # Let's use the approach: Minimize d_max subject to d_ij >= t for all i,j.
        # We will bisect on t.
        
        # First, find a good upper bound for t (optimal d_min)
        # and lower bound.
        
        # Actually, let's just optimize the ratio directly with a smooth objective
        # and use the SCP idea to handle the non-smooth min/max by
        # identifying active constraints.
        
        # Better SCP approach:
        # 1. Identify the pair with minimum distance (active min constraint)
        # 2. Identify the pair with maximum distance (active max constraint)
        # 3. Solve a convex problem: minimize d_max subject to d_min >= current_d_min * (1 + epsilon)
        #    This is not convex because d_min is a min.
        
        # Let's stick to a robust optimization loop:
        # Alternate between:
        # A) Minimize max distance (keeping points distinct)
        # B) Maximize min distance (keeping max distance fixed)
        
        best_pts = pts.copy()
        best_ratio = get_ratio(pts)
        
        max_iters = 50
        for it in range(max_iters):
            if time.time() - t_start > time_budget:
                break
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            d = pdist(pts)
            d_min = np.min(d)
            d_max = np.max(d)
            
            if d_min <= 1e-10 or d_max <= 1e-10:
                break
            
            # Step 1: Minimize d_max, subject to d_ij >= d_min * 0.99
            # This is a convex problem if we fix the lower bound.
            # Minimize max_ij d_ij s.t. d_ij >= L for all i,j.
            # This can be solved via SOCP, but we only have L-BFGS-B.
            # Let's use a penalty method.
            
            # Objective: min max(d) + lambda * sum(max(0, L - d_ij)^2)
            # But max(d) is non-smooth.
            
            # Let's use the log-ratio objective which is smooth away from 0.
            # And add a barrier for d_ij < d_min_current * 0.95
            # This effectively prevents the min distance from dropping too much
            # while allowing the max distance to shrink.
            
            L = d_min * 0.95
            
            def obj_scp(x):
                p = x.reshape(N, 2)
                dist = pdist(p)
                d_max = np.max(dist)
                # Penalty for violating lower bound
                viol = np.maximum(0, L - dist)
                penalty = np.sum(viol**2)
                # We want to minimize d_max, but also keep ratio high.
                # If we just minimize d_max, points might collapse.
                # The penalty prevents collapse below L.
                # But we also want to maximize the ratio, so maybe we should
                # minimize d_max / d_min? No, that's the original.
                
                # Let's try: Minimize d_max, with hard constraint d_ij >= L.
                # If infeasible, relax L.
                return d_max + 10.0 * penalty
            
            def grad_scp(x):
                p = x.reshape(N, 2)
                dist = pdist(p)
                d_max = np.max(dist)
                viol = np.maximum(0, L - dist)
                penalty = np.sum(viol**2)
                obj_val = d_max + 10.0 * penalty
                
                grad = np.zeros_like(x)
                
                # Gradient of d_max
                # Find pair with max distance
                idx_max = np.argmax(dist)
                # Convert linear index to (i, j)
                # pdist returns upper triangle
                # i = floor((-1 + sqrt(1 + 8*idx)) / 2)
                # j = idx - i*(N-i-1)/2 - i
                # This is tricky to get right. Let's use a dense matrix approach for gradient.
                
                # Dense distance matrix
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dist_mat = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist_mat, 0)
                
                d_max_val = np.max(dist_mat)
                # Find all pairs with distance close to d_max
                mask_max = dist_mat > d_max_val - 1e-6
                # Gradient of max is sum of gradients of active terms
                # d/dx_k d_ij = (p_i - p_j) / d_ij for p_i, (p_j - p_i) / d_ij for p_j
                
                for i in range(N):
                    for j in range(i+1, N):
                        if mask_max[i, j]:
                            d_ij = dist_mat[i, j]
                            if d_ij < 1e-10:
                                continue
                            diff_ij = p[i] - p[j]
                            grad_ij = diff_ij / d_ij
                            grad[2*i] += grad_ij[0]
                            grad[2*i+1] += grad_ij[1]
                            grad[2*j] -= grad_ij[0]
                            grad[2*j+1] -= grad_ij[1]
                
                # Gradient of penalty
                # penalty = sum (L - d_ij)^2 for d_ij < L
                # d/dx_k penalty = -2 * sum (L - d_ij) * d(d_ij)/dx_k
                for i in range(N):
                    for j in range(i+1, N):
                        d_ij = dist_mat[i, j]
                        if d_ij < L:
                            diff_ij = p[i] - p[j]
                            if d_ij < 1e-10:
                                continue
                            grad_ij = diff_ij / d_ij
                            coeff = -2.0 * (L - d_ij) * 10.0
                            grad[2*i] += coeff * grad_ij[0]
                            grad[2*i+1] += coeff * grad_ij[1]
                            grad[2*j] -= coeff * grad_ij[0]
                            grad[2*j+1] -= coeff * grad_ij[1]
                
                return obj_val, grad
            
            x0 = pts.flatten()
            try:
                result = minimize(obj_scp, x0, jac=grad_scp, method='L-BFGS-B',
                                  options={'maxiter': 50, 'maxls': 20})
                if result.success or result.fun < 1e10:
                    new_pts = result.x.reshape(N, 2)
                    new_ratio = get_ratio(new_pts)
                    if new_ratio > best_ratio * 0.9: # Allow some degradation
                        pts = new_pts
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_pts = new_pts.copy()
            except Exception:
                pass
            
            # Step 2: Maximize d_min, subject to d_ij <= d_max_current * 1.05
            # This increases the floor while keeping the ceiling roughly fixed.
            
            U = d_max * 1.05
            
            def obj_scp2(x):
                p = x.reshape(N, 2)
                dist = pdist(p)
                d_min = np.min(dist)
                viol = np.maximum(0, dist - U)
                penalty = np.sum(viol**2)
                # Maximize d_min -> Minimize -d_min
                # But -d_min is non-smooth.
                # Use: Minimize -log(d_min) + penalty
                if d_min < 1e-10:
                    return 1e10
                return -np.log(d_min) + 10.0 * penalty
            
            def grad_scp2(x):
                p = x.reshape(N, 2)
                dist = pdist(p)
                d_min = np.min(dist)
                viol = np.maximum(0, dist - U)
                penalty = np.sum(viol**2)
                if d_min < 1e-10:
                    return 1e10, np.zeros_like(x)
                obj_val = -np.log(d_min) + 10.0 * penalty
                
                grad = np.zeros_like(x)
                
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dist_mat = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist_mat, 0)
                
                d_min_val = np.min(dist_mat)
                mask_min = dist_mat < d_min_val + 1e-6
                
                for i in range(N):
                    for j in range(i+1, N):
                        if mask_min[i, j]:
                            d_ij = dist_mat[i, j]
                            if d_ij < 1e-10:
                                continue
                            diff_ij = p[i] - p[j]
                            grad_ij = diff_ij / d_ij
                            coeff = -1.0 / d_min_val
                            grad[2*i] += coeff * grad_ij[0]
                            grad[2*i+1] += coeff * grad_ij[1]
                            grad[2*j] -= coeff * grad_ij[0]
                            grad[2*j+1] -= coeff * grad_ij[1]
                        
                        if dist_mat[i, j] > U:
                            d_ij = dist_mat[i, j]
                            diff_ij = p[i] - p[j]
                            if d_ij < 1e-10:
                                continue
                            grad_ij = diff_ij / d_ij
                            coeff = 2.0 * (dist_mat[i, j] - U) * 10.0
                            grad[2*i] += coeff * grad_ij[0]
                            grad[2*i+1] += coeff * grad_ij[1]
                            grad[2*j] -= coeff * grad_ij[0]
                            grad[2*j+1] -= coeff * grad_ij[1]
                
                return obj_val, grad
            
            try:
                result = minimize(obj_scp2, x0, jac=grad_scp2, method='L-BFGS-B',
                                  options={'maxiter': 50, 'maxls': 20})
                if result.success or result.fun < 1e10:
                    new_pts = result.x.reshape(N, 2)
                    new_ratio = get_ratio(new_pts)
                    if new_ratio > best_ratio * 0.9:
                        pts = new_pts
                        if new_ratio > best_ratio:
                            best_ratio = new_ratio
                            best_pts = new_pts.copy()
            except Exception:
                pass
        
        return best_pts, best_ratio

    def coordinate_descent(pts, time_budget=100.0):
        """
        Fast local refinement by moving one point at a time.
        """
        pts = pts.copy()
        t_start = time.time()
        best_pts = pts.copy()
        best_ratio = get_ratio(pts)
        
        step_size = 0.1
        while time.time() - t_start < time_budget:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            improved = False
            for i in range(N):
                if time.time() - t_start > time_budget:
                    break
                
                old_pos = pts[i].copy()
                # Try moving in 4 directions
                for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
                               (step_size/np.sqrt(2), step_size/np.sqrt(2)),
                               (-step_size/np.sqrt(2), step_size/np.sqrt(2)),
                               (step_size/np.sqrt(2), -step_size/np.sqrt(2)),
                               (-step_size/np.sqrt(2), -step_size/np.sqrt(2))]:
                    pts[i] = old_pos + np.array([dx, dy])
                    ratio = get_ratio(pts)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_pts = pts.copy()
                        improved = True
                    else:
                        pts[i] = old_pos
                
                # If improved, try smaller steps
                if improved:
                    step_size = max(0.001, step_size * 0.7)
                else:
                    step_size = min(0.5, step_size * 1.2)
            
            # If no improvement with current step size, reduce it
            if not improved:
                step_size *= 0.8
                if step_size < 1e-4:
                    break
        
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

    # Phase 1: SCP from best candidate
    time_elapsed = time.time() - start_time
    scp_budget = min(300.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if scp_budget > 10:
        scp_pts, scp_ratio = sequential_convex_programming(best_pts, time_budget=scp_budget)
        if scp_ratio > best_ratio:
            best_pts = scp_pts
            best_ratio = scp_ratio

    # Phase 2: Coordinate Descent Polish
    time_elapsed = time.time() - start_time
    cd_budget = min(150.0, TIME_LIMIT - time_elapsed - SAFETY_MARGIN)
    if cd_budget > 5:
        cd_pts, cd_ratio = coordinate_descent(best_pts, time_budget=cd_budget)
        if cd_ratio > best_ratio:
            best_pts = cd_pts
            best_ratio = cd_ratio

    # Phase 3: Try SCP from a few other candidates if time permits
    time_elapsed = time.time() - start_time
    if TIME_LIMIT - time_elapsed - SAFETY_MARGIN > 50:
        for c in candidates[1:3]: # Try 2nd and 3rd best candidates
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            remaining = TIME_LIMIT - time.time() - SAFETY_MARGIN
            if remaining < 20:
                break
            c_pts, c_ratio = sequential_convex_programming(c, time_budget=min(remaining - 5, 100.0))
            if c_ratio > best_ratio:
                best_pts = c_pts
                best_ratio = c_ratio
                # Polish
                if TIME_LIMIT - time.time() - SAFETY_MARGIN > 10:
                    cd_pts2, cd_ratio2 = coordinate_descent(best_pts, time_budget=min(20.0, TIME_LIMIT - time.time() - SAFETY_MARGIN))
                    if cd_ratio2 > best_ratio:
                        best_pts = cd_pts2
                        best_ratio = cd_ratio2

    # Ensure finite and correct shape
    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)) or final_pts.shape != (16, 2):
        final_pts = np.random.randn(16, 2)

    return final_pts
