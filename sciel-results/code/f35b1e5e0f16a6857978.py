"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We use 145s to be safe.
    deadline = start_time + 145.0
    
    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        # Check bounds
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Compute pairwise distances
        n = len(centers)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def compute_radius_and_bottleneck(centers):
        """Compute radius and identify the bottleneck (closest pair or boundary violation)."""
        # Check bounds
        wall_dists = 0.5 - np.abs(centers)
        min_wall_idx = np.argmin(wall_dists)
        min_wall = wall_dists.flat[min_wall_idx]
        
        # Compute pairwise distances using vectorized approach
        n = len(centers)
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        
        min_pair_idx = np.argmin(dist_matrix)
        min_pair = dist_matrix.flat[min_pair_idx]
        
        # Determine which is the bottleneck
        if min_wall < min_pair / 2.0:
            # Wall is the bottleneck
            bottleneck_type = 'wall'
            bottleneck_idx = min_wall_idx // 2  # index of the point
            bottleneck_value = min_wall
        else:
            # Pair is the bottleneck
            i, j = np.unravel_index(min_pair_idx, (n, n))
            bottleneck_type = 'pair'
            bottleneck_idx = (i, j)
            bottleneck_value = min_pair / 2.0
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r), bottleneck_type, bottleneck_idx, bottleneck_value

    def hexagonal_initialization(n_rows, n_cols, d_scale=1.0):
        """Create hexagonal packing layout."""
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
        
        return pts

    def square_grid_init(m):
        """Create square grid initialization."""
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    def force_directed_relaxation(centers, time_limit=10.0):
        """
        Physics-based relaxation using repulsive forces for close pairs 
        and attractive forces for distant pairs to maintain structure.
        Uses a simple Euler integration with adaptive step size.
        """
        centers = centers.copy()
        start = time.time()
        
        # Parameters
        # We want to maximize min dist. 
        # Force is proportional to (target_dist - current_dist) if current < target, 
        # else weak attraction to prevent collapse.
        
        # Initial target distance estimate based on area
        area = 1.0
        r_est = np.sqrt(area / N) / 2.0
        target_dist = 2.0 * r_est * 1.05 # Slightly larger to push apart
        
        # We will iteratively adjust target_dist or just use forces.
        # A common method: 
        # F_ij = k * (d_target - d_ij) * (d_ij_vec / d_ij^2) if d_ij < d_target
        #        - k_attr * (d_ij - d_target) * (d_ij_vec / d_ij^2) if d_ij > d_target (weak)
        
        # Better: Just use repulsive forces scaled by how much they violate a "soft" constraint.
        # We want to increase the minimum distance.
        
        # Let's use a potential energy approach:
        # E = sum_{i<j} max(0, target_dist - d_ij)^2 + sum_i max(0, 0.5 - min(|x_i|,|y_i|))^2
        # Minimize E.
        
        # Actually, let's stick to direct force summation for speed.
        
        alpha = 0.01 # Learning rate / damping
        beta = 0.001 # Damping factor for velocity (if we used it, but we do direct pos update)
        
        for step in range(10000):
            if time.time() > time_limit + start:
                break
            
            forces = np.zeros_like(centers)
            
            # Compute pairwise forces
            # d_ij = c_i - c_j
            # F_i += k * (d_target - d_ij) * (d_ij_vec / d_ij^2) if d_ij < d_target
            
            # Vectorized computation is tricky for variable N if we want to avoid full matrix every step,
            # but N=200 is small enough for O(N^2) vectorization.
            
            # diff[i,j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Mask diagonal
            np.fill_diagonal(dists, np.inf)
            
            # We want to push apart if dist < target_dist
            # Let's define a "violation" v = max(0, target_dist - dist)
            # Force magnitude = v
            # Direction = diff / dist
            
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-10, dists, 1e-10)
            
            # Only apply force if dist < target_dist
            mask = dists < target_dist
            v = np.where(mask, target_dist - dists, 0.0)
            
            # Force vector for i due to j is proportional to v * (diff_ij / dist_ij)
            # F_ij = v * diff / dist
            # Since diff = c_i - c_j, this pushes i away from j.
            
            force_matrix = v[:, :, np.newaxis] * diff / dists_safe[:, :, np.newaxis]
            
            # Sum forces for each i
            forces += np.sum(force_matrix, axis=1)
            
            # Boundary forces: Push back if outside
            # If x > 0.5, force_x is negative. If x < -0.5, force_x is positive.
            # We can model this as: if dist_to_wall < 0, push in.
            # Actually, centers must be inside. 
            # Let's add a strong repulsive force from walls.
            # Wall distance: 0.5 - |x|
            # If 0.5 - |x| < 0, we are outside. 
            # We want to push towards center.
            
            # Let's use a simple penalty gradient.
            # For x: if x > 0.5, F_x = -k_wall * (x - 0.5). If x < -0.5, F_x = k_wall * (-0.5 - x).
            # Inside, no force from walls unless we want to keep them away from walls?
            # No, we just need them inside. The pairwise forces will handle spacing.
            # However, if a circle is pushed against the wall, the pairwise force might push it further out if not careful.
            # So we need a hard constraint or strong wall force.
            
            k_wall = 10.0
            # x boundary
            over_x_pos = np.maximum(0, centers[:, 0] - 0.5)
            under_x_neg = np.maximum(0, -0.5 - centers[:, 0])
            forces[:, 0] -= k_wall * over_x_pos
            forces[:, 0] += k_wall * under_x_neg
            
            over_y_pos = np.maximum(0, centers[:, 1] - 0.5)
            under_y_neg = np.maximum(0, -0.5 - centers[:, 1])
            forces[:, 1] -= k_wall * over_y_pos
            forces[:, 1] += k_wall * under_y_neg
            
            # Update positions
            # Adaptive step size based on force magnitude
            max_force = np.max(np.linalg.norm(forces, axis=1))
            if max_force > 1e-6:
                step_size = alpha / max_force
                # Limit step size to prevent instability
                step_size = min(step_size, 0.01)
                centers += forces * step_size
            
            # Clamp to bounds (safety net)
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
            # Check convergence
            if max_force < 1e-4:
                break
            
            # Gradually increase target_dist to push for larger radius?
            # Or just let it settle.
            # Let's slowly increase target_dist to encourage expansion.
            # target_dist *= 1.0001 # Very slow
            
        return centers

    def soft_min_objective(centers, k=100.0):
        """
        Objective function for optimization: -soft_min(distances, wall_dists)
        We want to maximize the minimum distance.
        soft_min(x) = -1/k * log(sum(exp(-k*x_i)))
        """
        n = len(centers)
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        flat_wall = wall_dists.flatten()
        
        # Pairwise distances
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        flat_pair = dist_matrix.flatten()
        
        # Combine all constraints
        # Note: Pairwise distance constraint is d_ij >= 2r => r <= d_ij/2
        # Wall constraint is d_wall >= r => r <= d_wall
        # So we want to maximize min( min(d_wall), min(d_pair/2) )
        
        # Let's work with values v_i such that r <= v_i.
        # v_wall = d_wall
        # v_pair = d_pair / 2
        
        vals = np.concatenate([flat_wall, flat_pair / 2.0])
        
        # Soft min
        # To avoid overflow/underflow in exp, shift by min_val
        min_val = np.min(vals)
        # If min_val is very negative, exp(-k*val) might overflow? 
        # vals are positive (distances).
        # We want soft_min(vals).
        # soft_min = -1/k * log(sum(exp(-k*vals)))
        # Let's use stable log-sum-exp.
        # log_sum_exp(-k*vals) = max(-k*vals) + log(sum(exp(-k*vals - max(-k*vals))))
        
        neg_k_vals = -k * vals
        max_neg_k_val = np.max(neg_k_vals)
        # If max_neg_k_val is -inf (shouldn't happen), handle it.
        
        log_sum_exp = max_neg_k_val + np.log(np.sum(np.exp(neg_k_vals - max_neg_k_val)))
        soft_min_val = -log_sum_exp / k
        
        # We want to MAXIMIZE soft_min_val.
        # So minimize -soft_min_val.
        return -soft_min_val

    def soft_min_grad(centers, k=100.0):
        """
        Gradient of -soft_min_objective.
        """
        n = len(centers)
        wall_dists = 0.5 - np.abs(centers)
        flat_wall = wall_dists.flatten()
        
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        flat_pair = dist_matrix.flatten()
        
        vals = np.concatenate([flat_wall, flat_pair / 2.0])
        
        neg_k_vals = -k * vals
        max_neg_k_val = np.max(neg_k_vals)
        exp_vals = np.exp(neg_k_vals - max_neg_k_val)
        sum_exp = np.sum(exp_vals)
        probs = exp_vals / sum_exp # These are the weights for the soft min
        
        # Gradient of soft_min w.r.t. vals:
        # d(soft_min)/d(v_i) = -1/k * d/d(v_i) [ -1/k log(sum(exp(-k v_j))) ]
        # = 1/k^2 * [ sum_j (-k exp(-k v_j)) / sum_j exp(-k v_j) ] * (-1) ? 
        # Let S = sum exp(-k v_j).
        # L = -1/k log(S).
        # dL/dv_i = -1/k * (1/S) * dS/dv_i
        # dS/dv_i = -k exp(-k v_i)
        # dL/dv_i = -1/k * (1/S) * (-k exp(-k v_i)) = exp(-k v_i) / S = probs[i]
        
        # So gradient of soft_min w.r.t. vals is probs.
        # We are minimizing -soft_min, so gradient is -probs.
        
        grad_vals = -probs
        
        # Now map grad_vals back to centers.
        grad_centers = np.zeros_like(centers)
        
        # 1. Wall part
        # v_wall_flat = 0.5 - |c|
        # d(v_wall)/dc = -sign(c)
        # d(-soft_min)/dc = d(-soft_min)/dv * dv/dc
        # = (-probs_wall) * (-sign(c)) = probs_wall * sign(c)
        
        grad_wall_flat = grad_vals[:len(flat_wall)]
        grad_wall = grad_wall_flat.reshape(n, 2)
        
        sign_c = np.sign(centers)
        # If c is 0, sign is 0, but subgradient can be anything. 0 is fine.
        grad_centers += grad_wall * sign_c
        
        # 2. Pair part
        # v_pair_flat = d_ij / 2
        # d(v_pair)/dc_i = 0.5 * (c_i - c_j) / d_ij
        # d(v_pair)/dc_j = 0.5 * (c_j - c_i) / d_ij
        
        grad_pair_flat = grad_vals[len(flat_wall):]
        # Reshape to (n, n)
        # Note: cdist returns upper triangle? No, full matrix.
        # We flattened row-major.
        # Indices in flat_pair correspond to (i, j) in dist_matrix.
        
        # We need to accumulate gradients for each i and j.
        # For each pair (i, j) with i != j:
        # term = grad_pair_flat[i*n + j]
        # d(v)/dc_i = 0.5 * diff_ij / dist_ij
        # d(v)/dc_j = -0.5 * diff_ij / dist_ij
        
        # We can vectorize this.
        # dist_matrix is (n, n)
        # diff_matrix is (n, n, 2)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0) # Avoid div by 0, though grad is 0 on diagonal
        
        # grad_pair_matrix is (n, n)
        grad_pair_matrix = grad_pair_flat.reshape(n, n)
        
        # factor = 0.5 * grad_pair_matrix / dists
        factor = 0.5 * grad_pair_matrix / dists
        
        # grad_centers[i] += sum_j factor[i,j] * diff[i,j]
        # grad_centers[j] += sum_i factor[i,j] * (-diff[i,j]) = - sum_i factor[i,j] * diff[i,j]
        # Note factor[j,i] is related to factor[i,j]? 
        # grad_pair_matrix is symmetric? 
        # v_ij = d_ij/2. Symmetric.
        # probs are symmetric? exp(-k v_ij) is symmetric. Yes.
        # So factor is symmetric.
        
        # Contribution to i from all j:
        contrib_i = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Contribution to j from all i:
        # For a fixed j, sum over i of factor[i,j] * (-diff[i,j])
        # diff[i,j] = c_i - c_j
        # -diff[i,j] = c_j - c_i
        # factor[i,j] = factor[j,i]
        # So sum_i factor[j,i] * (c_j - c_i)
        # This is exactly the same calculation as contrib_i but for index j.
        # So grad_centers += contrib_i
        
        grad_centers += contrib_i
        
        return grad_centers

    def run_optimization(centers, time_limit=20.0):
        """
        Run scipy optimization using the soft-min surrogate.
        """
        start = time.time()
        
        # Flatten centers for scipy
        x0 = centers.flatten()
        
        def obj(x):
            c = x.reshape(-1, 2)
            return soft_min_objective(c, k=50.0)
            
        def grad(x):
            c = x.reshape(-1, 2)
            return soft_min_grad(c, k=50.0).flatten()
            
        # Use L-BFGS-B as it handles bounds well and is efficient for this size
        # Bounds: -0.5 to 0.5
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                obj, 
                x0, 
                method='L-BFGS-B', 
                jac=grad, 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            # Check time
            if time.time() > start + time_limit:
                # If it took too long, we might want to return best found so far?
                # minimize doesn't easily give intermediate results.
                # We'll just use the result.
                pass
                
            new_centers = result.x.reshape(-1, 2)
            return new_centers
            
        except Exception:
            return centers

    # Generate initial configurations
    best_centers = None
    best_radius = 0.0
    
    # 1. Hexagonal Initialization
    for n_rows in range(13, 17):
        if time.time() > deadline - 60:
            break
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.95, 1.0, 1.05]:
            pts = hexagonal_initialization(n_rows, n_cols, d_scale)
            if pts is not None:
                r, _, _, _ = compute_radius_and_bottleneck(pts)
                if r > best_radius:
                    best_radius = r
                    best_centers = pts.copy()

    # 2. Square Grid
    m = int(np.ceil(np.sqrt(N)))
    grid = square_grid_init(m)
    r, _, _, _ = compute_radius_and_bottleneck(grid)
    if r > best_radius:
        best_radius = r
        best_centers = grid.copy()

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_radius = compute_radius(best_centers)

    # Strategy:
    # 1. Force Directed Relaxation to get a good local minimum quickly.
    # 2. Soft-Min Optimization to polish it.
    # 3. Repeat with restarts if time permits.
    
    # Phase 1: Force Directed Relaxation
    if time.time() < deadline - 40:
        relaxed = force_directed_relaxation(best_centers, time_limit=10.0)
        r_relaxed = compute_radius(relaxed)
        if r_relaxed > best_radius:
            best_radius = r_relaxed
            best_centers = relaxed.copy()
            
    # Phase 2: Soft-Min Optimization (Polish)
    if time.time() < deadline - 20:
        optimized = run_optimization(best_centers, time_limit=15.0)
        r_opt = compute_radius(optimized)
        if r_opt > best_radius:
            best_radius = r_opt
            best_centers = optimized.copy()
            
    # Phase 3: Restarted Search
    # If we have time, try a different initial configuration or random perturbation of the best one.
    
    if time.time() < deadline - 30:
        # Perturb best centers
        perturbed = best_centers + np.random.randn(N, 2) * 0.01
        perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
        perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
        
        # Quick relaxation
        relaxed2 = force_directed_relaxation(perturbed, time_limit=5.0)
        r_relaxed2 = compute_radius(relaxed2)
        
        if r_relaxed2 > best_radius:
            best_radius = r_relaxed2
            best_centers = relaxed2.copy()
            
        # Polish
        if time.time() < deadline - 10:
            optimized2 = run_optimization(best_centers, time_limit=10.0)
            r_opt2 = compute_radius(optimized2)
            if r_opt2 > best_radius:
                best_radius = r_opt2
                best_centers = optimized2.copy()
                
    # Final Polish with tighter tolerance if time remains
    if time.time() < deadline - 5:
        # One more pass with higher k in soft-min?
        # Or just standard L-BFGS again
        final_opt = run_optimization(best_centers, time_limit=5.0)
        r_final = compute_radius(final_opt)
        if r_final > best_radius:
            best_radius = r_final
            best_centers = final_opt.copy()

    return best_centers
# EVOLVE-BLOCK-END
