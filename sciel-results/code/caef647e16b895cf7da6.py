"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_radii_from_centers(centers, margin=0.0):
        """
        Given circle centers, compute maximum non-overlapping radii using
        all pairwise distances and boundary distances.
        Returns array of radii. Vectorized for speed.
        """
        centers = np.asarray(centers)
        n = len(centers)
        
        # Compute boundary distances vectorized
        d_left = centers[:, 0]
        d_right = 1.0 - centers[:, 0]
        d_bottom = centers[:, 1]
        d_top = 1.0 - centers[:, 1]
        radii = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        
        # Enforce non-overlap for all pairs (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist, axis=1)  # min distance from i to any j
        pair_constraint = min_pair_dist / 2.0
        radii = np.minimum(radii, pair_constraint)
        
        # Apply margin if needed
        if margin > 0:
            radii = np.maximum(0.0, radii - margin)
        
        return radii

    def objective_penalty(x, mu):
        """
        Objective: Maximize sum(r) - mu * sum(violations)
        x is flattened [x1, y1, r1, ..., xN, yN, rN]
        """
        x = x.reshape(N, 3)
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        
        # Objective: -sum(r)
        obj = -np.sum(rs)
        
        # Penalty for boundary violations
        # r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
        # Violation = max(0, r_i - min(boundaries))
        bounds = np.minimum(np.minimum(xs, 1.0 - xs), np.minimum(ys, 1.0 - ys))
        bound_viol = np.maximum(0.0, rs - bounds)
        obj += mu * np.sum(bound_viol**2)
        
        # Penalty for overlap violations
        # r_i + r_j <= dist(i,j)
        # Violation = max(0, r_i + r_j - dist(i,j))
        diff = x[:, np.newaxis, :2] - x[np.newaxis, :, :2]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # We only need upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        r_sum = rs[i_idx] + rs[j_idx]
        d_ij = dist[i_idx, j_idx]
        overlap_viol = np.maximum(0.0, r_sum - d_ij)
        obj += mu * np.sum(overlap_viol**2)
        
        return obj

    def gradient_penalty(x, mu):
        """
        Gradient of the penalty objective.
        """
        x = x.reshape(N, 3)
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        
        grad = np.zeros_like(x)
        
        # d(-sum(r))/dr = -1
        grad[:, 2] = -1.0
        
        # Gradient of boundary penalty
        # P_b = mu * sum( max(0, r_i - b_i)^2 )
        # dP_b/dr_i = 2*mu*max(0, r_i - b_i)
        # dP_b/dx_i = -2*mu*max(0, r_i - b_i) * sign(x_i - 0.5) (if x is limiting)
        # Actually, b_i = min(xs, 1-xs, ys, 1-ys). 
        # Let's compute gradients w.r.t x, y properly.
        # If r_i > b_i, we are violating.
        # b_i is determined by the closest boundary.
        # d(b_i)/d(x_i) is 1 if x_i < 0.5 and x_i is the min, else -1 if x_i > 0.5 and x_i is min, else 0.
        
        b_x = np.minimum(xs, 1.0 - xs)
        b_y = np.minimum(ys, 1.0 - ys)
        b = np.minimum(b_x, b_y)
        
        viol_b = np.maximum(0.0, rs - b)
        if np.any(viol_b > 0):
            # dP/dr = 2*mu*viol_b
            grad[:, 2] += 2.0 * mu * viol_b
            
            # dP/dx = -2*mu*viol_b * d(b)/dx
            # d(b)/dx is non-zero only if x-boundary is the limiting one.
            # If b == b_x, then d(b)/dx = 1 if x < 0.5 else -1.
            # If b == b_y, d(b)/dx = 0.
            limit_x = (b == b_x) & (b_y > b_x) # x is strictly limiting
            limit_x_tie = (b == b_x) & (b_y == b_x) # tie
            # In tie, gradient is subgradient, we can pick one or average. Let's pick based on position.
            # If x < 0.5, d(b)/dx = 1. If x > 0.5, d(b)/dx = -1.
            sign_x = np.where(xs < 0.5, 1.0, -1.0)
            
            # If x is limiting or tied, we apply the gradient.
            # For simplicity and robustness, we apply gradient if x is the argmin.
            # If tie, we apply 0.5 * (grad_x + grad_y)? No, just pick x if x<0.5.
            
            mask_x_limit = (b_x <= b_y) 
            # If x is limiting (or tied), d(b)/dx = sign_x
            # If y is limiting (strictly), d(b)/dx = 0
            
            grad[:, 0] -= 2.0 * mu * viol_b * np.where(mask_x_limit, sign_x, 0.0)
            
            sign_y = np.where(ys < 0.5, 1.0, -1.0)
            mask_y_limit = (b_y < b_x)
            grad[:, 1] -= 2.0 * mu * viol_b * np.where(mask_y_limit, sign_y, 0.0)
            
            # Handle ties: if b_x == b_y, both are limiting.
            # In that case, d(b)/dx is not unique. We can use 0.5 for both?
            # Or just ignore ties as they are measure zero.
            # Let's add 0.5 contribution to both if tie.
            tie_mask = (b_x == b_y) & (viol_b > 0)
            if np.any(tie_mask):
                # In tie, the constraint is r <= min(x, 1-x, y, 1-y).
                # The subgradient of min is convex combination.
                # We'll just add 0.5 * sign to both to be safe.
                grad[:, 0] -= 2.0 * mu * viol_b * 0.5 * sign_x * tie_mask
                grad[:, 1] -= 2.0 * mu * viol_b * 0.5 * sign_y * tie_mask
                # And subtract the previous full contribution if we double counted?
                # My mask_x_limit includes ties. So I added full sign_x.
                # I want 0.5 * sign_x. So I should subtract 0.5 * sign_x.
                grad[:, 0] += 2.0 * mu * viol_b * 0.5 * sign_x * tie_mask
                grad[:, 1] += 2.0 * mu * viol_b * 0.5 * sign_y * tie_mask

        # Gradient of overlap penalty
        # P_o = mu * sum_{i<j} max(0, r_i + r_j - d_ij)^2
        i_idx, j_idx = np.triu_indices(N, k=1)
        r_sum = rs[i_idx] + rs[j_idx]
        
        diff = x[i_idx, :2] - x[j_idx, :2]
        d_ij = np.sqrt(np.sum(diff**2, axis=1))
        
        viol_o = np.maximum(0.0, r_sum - d_ij)
        
        if np.any(viol_o > 0):
            # dP/dr_i = 2*mu*viol_o for all i involved
            # For each pair (i,j), add 2*mu*viol_o to grad[i,2] and grad[j,2]
            np.add.at(grad[:, 2], i_idx, 2.0 * mu * viol_o)
            np.add.at(grad[:, 2], j_idx, 2.0 * mu * viol_o)
            
            # dP/dx_i = -2*mu*viol_o * (x_i - x_j) / d_ij
            # dP/dy_i = -2*mu*viol_o * (y_i - y_j) / d_ij
            
            # Note: d_ij is in denominator. If d_ij is 0, we have a problem.
            # But if d_ij is 0, and r_i+r_j > 0, viol is large.
            # Gradient is undefined at 0. We can skip or use small epsilon.
            valid_d = d_ij > 1e-8
            if np.any(valid_d):
                inv_d = np.where(valid_d, 1.0 / d_ij, 0.0)
                grad_x_i = -2.0 * mu * viol_o * inv_d * diff[:, 0]
                grad_y_i = -2.0 * mu * viol_o * inv_d * diff[:, 1]
                
                np.add.at(grad[:, 0], i_idx, grad_x_i)
                np.add.at(grad[:, 1], i_idx, grad_y_i)
                
                # For j, it's the negative
                np.add.at(grad[:, 0], j_idx, -grad_x_i)
                np.add.at(grad[:, 1], j_idx, -grad_y_i)
        
        return grad.flatten()

    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers

    def local_search(x, max_iters=2000, step_size=0.02, seed=0):
        """
        Discrete best-fit local search on full state (x, y, r).
        """
        rng = np.random.RandomState(seed)
        current_x = x.copy()
        current_score = np.sum(current_x[:, 2])
        
        best_x = current_x.copy()
        best_score = current_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random circle
            i = rng.randint(N)
            
            # Try to move center or change radius
            # We'll perturb x, y, r slightly
            dx = rng.uniform(-step_size, step_size)
            dy = rng.uniform(-step_size, step_size)
            dr = rng.uniform(-step_size, step_size)
            
            new_x = current_x.copy()
            new_x[i, 0] = np.clip(new_x[i, 0] + dx, 0.0, 1.0)
            new_x[i, 1] = np.clip(new_x[i, 1] + dy, 0.0, 1.0)
            new_x[i, 2] = np.clip(new_x[i, 2] + dr, 0.0, 0.5)
            
            # Check validity
            if is_valid(new_x, tol=1e-8):
                new_score = np.sum(new_x[:, 2])
                if new_score > current_score + 1e-10:
                    current_x = new_x
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()
            else:
                # If invalid, try to fix by reducing radius?
                # Or just reject.
                pass
        
        return best_x

    def optimize_packing(x_init, max_iters=100):
        """
        Run L-BFGS-B on the penalty objective.
        """
        x0 = x_init.flatten()
        
        # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 0.5))
            
        # Start with moderate penalty
        mu = 10.0
        
        # We will increase mu over time to enforce constraints
        # But L-BFGS-B might struggle if mu is too high initially.
        
        best_x = x_init
        best_score = np.sum(x_init[:, 2])
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(
                    lambda z: objective_penalty(z, mu),
                    lambda z: gradient_penalty(z, mu),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x_opt = result.x.reshape(N, 3)
                
                # Check validity
                if is_valid(x_opt, tol=1e-6):
                    score = np.sum(x_opt[:, 2])
                    if score > best_score:
                        best_score = score
                        best_x = x_opt.copy()
                        x0 = x_opt.flatten()
                        
                    # If constraints are satisfied, we can stop or increase mu slightly?
                    # Actually, if valid, the penalty term is 0.
                    # We can try to increase mu to see if we can push radii further?
                    # No, if valid, the objective is just -sum(r).
                    # The gradient will push r up until it hits a constraint.
                    # So we should keep mu high enough to prevent violation.
                    
                    # Let's increase mu if we are close to constraints?
                    # For now, just keep mu constant or increase slowly.
                    mu = min(mu * 1.5, 1000.0)
                else:
                    # If invalid, increase penalty
                    mu = min(mu * 2.0, 1000.0)
                    # Use the best valid point or the current point?
                    # If invalid, we can't use it as best.
                    # But we can use it as start for next iteration with higher mu.
                    x0 = x_opt.flatten()
                    
            except Exception:
                break
                
        return best_x

    # Strategy: Multiple initializations + Penalty Optimization + Local Search
    best_result = None
    best_score = -np.inf
    
    inits = []
    inits.append(initial_grid_centers())
    for seed in [1, 7, 13, 23, 37, 53, 71, 89, 101, 131]:
        inits.append(random_centers(seed))
    
    # Create initial full states (x, y, r)
    # r initialized to min distance to boundary / 2 (conservative)
    full_inits = []
    for centers in inits:
        rs = compute_radii_from_centers(centers, margin=0.01)
        full_state = np.column_stack([centers, rs])
        full_inits.append(full_state)
    
    for idx, init_state in enumerate(full_inits):
        if time.time() > deadline - 15:
            break
        
        # Step 1: Penalty Optimization
        opt_state = optimize_packing(init_state, max_iters=50)
        
        # Step 2: Local Search from optimized state
        ls_state = local_search(opt_state, max_iters=1000, step_size=0.01, seed=idx)
        
        # Step 3: Another round of penalty optimization from local search result
        opt_state2 = optimize_packing(ls_state, max_iters=30)
        
        # Check best of the three
        candidates = [init_state, opt_state, ls_state, opt_state2]
        for cand in candidates:
            if is_valid(cand, tol=1e-6):
                score = np.sum(cand[:, 2])
                if score > best_score:
                    best_score = score
                    best_result = cand.copy()
    
    if best_result is None:
        # Fallback
        centers = initial_grid_centers()
        radii = compute_radii_from_centers(centers, margin=0.001)
        best_result = np.column_stack([centers, radii])
        
    return best_result

# EVOLVE-BLOCK-END
