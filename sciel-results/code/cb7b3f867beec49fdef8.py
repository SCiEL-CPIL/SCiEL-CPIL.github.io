"""Combined circle packing: Multi-start smooth optimization + high-precision SLSQP polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_sum = 0.0
    best_result = None
    
    def check_validity(x, y, r, tol=1e-7):
        """Check if packing is valid (all constraints satisfied)."""
        # Boundary check
        if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        # Collision check
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def smooth_objective(z, beta=50.0):
        """
        Smooth penalty-based objective: minimize -sum(r) + penalty for violations.
        Uses softplus approximation for max(0, violation).
        """
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]
        
        # Clip radii to a small positive value for stability in softplus
        r_safe = np.maximum(r, 1e-5)
        
        sum_r = np.sum(r_safe)
        
        # Boundary violations: c_i >= 0 required. Violation is max(0, -c_i).
        # c1: x - r >= 0 -> viol = max(0, r - x)
        # c2: 1 - x - r >= 0 -> viol = max(0, x + r - 1)
        # c3: y - r >= 0 -> viol = max(0, r - y)
        # c4: 1 - y - r >= 0 -> viol = max(0, y + r - 1)
        
        v_bound_raw = np.concatenate([
            r_safe - x,
            x + r_safe - 1.0,
            r_safe - y,
            y + r_safe - 1.0
        ])
        
        # Overlap violations: dist^2 - (r_i+r_j)^2 >= 0
        # We use squared distance to avoid sqrt in the main loop for speed, 
        # but softplus needs the actual constraint value.
        # Let's use the standard form: d - (ri+rj) >= 0
        # For smoothness and speed, we can approximate the penalty on squared terms 
        # or just compute the linear violation. Linear violation is smoother.
        
        x_diff = x[:, None] - x[None, :]
        y_diff = y[:, None] - y[None, :]
        dist_sq = x_diff**2 + y_diff**2
        sum_r_mat = r_safe[:, None] + r_safe[None, :]
        
        # dist = sqrt(dist_sq)
        # To avoid sqrt cost in every eval, we can use a smooth approximation 
        # or just compute it. N=26 is small enough that sqrt is fine.
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        v_coll_raw = dist - sum_r_mat
        # Mask diagonal
        v_coll_raw = np.where(np.triu(np.ones((N, N), dtype=bool), k=1), v_coll_raw, 0)
        v_coll_raw = v_coll_raw.flatten()
        
        # Combine all violations
        all_viol_raw = np.concatenate([v_bound_raw, v_coll_raw])
        
        # Softplus penalty: sum(softplus(-beta * viol)) / beta
        # softplus(z) = log(1 + exp(z))
        # We want penalty for viol < 0 (i.e., constraint satisfied is viol >= 0? No, 
        # my v_bound_raw is defined as (r-x) which is positive when violated.
        # So violation amount is max(0, v_bound_raw).
        # Let's stick to: penalty = sum(softplus(beta * v_raw)) / beta
        # where v_raw is positive when violated.
        
        z_pen = beta * all_viol_raw
        # Stable softplus: where(z > 0, z + log1p(exp(-z)), log1p(exp(z)))
        softplus_vals = np.where(z_pen > 0, z_pen + np.log1p(np.exp(-z_pen)), np.log1p(np.exp(z_pen)))
        
        penalty = np.sum(softplus_vals) / beta
        
        return -sum_r + penalty

    def boundary_constraints(z):
        """Hard constraints for SLSQP polish."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]
        return np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])

    def overlap_constraints(z):
        """Hard constraints for SLSQP polish."""
        x = z[:N]
        y = z[N:2*N]
        r = z[2*N:]
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                cons.append(d - r[i] - r[j])
        return np.array(cons)

    def all_constraints(z):
        return np.concatenate([boundary_constraints(z), overlap_constraints(z)])

    def objective_slsqp(z):
        """Simple objective for SLSQP polish."""
        r = z[2*N:]
        return -np.sum(r)

    def gradient_slsqp(z):
        g = np.zeros(3*N)
        g[2*N:] = -1.0
        return g

    def polish_slsqp(x0, y0, r0, max_iter=100, ftol=1e-12):
        """High-precision SLSQP polish."""
        z0 = np.concatenate([x0, y0, r0])
        
        # Tight bounds around current solution to help SLSQP
        bounds = []
        for i in range(N):
            bounds.append((max(0.0, x0[i] - 0.05), min(1.0, x0[i] + 0.05)))
            bounds.append((max(0.0, y0[i] - 0.05), min(1.0, y0[i] + 0.05)))
        for i in range(N):
            bounds.append((max(1e-5, r0[i] * 0.95), min(0.5, r0[i] * 1.05)))

        try:
            res = minimize(objective_slsqp, z0, method='SLSQP', jac=gradient_slsqp,
                          constraints={'type': 'ineq', 'fun': all_constraints},
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': ftol, 'disp': False})
            
            if res.success:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                # Post-process: ensure strict validity
                if check_validity(x_opt, y_opt, r_opt, tol=1e-8):
                    return x_opt, y_opt, r_opt
                else:
                    # Try to repair by shrinking radii slightly
                    for scale in [0.9999, 0.9995, 0.999]:
                        r_adjusted = r_opt * scale
                        if check_validity(x_opt, y_opt, r_adjusted, tol=1e-8):
                            return x_opt, y_opt, r_adjusted
                    return None
            else:
                return None
        except Exception:
            return None

    def smooth_optimize(x0, y0, r0, max_iter=500):
        """Run optimization using L-BFGS-B on the smooth penalty objective."""
        z0 = np.concatenate([x0, y0, r0])
        
        bounds = [(0, 1)] * (2*N) + [(1e-5, 0.5)] * N
        
        try:
            res = minimize(smooth_objective, z0, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            
            # The smooth optimizer might leave small violations.
            # We need to project back to feasibility or check validity.
            # For now, return the result. The polish step will handle strictness.
            # But we must ensure it's not wildly invalid.
            if check_validity(x_opt, y_opt, r_opt, tol=1e-5):
                return x_opt, y_opt, r_opt
            
            # If invalid, try to repair by shrinking radii
            # This is a heuristic repair
            for scale in [0.99, 0.95, 0.9]:
                r_adjusted = r_opt * scale
                if check_validity(x_opt, y_opt, r_adjusted, tol=1e-5):
                    return x_opt, y_opt, r_adjusted
            
            return None
        except Exception:
            return None

    def greedy_best_fit_init(n_samples=2000):
        """
        Greedy Best-Fit initialization.
        Place circles one by one at the position that allows the largest radius.
        """
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Grid sampling for "best position" search
        # Use random sampling for better coverage of non-grid optimal spots
        xs = np.random.uniform(0.01, 0.99, n_samples)
        ys = np.random.uniform(0.01, 0.99, n_samples)
        
        for k in range(N):
            best_r = 0
            best_x = 0.5
            best_y = 0.5
            
            for i in range(n_samples):
                gx = xs[i]
                gy = ys[i]
                
                # Max radius at (gx, gy) given existing circles
                max_r = min(gx, 1-gx, gy, 1-gy)
                
                # Check against existing circles
                for j in range(k):
                    d = np.sqrt((gx - x[j])**2 + (gy - y[j])**2)
                    max_r = min(max_r, d - r[j])
                
                if max_r > best_r:
                    best_r = max_r
                    best_x = gx
                    best_y = gy
            
            # If best_r is very small, we might be stuck.
            # For 26 circles, it should be fine.
            x[k] = best_x
            y[k] = best_y
            r[k] = best_r
            
        return x, y, r

    def hexagonal_init(n_circles):
        """Create hexagonal packing initialization."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        row_spacing = 0.866
        col_spacing = 1.0
        
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
        
        total_width = (cols - 1) * col_spacing
        total_height = (rows - 1) * row_spacing
        
        margin_x = 0.05
        margin_y = 0.05
        
        scale_x = (1 - 2*margin_x) / max(total_width, 1e-6)
        scale_y = (1 - 2*margin_y) / max(total_height, 1e-6)
        scale = min(scale_x, scale_y, 1.0)
        
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x[i] = margin_x + col * col_spacing * scale + (row % 2) * 0.5 * col_spacing * scale
            y[i] = margin_y + row * row_spacing * scale
            r[i] = 0.5 * min(col_spacing, row_spacing) * scale * 0.95
        
        for i in range(n_circles):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r

    def random_init():
        """Random initialization."""
        r = np.random.uniform(0.03, 0.1, N)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        return x, y, r

    def grid_init():
        """Grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.95
        
        return x, y, r

    def center_ring_init():
        """Center-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.12
        
        n_ring = N - 1
        ring_radius = 0.35
        for i in range(1, N):
            angle = 2 * np.pi * (i - 1) / n_ring
            x[i] = 0.5 + ring_radius * np.cos(angle)
            y[i] = 0.5 + ring_radius * np.sin(angle)
            r[i] = 0.08
        
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r

    def two_ring_init():
        """Two-ring initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        x[0] = 0.5
        y[0] = 0.5
        r[0] = 0.10
        
        n_inner = 8
        inner_radius = 0.25
        for i in range(1, 1 + n_inner):
            angle = 2 * np.pi * (i - 1) / n_inner
            x[i] = 0.5 + inner_radius * np.cos(angle)
            y[i] = 0.5 + inner_radius * np.sin(angle)
            r[i] = 0.07
        
        n_outer = N - 1 - n_inner
        outer_radius = 0.42
        for i in range(1 + n_inner, N):
            idx = i - 1 - n_inner
            angle = 2 * np.pi * idx / n_outer + np.pi / n_outer
            x[i] = 0.5 + outer_radius * np.cos(angle)
            y[i] = 0.5 + outer_radius * np.sin(angle)
            r[i] = 0.06
        
        for i in range(N):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r

    def try_optimize(x0, y0, r0):
        """Try to optimize from given initial state, return result if better."""
        nonlocal best_sum, best_result
        
        # Step 1: Smooth optimization (L-BFGS-B)
        result_smooth = smooth_optimize(x0, y0, r0, max_iter=200)
        
        current_x, current_y, current_r = x0, y0, r0
        if result_smooth is not None:
            current_x, current_y, current_r = result_smooth
        
        # Step 2: High-precision SLSQP polish
        result_polish = polish_slsqp(current_x, current_y, current_r, max_iter=50, ftol=1e-10)
        
        final_x, final_y, final_r = current_x, current_y, current_r
        if result_polish is not None:
            final_x, final_y, final_r = result_polish
        
        current_sum = np.sum(final_r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = np.column_stack((final_x, final_y, final_r))

    # Multi-start optimization
    
    # 1. Greedy Best-Fit starts (strong initial feasible packing)
    for k in range(8):
        if time.time() > deadline:
            break
        x0, y0, r0 = greedy_best_fit_init(n_samples=1000)
        # Add small noise
        x0 += np.random.uniform(-0.005, 0.005, N)
        y0 += np.random.uniform(-0.005, 0.005, N)
        try_optimize(x0, y0, r0)

    # 2. Hexagonal starts
    for k in range(8):
        if time.time() > deadline:
            break
        x0, y0, r0 = hexagonal_init(N)
        x0 += np.random.uniform(-0.02, 0.02, N)
        y0 += np.random.uniform(-0.02, 0.02, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)
    
    # 3. Random starts
    for k in range(8):
        if time.time() > deadline:
            break
        x0, y0, r0 = random_init()
        try_optimize(x0, y0, r0)
    
    # 4. Grid starts
    for k in range(5):
        if time.time() > deadline:
            break
        x0, y0, r0 = grid_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        try_optimize(x0, y0, r0)
    
    # 5. Center-ring starts
    for k in range(5):
        if time.time() > deadline:
            break
        x0, y0, r0 = center_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)
    
    # 6. Two-ring starts
    for k in range(5):
        if time.time() > deadline:
            break
        x0, y0, r0 = two_ring_init()
        x0 += np.random.uniform(-0.01, 0.01, N)
        y0 += np.random.uniform(-0.01, 0.01, N)
        r0 *= np.random.uniform(0.9, 1.1, N)
        try_optimize(x0, y0, r0)

    # Final high-precision polish on the best result found
    if best_result is not None and time.time() < deadline - 5:
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]
        
        # Try SLSQP polish again with tighter tolerance
        result = polish_slsqp(x_best, y_best, r_best, max_iter=100, ftol=1e-12)
        if result is not None:
            x_polish, y_polish, r_polish = result
            current_sum = np.sum(r_polish)
            if current_sum > best_sum:
                best_sum = current_sum
                best_result = np.column_stack((x_polish, y_polish, r_polish))
        
    # Fallback to seed if no better solution found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
