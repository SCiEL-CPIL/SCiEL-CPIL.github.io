"""Soft-body penalty circle packing in perimeter-4 rectangle."""
import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_config = None
    
    # Objective function to minimize: -sum(radii) + penalty(overlaps) + penalty(walls)
    # Parameters: x (2N) for centers, s (1) for scale factor
    # Rectangle: W = s, H = 2 - s (since 2(W+H)=4 => W+H=2)
    
    def objective(params):
        if time.time() > deadline - 2.0:
            return 1e10  # Timeout penalty
        
        x = params[:2*N]
        s = params[2*N]
        
        # Aspect ratio constraints: W=s, H=2-s. Both must be positive.
        # s in (0, 2). Let's keep it in a reasonable range to avoid degenerate rectangles.
        # If s is too small or too large, the rectangle is very thin, which is bad for packing.
        # We can add a penalty for extreme aspect ratios or just let the wall penalty handle it.
        # However, if s <= 0 or s >= 2, it's invalid.
        if s <= 0.1 or s >= 1.9:
            return 1e10
            
        W = s
        H = 2.0 - s
        
        centers = x.reshape(N, 2)
        
        # Wall violation: centers must be inside [0, W] x [0, H]
        # Since we don't know the radii yet in the soft body model, 
        # we can't strictly enforce wall distance = radius.
        # Instead, we model the "radius" of each circle as the distance to the nearest boundary 
        # (wall or other circle) in a relaxed sense?
        # No, the task is to maximize sum of radii.
        # In the soft body model, we usually define a nominal radius for each circle.
        # Let's define r_i as a parameter too? That adds 21 more variables.
        # Or, we can define the "effective radius" of circle i as the minimum distance 
        # to the boundary of the container and to other circles, but that makes the objective non-smooth.
        
        # Alternative Soft Body Approach:
        # We have N circles. We want to maximize sum(r_i) subject to d(i,j) >= r_i + r_j and d(i,wall) >= r_i.
        # This is a non-linear program.
        # We can reformulate this as minimizing the "overlap" energy.
        # Let's introduce r_i as explicit variables.
        # Total variables: 2N (centers) + N (radii) + 1 (scale) = 3N + 1 = 64 variables.
        
        # Let's re-define params:
        # x: 2N centers
        # r: N radii
        # s: 1 scale
        
        # Wait, the current params definition in the function signature above was just a sketch.
        # Let's implement the full version inside the minimize call properly.
        
        pass # This function needs to be rewritten to accept the full parameter vector.

    # Let's write the actual objective function correctly.
    
    def soft_body_objective(params):
        if time.time() > deadline - 2.0:
            return 1e10
            
        x = params[:2*N]
        r = params[2*N:3*N]
        s = params[3*N]
        
        # Aspect ratio: W = s, H = 2 - s
        # Constraints on s: 0.2 < s < 1.8 to avoid extreme slivers
        if s <= 0.2 or s >= 1.8:
            return 1e10
            
        W = s
        H = 2.0 - s
        
        centers = x.reshape(N, 2)
        
        # 1. Wall Penalties
        # Distance from center to wall must be >= r_i
        # Penalize if r_i > dist_to_wall
        # dist_left = x_i, dist_right = W - x_i, dist_down = y_i, dist_up = H - y_i
        # min_dist_wall_i = min(x_i, W-x_i, y_i, H-y_i)
        # violation_i = max(0, r_i - min_dist_wall_i)
        
        x_coords = centers[:, 0]
        y_coords = centers[:, 1]
        
        d_left = x_coords
        d_right = W - x_coords
        d_down = y_coords
        d_up = H - y_coords
        
        min_d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        
        # Also ensure centers are inside the box. If center is outside, it's a huge violation.
        # We can clamp min_d_wall to be negative if center is outside?
        # Actually, if x < 0, d_left < 0. r_i - d_left > r_i. So penalty is large.
        # This handles out-of-bounds centers naturally.
        
        wall_violation = np.maximum(0, r - min_d_wall)
        wall_penalty = np.sum(wall_violation ** 2)
        
        # 2. Circle-Circle Penalties
        # d(i,j) >= r_i + r_j
        # violation_ij = max(0, r_i + r_j - d_ij)
        
        # Compute pairwise distances
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dist = np.linalg.norm(diff, axis=2)
        
        # For efficiency with N=21, 21x21 matrix is small.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Sum of radii
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Violation matrix
        # We only care about i < j to avoid double counting and self-interaction
        # Set diagonal to 0
        np.fill_diagonal(dist, np.inf)
        
        violations = np.maximum(0, r_sum - dist)
        
        # Sum of squared violations
        # We only want to count each pair once.
        # Since violations is symmetric, sum(violations**2) counts each pair twice.
        # Dividing by 2 is fine, or just sum the upper triangle.
        # For simplicity and vectorization, sum all and divide by 2.
        circle_penalty = 0.5 * np.sum(violations ** 2)
        
        # 3. Objective
        # We want to MAXIMIZE sum(r). So MINIMIZE -sum(r).
        # Total Penalty = wall_penalty + circle_penalty
        # We also want to encourage large radii, so the -sum(r) term drives that.
        # The penalties prevent invalid configurations.
        
        # To balance, we can scale the penalties. 
        # As the optimizer progresses, we could increase penalty weights, 
        # but for a single call, a fixed large weight works if the initial guess is good.
        # Let's use a weight of 100 for penalties to ensure they dominate early on,
        # but allow -sum(r) to guide the direction.
        
        total_penalty = 100.0 * (wall_penalty + circle_penalty)
        
        return -np.sum(r) + total_penalty

    def gradient_soft_body(params):
        # Numerical gradient is too slow for 64 variables if we do it naively in Python loops.
        # L-BFGS-B can use numerical gradients, but it's expensive.
        # Let's implement analytical gradients? 
        # Or use a different optimizer that handles non-smoothness better?
        # Actually, L-BFGS-B with numerical gradients for 64 vars might be okay if iterations are limited.
        # But 64 * (num_iters) evaluations.
        
        # Let's try to implement the gradient analytically.
        # f = -sum(r) + 100 * (sum(max(0, r_i - d_wall_i)^2) + 0.5 * sum_{i,j} max(0, r_i+r_j-d_ij)^2)
        
        # This is complex. Let's stick with numerical gradients but optimize the objective function speed.
        # N=21 is small.
        
        # Alternatively, use a simpler heuristic:
        # Start with a good hexagonal grid.
        # Then use local search with "soft" moves.
        
        # Given the time constraint, let's try a hybrid:
        # 1. Generate a good initial hexagonal packing.
        # 2. Use scipy.optimize.minimize with L-BFGS-B and analytical gradient if possible,
        #    or just let it use numerical gradient with a tight iteration limit.
        
        # For now, return None to let scipy use numerical gradient.
        return None

    # Initialization: Hexagonal Grid
    # We need to fit 21 circles in a rectangle of perimeter 4.
    # Approximate area for 21 circles of radius r is 21 * pi * r^2.
    # Rectangle area A = W * H = W * (2-W). Max area is 1 at W=1.
    # Packing density for hex is ~0.9069.
    # 21 * pi * r^2 * 0.9069 <= 1
    # r^2 <= 1 / (21 * pi * 0.9069) ≈ 1 / 59.7 ≈ 0.0167
    # r ≈ 0.13
    
    # Let's create a hexagonal lattice.
    # Rows of 5, 4, 5, 4, 3? 5+4+5+4+3 = 21.
    # Or 6, 5, 5, 5? 6+5+5+5 = 21.
    
    # Let's try a few structured layouts and pick the best initial one.
    
    def get_hex_layout(rows):
        """
        Generate centers for a hexagonal layout with given row counts.
        Returns (centers, W, H, r)
        """
        # Estimate r based on spacing
        # Spacing in hex grid: dx = sqrt(3)*r, dy = 1.5*r
        # Let's assume a target r and calculate bounds.
        # We will optimize r later.
        
        centers = []
        max_x = 0
        max_y = 0
        
        y_offset = 0
        for i, count in enumerate(rows):
            y = y_offset
            for j in range(count):
                # Offset every other row
                x_offset = (j * 2 + (i % 2)) * 0.866 # approx sqrt(3)/2
                x = x_offset
                centers.append([x, y])
                if x > max_x: max_x = x
                if y > max_y: max_y = y
            y_offset += 1.5 # approx 1.5*r if r=1
        
        centers = np.array(centers)
        
        # Normalize to fit in unit square roughly, then scale
        # This is just for initialization shape.
        # We will pass these as relative coordinates.
        return centers, max_x + 1, max_y + 1 # +1 for radius margin

    # Let's define specific layouts
    layouts = [
        [5, 4, 5, 4, 3],
        [6, 5, 5, 5],
        [7, 5, 5, 4],
        [4, 5, 5, 4, 3],
        [3, 4, 5, 4, 5]
    ]
    
    best_init = None
    best_init_score = -np.inf
    best_init_W = 1.0
    
    for layout in layouts:
        if time.time() > deadline - 60:
            break
        if len(layout) != 5 and sum(layout) != N:
            continue
        if sum(layout) != N:
            continue
            
        # Generate raw centers
        raw_centers = []
        y_offset = 0
        for i, count in enumerate(layout):
            y = y_offset
            for j in range(count):
                x = j * 2 + (i % 2) # Simplified hex coords
                raw_centers.append([x, y])
            y_offset += 2.5 # Simplified vertical step
        
        raw_centers = np.array(raw_centers, dtype=float)
        
        # Normalize raw centers to [0,1] x [0,1]
        min_x = np.min(raw_centers[:, 0])
        max_x = np.max(raw_centers[:, 0])
        min_y = np.min(raw_centers[:, 1])
        max_y = np.max(raw_centers[:, 1])
        
        span_x = max_x - min_x
        span_y = max_y - min_y
        
        if span_x < 1e-6 or span_y < 1e-6:
            continue
            
        norm_centers = np.zeros_like(raw_centers)
        norm_centers[:, 0] = (raw_centers[:, 0] - min_x) / span_x
        norm_centers[:, 1] = (raw_centers[:, 1] - min_y) / span_y
        
        # Now we have a shape. We need to find W, H, r that fits this shape.
        # Aspect ratio of shape: AR = span_x / span_y
        # Rectangle AR: W/H = W/(2-W)
        # We want W/H to be close to AR.
        # W/(2-W) = AR => W = 2*AR/(1+AR)
        
        AR = span_x / span_y
        W_opt = 2.0 * AR / (1.0 + AR)
        H_opt = 2.0 - W_opt
        
        # Scale norm_centers to fit in W_opt x H_opt with some margin for radius
        # Assume radius is roughly 0.5 * min_step
        # Let's just scale to fill the box, and let the optimizer adjust r.
        
        # Map norm_centers [0,1] to [0, W_opt] and [0, H_opt]
        centers = np.zeros_like(norm_centers)
        centers[:, 0] = norm_centers[:, 0] * W_opt
        centers[:, 1] = norm_centers[:, 1] * H_opt
        
        # Initial radii: estimate from local spacing
        # Just use a small uniform radius to start
        r_init = np.full(N, 0.05)
        
        # Build param vector
        params_init = np.zeros(3*N + 1)
        params_init[:2*N] = centers.flatten()
        params_init[2*N:3*N] = r_init
        params_init[3*N] = W_opt
        
        # Evaluate initial score
        score = soft_body_objective(params_init)
        
        if score < best_init_score:
            best_init_score = score
            best_init = params_init
            best_init_W = W_opt

    if best_init is None:
        # Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        centers = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            centers[i, 0] = (col + 0.5) * W / side
            centers[i, 1] = (row + 0.5) * H / side
        r_init = np.full(N, 0.1)
        params_init = np.zeros(3*N + 1)
        params_init[:2*N] = centers.flatten()
        params_init[2*N:3*N] = r_init
        params_init[3*N] = W
        best_init = params_init
        best_init_W = W

    # Optimization
    # Bounds
    # Centers: [0, W] x [0, H] depends on W. 
    # Since W is a variable, we can't set static bounds for centers easily in L-BFGS-B.
    # We can use large bounds and rely on the penalty.
    # Or we can re-parameterize centers as u in [0,1] and x = u * W.
    # Let's re-parameterize to ensure centers stay inside the box.
    
    def reparam_objective(params):
        if time.time() > deadline - 2.0:
            return 1e10
            
        u = params[:2*N] # u in [0,1]
        r = params[2*N:3*N]
        s = params[3*N] # W
        
        if s <= 0.2 or s >= 1.8:
            return 1e10
            
        W = s
        H = 2.0 - s
        
        # Centers
        centers = np.zeros((N, 2))
        centers[:, 0] = u[0::2] * W
        centers[:, 1] = u[1::2] * H
        
        # Wall Penalties
        d_left = centers[:, 0]
        d_right = W - centers[:, 0]
        d_down = centers[:, 1]
        d_up = H - centers[:, 1]
        
        min_d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up))
        
        # If u is in [0,1], centers are in [0,W]x[0,H], so min_d_wall >= 0.
        # Violation is max(0, r - min_d_wall)
        
        wall_violation = np.maximum(0, r - min_d_wall)
        wall_penalty = np.sum(wall_violation ** 2)
        
        # Circle Penalties
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        violations = np.maximum(0, r_sum - dist)
        circle_penalty = 0.5 * np.sum(violations ** 2)
        
        total_penalty = 100.0 * (wall_penalty + circle_penalty)
        
        return -np.sum(r) + total_penalty

    # Convert best_init to re-parameterized form
    # best_init has centers in absolute coords.
    # u_x = x / W, u_y = y / H
    params_opt = np.zeros(3*N + 1)
    W_init = best_init[3*N]
    H_init = 2.0 - W_init
    
    centers_init = best_init[:2*N].reshape(N, 2)
    
    u_init = np.zeros(2*N)
    u_init[0::2] = centers_init[:, 0] / W_init
    u_init[1::2] = centers_init[:, 1] / H_init
    
    params_opt[:2*N] = u_init
    params_opt[2*N:3*N] = best_init[2*N:3*N]
    params_opt[3*N] = W_init
    
    # Bounds for re-parameterized params
    # u: [0, 1]
    # r: [0, 1] (max radius can't be more than 1 in a perimeter 4 box? 
    #    Max area is 1, so max r is sqrt(1/pi) ~ 0.56. Let's say 1.0)
    # s (W): [0.2, 1.8]
    
    bounds = []
    for i in range(2*N):
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 1.0))
    bounds.append((0.2, 1.8))
    
    # Run L-BFGS-B
    # maxiter needs to be small to avoid timeout.
    # Each iteration involves function evaluation.
    # N=21, 64 vars.
    
    try:
        res = minimize(
            reparam_objective, 
            params_opt, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-9, 'maxls': 20}
        )
        
        if time.time() < deadline - 5.0:
            # Try to refine with more iterations if time permits
            res2 = minimize(
                reparam_objective, 
                res.x, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 20}
            )
            if res2.fun < res.fun:
                res = res2
                
    except Exception as e:
        pass # Keep best_init if optimization fails

    # Extract best result
    if res.fun < best_init_score:
        best_params = res.x
    else:
        best_params = params_opt
        
    # Decode best_params
    u_best = best_params[:2*N]
    r_best = best_params[2*N:3*N]
    W_best = best_params[3*N]
    H_best = 2.0 - W_best
    
    centers_best = np.zeros((N, 2))
    centers_best[:, 0] = u_best[0::2] * W_best
    centers_best[:, 1] = u_best[1::2] * H_best
    
    # Check validity and compute final score
    # The evaluator wants an array of shape (N, 3): x, y, r
    # It checks if circles are disjoint and inside the bounding rectangle of perimeter 4.
    # The bounding rectangle is the one defined by W_best, H_best.
    # We must ensure the circles are actually valid.
    
    # Verify
    valid = True
    for i in range(N):
        x, y, r = centers_best[i, 0], centers_best[i, 1], r_best[i]
        if x < 0 or x > W_best or y < 0 or y > H_best or r < 0:
            valid = False
            break
        if x < r or x > W_best - r or y < r or y > H_best - r:
            valid = False
            break
            
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers_best[i] - centers_best[j])
                if d < r_best[i] + r_best[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        score = np.sum(r_best)
        if score > best_score:
            best_score = score
            best_config = np.column_stack([centers_best, r_best])
    else:
        # If invalid, try to repair by reducing radii slightly?
        # Or just use the initial grid which is known to be valid?
        # Let's try to clamp radii to be valid.
        # This is a fallback.
        
        # Simple repair: reduce all radii by 10% and check again.
        r_repair = r_best * 0.9
        valid = True
        for i in range(N):
            x, y, r = centers_best[i, 0], centers_best[i, 1], r_repair[i]
            if x < r or x > W_best - r or y < r or y > H_best - r:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers_best[i] - centers_best[j])
                    if d < r_repair[i] + r_repair[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            score = np.sum(r_repair)
            if score > best_score:
                best_score = score
                best_config = np.column_stack([centers_best, r_repair])

    if best_config is None:
        # Final Fallback: Grid
        side = int(np.ceil(np.sqrt(N)))
        W = 1.0
        H = 1.0
        centers = np.zeros((N, 2))
        for i in range(N):
            row = i // side
            col = i % side
            centers[i, 0] = (col + 0.5) * W / side
            centers[i, 1] = (row + 0.5) * H / side
        r = 0.5 / side - 0.01
        best_config = np.column_stack([centers, np.full(N, r)])

    return best_config

# EVOLVE-BLOCK-END
