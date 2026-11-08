import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # State variables:
    # pos: (N, 2) array of centers
    # r: (N,) array of radii
    # W: width of rectangle
    # H: height of rectangle (constrained: W + H = 2.0)
    
    def evaluate(pos, r, W, H):
        """
        Returns (value, penalty, valid)
        value: sum of radii (if valid)
        penalty: sum of constraint violations (overlaps, wall violations)
        valid: boolean
        """
        n = len(r)
        penalty = 0.0
        valid = True
        
        # Wall constraints
        # x in [r_i, W - r_i]
        # y in [r_i, H - r_i]
        if np.any(pos[:, 0] < r) or np.any(pos[:, 0] > W - r):
            valid = False
            # Penalty for wall violations
            wall_pen_x = np.sum(np.maximum(0, r - pos[:, 0])) + np.sum(np.maximum(0, pos[:, 0] - (W - r)))
            penalty += wall_pen_x * 10.0
            
        if np.any(pos[:, 1] < r) or np.any(pos[:, 1] > H - r):
            valid = False
            wall_pen_y = np.sum(np.maximum(0, r - pos[:, 1])) + np.sum(np.maximum(0, pos[:, 1] - (H - r)))
            penalty += wall_pen_y * 10.0

        # Circle-circle constraints
        # dist(i,j) >= r_i + r_j
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-9:
                    valid = False
                    penalty += (min_dist - dist) * 10.0

        value = np.sum(r) if valid else -np.inf
        return value, penalty, valid

    def compute_forces(pos, r, W, H):
        """
        Compute gradient of the objective (sum of radii) subject to constraints.
        We use a force-based approach:
        1. Repulsive forces between overlapping circles.
        2. Attractive forces towards walls if not touching? No, we want to maximize radius.
        
        Actually, a better approach for "maximize sum of radii" is to treat it as:
        Maximize sum(r_i)
        Subject to:
          x_i in [r_i, W - r_i]
          y_i in [r_i, H - r_i]
          dist(i,j) >= r_i + r_j
          W + H = 2
        
        We can use a penalty method or a barrier method.
        Let's use a simple iterative solver:
        1. Increase all radii slightly.
        2. Resolve overlaps by moving circles apart.
        3. Resolve wall violations by moving circles inward or reducing radius.
        
        Alternative: Simulated Annealing / Langevin Dynamics
        Objective: Maximize sum(r_i)
        Energy: -sum(r_i) + lambda * (sum of violations)^2
        
        Let's use a gradient-free local search with "growth" steps.
        
        Better Strategy: 
        1. Start with small circles.
        2. Iteratively increase radii.
        3. Use a relaxation scheme to find positions that fit the increased radii.
        """
        pass # Not used directly, logic embedded in main loop

    # Initial guess
    # Try a few different initial aspect ratios and configurations
    best_value = -np.inf
    best_pos = None
    best_r = None
    best_W = None
    best_H = None
    
    # Initial configurations to try
    # 1. 5x5 grid in 1x1 square
    # 2. 7x3 grid in 2x0.5 rectangle (very wide) - might not work well
    # 3. Random initialization
    
    configs = []
    
    # Config 1: 5x5 grid in 1x1
    W0 = 1.0
    H0 = 1.0
    pos0 = np.zeros((N, 2))
    r0 = np.full(N, 0.05)
    for i in range(N):
        row = i // 5
        col = i % 5
        pos0[i, 0] = (col + 0.5) * (W0 / 5)
        pos0[i, 1] = (row + 0.5) * (H0 / 5)
    configs.append((pos0, r0, W0, H0))
    
    # Config 2: Random
    for _ in range(3):
        W0 = 1.0 + np.random.uniform(-0.2, 0.2)
        H0 = 2.0 - W0
        if H0 <= 0.1: continue
        pos0 = np.random.uniform(0.1, 0.9, (N, 2))
        pos0[:, 0] *= W0
        pos0[:, 1] *= H0
        r0 = np.full(N, 0.05)
        configs.append((pos0, r0, W0, H0))
        
    for pos_init, r_init, W_init, H_init in configs:
        if time.time() > deadline:
            break
            
        pos = pos_init.copy()
        r = r_init.copy()
        W = W_init
        H = H_init
        
        # Optimization Loop
        # We will use a simple iterative method:
        # 1. Try to increase all radii by a small factor.
        # 2. Check feasibility. If infeasible, use a relaxation step to move circles.
        # 3. If feasible, accept the radius increase.
        # 4. Also, occasionally adjust W and H.
        
        current_value = np.sum(r)
        step_size = 1e-3
        no_improve_count = 0
        max_no_improve = 1000
        
        for step in range(20000):
            if time.time() > deadline:
                break
                
            # 1. Attempt to grow radii
            growth_factor = 1.0 + 1e-4
            new_r = r * growth_factor
            
            # 2. Check if new radii fit in current positions
            # We need to solve for new positions if they don't fit.
            # Instead of solving directly, we use a "push" method.
            
            # Check wall constraints for new_r
            # If pos[i,0] < new_r[i], push pos[i,0] to new_r[i]
            # If pos[i,0] > W - new_r[i], push pos[i,0] to W - new_r[i]
            
            # This is a non-linear constraint satisfaction problem.
            # Let's use a simple gradient descent on the penalty function.
            
            # Define penalty function for given pos, r, W, H
            def penalty_func(pos, r, W, H):
                pen = 0.0
                # Walls
                pen += np.sum(np.maximum(0, r - pos[:, 0]))**2
                pen += np.sum(np.maximum(0, pos[:, 0] - (W - r)))**2
                pen += np.sum(np.maximum(0, r - pos[:, 1]))**2
                pen += np.sum(np.maximum(0, pos[:, 1] - (H - r)))**2
                
                # Circles
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        min_d = r[i] + r[j]
                        if dist < min_d:
                            pen += (min_d - dist)**2
                return pen
            
            # We want to minimize penalty while keeping r as large as possible.
            # Actually, let's just try to find a valid configuration for new_r.
            # If we can, we accept new_r. If not, we keep old r and try to improve positions?
            # Or, we use the penalty gradient to adjust positions for the *current* r,
            # and only grow r when penalty is zero.
            
            # Strategy:
            # 1. Check current penalty.
            # 2. If penalty > 0, use gradient descent to reduce penalty (move circles).
            # 3. If penalty == 0, grow radii.
            
            pen = penalty_func(pos, r, W, H)
            
            if pen > 1e-6:
                # Reduce penalty by moving circles
                # Compute gradients of penalty w.r.t pos
                grad_pos = np.zeros_like(pos)
                
                # Wall gradients
                # d/dx max(0, r - x) = -1 if x < r else 0
                mask_left_x = pos[:, 0] < r
                grad_pos[mask_left_x, 0] -= 2 * (r[mask_left_x] - pos[mask_left_x, 0])
                
                mask_right_x = pos[:, 0] > W - r
                grad_pos[mask_right_x, 0] += 2 * (pos[mask_right_x, 0] - (W - r[mask_right_x]))
                
                mask_left_y = pos[:, 1] < r
                grad_pos[mask_left_y, 1] -= 2 * (r[mask_left_y] - pos[mask_left_y, 1])
                
                mask_right_y = pos[:, 1] > H - r
                grad_pos[mask_right_y, 1] += 2 * (pos[mask_right_y, 1] - (H - r[mask_right_y]))
                
                # Circle gradients
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        min_d = r[i] + r[j]
                        if dist < min_d and dist > 1e-12:
                            diff = min_d - dist
                            # Gradient of (diff)^2 w.r.t pos_i
                            # d/dx_i (min_d - dist)^2 = 2*(min_d-dist) * (-d(dist)/dx_i)
                            # d(dist)/dx_i = dx / dist
                            grad = 2 * diff * (dx / dist)
                            grad_pos[i, 0] -= grad # Wait, sign check.
                            # Let f = (min_d - dist)^2.
                            # df/dx_i = 2(min_d - dist) * (-1) * d(dist)/dx_i
                            # d(dist)/dx_i = (x_i - x_j) / dist
                            # So df/dx_i = -2(min_d - dist) * (x_i - x_j) / dist
                            # We want to move in direction -grad to minimize f.
                            # So update pos -= lr * grad.
                            # If x_i > x_j (dx > 0) and dist < min_d, we want to increase dx (push apart).
                            # grad_x_i = -2 * positive * positive / positive = negative.
                            # pos_i -= lr * negative -> pos_i increases. Correct.
                            
                            grad_pos[i, 0] = -2 * diff * (dx / dist)
                            grad_pos[j, 0] = -2 * diff * (-dx / dist) # d(dist)/dx_j = -dx/dist
                            
                            grad_pos[i, 1] = -2 * diff * (dy / dist)
                            grad_pos[j, 1] = -2 * diff * (-dy / dist)
                            
                # Update positions
                lr = 0.1
                pos -= lr * grad_pos
                
                # Ensure positions stay within bounds roughly to avoid exploding
                # pos[:, 0] = np.clip(pos[:, 0], 0, W)
                # pos[:, 1] = np.clip(pos[:, 1], 0, H)
                
            else:
                # Penalty is low, try to grow radii
                new_r = r * (1.0 + 1e-5)
                # Check if new_r is feasible with current pos
                # We can just set r = new_r and let the next loop handle the penalty
                r = new_r
                no_improve_count = 0
                
            # Occasionally adjust aspect ratio
            if step % 100 == 0:
                # Try to adjust W slightly
                dW = np.random.uniform(-0.01, 0.01)
                new_W = W + dW
                new_H = 2.0 - new_W
                if new_W > 0.1 and new_H > 0.1:
                    # Scale positions to new aspect ratio?
                    # Or just keep positions and let penalty solver fix it?
                    # Scaling might be better to maintain relative structure
                    pos[:, 0] *= (new_W / W)
                    pos[:, 1] *= (new_H / H)
                    W = new_W
                    H = new_H
            
            # Update best
            val, _, valid = evaluate(pos, r, W, H)
            if valid and val > best_value:
                best_value = val
                best_pos = pos.copy()
                best_r = r.copy()
                best_W = W
                best_H = H
                no_improve_count = 0
            elif not valid:
                # If we made it worse, revert?
                # For now, we just let the penalty solver fix it.
                pass
            
            if no_improve_count > max_no_improve:
                break

    # Final refinement
    # If we have a valid best, try a few more small perturbations
    if best_pos is not None:
        # ... (omitted for brevity, the main loop does the work)
        pass

    # Construct result
    result = np.zeros((N, 3))
    if best_pos is not None:
        result[:, 0] = best_pos[:, 0]
        result[:, 1] = best_pos[:, 1]
        result[:, 2] = best_r
    else:
        # Fallback to grid
        W = 1.0
        H = 1.0
        for i in range(N):
            row = i // 5
            col = i % 5
            result[i, 0] = (col + 0.5) * 0.2
            result[i, 1] = (row + 0.5) * 0.2
            result[i, 2] = 0.1

    return result
