"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Pre-compute indices for pair checks
    i_indices, j_indices = np.triu_indices(N, k=1)
    
    def is_valid(pos, rad, tol=1e-8):
        # Check boundaries
        if np.any(pos[:, 0] - rad < -tol) or np.any(pos[:, 0] + rad > 1 + tol):
            return False
        if np.any(pos[:, 1] - rad < -tol) or np.any(pos[:, 1] + rad > 1 + tol):
            return False
        
        # Check overlaps
        dx = pos[i_indices, 0] - pos[j_indices, 0]
        dy = pos[i_indices, 1] - pos[j_indices, 1]
        dist_sq = dx*dx + dy*dy
        rad_sum = rad[i_indices] + rad[j_indices]
        # dist < rad_sum  => dist_sq < rad_sum^2
        if np.any(dist_sq < rad_sum**2 - tol):
            return False
        return True

    def initial_placement(seed):
        np.random.seed(seed)
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Hexagonal grid initialization
        # For 26 circles, a 5x6 grid (30 cells) with 4 removed or 5x5 (25) + 1
        # Let's use a 5x5 grid plus one extra, or a 6x5 grid.
        # 5x5 = 25 circles. Add 1 more.
        # Or just use a grid of 5x6 = 30 and remove 4 random ones?
        # Better: Use a dense hexagonal pattern.
        
        # Simple grid 5x6
        side_x = 5
        side_y = 6
        total_grid = side_x * side_y
        if total_grid < N:
             side_x = 6
             side_y = 5
             
        for i in range(N):
            row = i // side_x
            col = i % side_x
            # Hex offset
            offset = 0.5 / side_x if row % 2 == 1 else 0
            positions[i, 0] = (col + offset + 0.5) / side_x
            positions[i, 1] = (row + 0.5) / side_y
            radii[i] = 0.4 / max(side_x, side_y)
            
        # Add jitter
        positions += np.random.uniform(-0.02, 0.02, (N, 2))
        radii += np.random.uniform(-0.005, 0.005, N)
        
        # Clip
        positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
        positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.95)
        radii = np.clip(radii, 0.01, 0.2)
        
        return positions, radii

    def optimize_run(init_pos, init_rad, time_budget, seed):
        np.random.seed(seed)
        pos = init_pos.copy()
        rad = init_rad.copy()
        
        # State for momentum
        vel_pos = np.zeros_like(pos)
        vel_rad = np.zeros_like(rad)
        
        # Hyperparameters
        lr_pos = 0.01
        lr_rad = 0.01
        momentum = 0.9
        penalty_weight = 100.0
        radius_weight = 1.0
        
        start_time = time.time()
        best_score = np.sum(rad)
        best_pos = pos.copy()
        best_rad = rad.copy()
        
        # We want to minimize:
        # E = penalty_weight * (overlap_penalty + boundary_penalty) - radius_weight * sum(rad)
        # Overlap penalty: sum(max(0, r_i + r_j - dist)^2)
        # Boundary penalty: sum(max(0, r - x)^2 + max(0, r - (1-x))^2 + ...)
        
        while time.time() - start_time < time_budget:
            # Compute gradients
            grad_pos = np.zeros_like(pos)
            grad_rad = np.zeros_like(rad)
            
            # 1. Boundary Gradients
            # Left/Right
            viol_left = np.maximum(0, rad - pos[:, 0])
            viol_right = np.maximum(0, rad - (1 - pos[:, 0]))
            viol_bottom = np.maximum(0, rad - pos[:, 1])
            viol_top = np.maximum(0, rad - (1 - pos[:, 1]))
            
            # dE/dx for left: -2 * viol_left (if active)
            # dE/dx for right: 2 * viol_right (if active)
            grad_pos[:, 0] += -2 * penalty_weight * viol_left
            grad_pos[:, 0] += 2 * penalty_weight * viol_right
            grad_pos[:, 1] += -2 * penalty_weight * viol_bottom
            grad_pos[:, 1] += 2 * penalty_weight * viol_top
            
            # dE/dr for boundaries
            grad_rad += 2 * penalty_weight * (viol_left + viol_right + viol_bottom + viol_top)
            
            # 2. Overlap Gradients
            dx = pos[i_indices, 0] - pos[j_indices, 0]
            dy = pos[i_indices, 1] - pos[j_indices, 1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            rad_sum = rad[i_indices] + rad[j_indices]
            
            # overlap = max(0, rad_sum - dist)
            overlap = np.maximum(0, rad_sum - dist)
            
            # dE/ddist = -2 * penalty_weight * overlap
            # dE/dx_i = -2 * penalty_weight * overlap * (dx / dist)
            # dE/dx_j = -2 * penalty_weight * overlap * (-dx / dist)
            
            force_vec = -2 * penalty_weight * overlap / dist
            # force_vec has shape (num_pairs,)
            
            # Accumulate to grad_pos
            # For each pair (i, j):
            # grad_pos[i] += force_vec * (dx, dy)  -- wait, dE/dx_i = -dE/ddist * ddist/dx_i
            # ddist/dx_i = dx/dist
            # So dE/dx_i = (-2 * penalty_weight * overlap) * (dx/dist)
            # Let's re-verify signs.
            # E = (rad_sum - dist)^2
            # dE/ddist = -2(rad_sum - dist)
            # ddist/dx_i = (x_i - x_j)/dist
            # dE/dx_i = -2(rad_sum - dist) * (x_i - x_j)/dist
            # If overlap > 0, rad_sum > dist.
            # We want to increase dist, so we want to move i away from j.
            # If x_i > x_j, dx > 0. dE/dx_i is negative. Gradient descent moves x_i in direction -grad, so +x. Correct.
            
            grad_pos[i_indices, 0] += force_vec * dx
            grad_pos[i_indices, 1] += force_vec * dy
            grad_pos[j_indices, 0] -= force_vec * dx
            grad_pos[j_indices, 1] -= force_vec * dy
            
            # dE/dr_i = 2(rad_sum - dist)
            grad_rad[i_indices] += 2 * penalty_weight * overlap
            grad_rad[j_indices] += 2 * penalty_weight * overlap
            
            # 3. Radius Maximization Gradient
            # We want to maximize sum(rad), so minimize -sum(rad)
            # d(-sum(rad))/dr_i = -1
            grad_rad -= radius_weight
            
            # Update with Momentum
            vel_pos = momentum * vel_pos - lr_pos * grad_pos
            vel_rad = momentum * vel_rad - lr_rad * grad_rad
            
            pos += vel_pos
            rad += vel_rad
            
            # Clip
            rad = np.clip(rad, 0.001, 0.5)
            pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
            pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
            
            # Check validity and update best
            # We only update best if it's strictly valid (or very close)
            if is_valid(pos, rad, tol=1e-6):
                score = np.sum(rad)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_rad = rad.copy()
            
            # Adaptive learning rate? Maybe reduce lr over time
            # lr_pos *= 0.9999
            # lr_rad *= 0.9999
            
        return best_pos, best_rad, best_score

    # Run multiple optimization runs
    start_time = time.time()
    time_limit = 108
    
    best_overall_pos = None
    best_overall_rad = None
    best_overall_score = -np.inf
    
    num_runs = 20
    time_per_run = time_limit / num_runs
    
    for run in range(num_runs):
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 0.5:
            break
        
        seed = 100 + run
        init_pos, init_rad = initial_placement(seed)
        
        pos, rad, score = optimize_run(init_pos, init_rad, min(time_per_run, remaining_time), seed)
        
        if pos is not None and score > best_overall_score:
            best_overall_score = score
            best_overall_pos = pos
            best_overall_rad = rad
            
    # Final validation
    if best_overall_pos is None or not is_valid(best_overall_pos, best_overall_rad, tol=1e-5):
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            cells.append([(col + 0.5) / side, (row + 0.5) / side, 0.3 / side])
        return np.array(cells)
    
    return np.column_stack([best_overall_pos, best_overall_rad])
# EVOLVE-BLOCK-END
