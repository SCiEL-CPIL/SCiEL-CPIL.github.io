"""Circle packing in unit square - optimized approach."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Pack N circles with disjoint interiors inside [0,1]^2 maximizing sum of radii.
    Uses a custom vectorized gradient descent with momentum and multiple restarts.
    """
    n = N
    best_score = -np.inf
    best_result = None
    
    # Parameters for optimization
    n_restarts = 50
    n_iter = 2000
    lr = 0.005
    momentum = 0.9
    penalty_weight = 50.0
    boundary_weight = 50.0
    
    # Precompute pairs for overlap constraints
    # i, j such that i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
    for restart in range(n_restarts):
        # Initialization: Random positions, small radii
        # Use a grid-like init for some restarts, random for others
        if restart % 2 == 0:
            # Grid init
            side = int(np.ceil(np.sqrt(n)))
            xs = np.zeros(n)
            ys = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xs[i] = (col + 0.5) / side
                ys[i] = (row + 0.5) / side
            # Add small noise
            xs += np.random.uniform(-0.05, 0.05, n)
            ys += np.random.uniform(-0.05, 0.05, n)
        else:
            # Random init
            xs = np.random.uniform(0.1, 0.9, n)
            ys = np.random.uniform(0.1, 0.9, n)
            
        rs = np.full(n, 0.05)
        
        # Clip initial positions to be inside
        xs = np.clip(xs, 0.05, 0.95)
        ys = np.clip(ys, 0.05, 0.95)
        
        # State: x, y, r
        # Velocities for momentum
        vx = np.zeros(n)
        vy = np.zeros(n)
        vr = np.zeros(n)
        
        current_xs = xs.copy()
        current_ys = ys.copy()
        current_rs = rs.copy()
        
        for it in range(n_iter):
            # Calculate gradients
            grad_x = np.zeros(n)
            grad_y = np.zeros(n)
            grad_r = np.full(n, -1.0) # Objective: -sum(r)
            
            # 1. Boundary Penalties
            # Left: max(0, r - x)
            viol_l = np.maximum(0, current_rs - current_xs)
            grad_x -= boundary_weight * 2 * viol_l
            grad_r += boundary_weight * 2 * viol_l
            
            # Right: max(0, x + r - 1)
            viol_r = np.maximum(0, current_xs + current_rs - 1)
            grad_x += boundary_weight * 2 * viol_r
            grad_r += boundary_weight * 2 * viol_r
            
            # Bottom: max(0, r - y)
            viol_b = np.maximum(0, current_rs - current_ys)
            grad_y -= boundary_weight * 2 * viol_b
            grad_r += boundary_weight * 2 * viol_b
            
            # Top: max(0, y + r - 1)
            viol_t = np.maximum(0, current_ys + current_rs - 1)
            grad_y += boundary_weight * 2 * viol_t
            grad_r += boundary_weight * 2 * viol_t
            
            # 2. Overlap Penalties
            # dist^2 between i and j
            dx = current_xs[i_idx] - current_xs[j_idx]
            dy = current_ys[i_idx] - current_ys[j_idx]
            dist_sq = dx*dx + dy*dy
            # Avoid division by zero
            dist_sq = np.maximum(dist_sq, 1e-12)
            dist = np.sqrt(dist_sq)
            
            min_dist = current_rs[i_idx] + current_rs[j_idx]
            # Overlap amount: min_dist - dist
            overlap = min_dist - dist
            
            # Only contribute if overlapping
            mask = overlap > 0
            if np.any(mask):
                # Gradient of (overlap)^2 w.r.t positions and radii
                # Let d = dist, m = r_i + r_j
                # f = (m - d)^2
                # df/dx_i = 2(m-d) * (-dx/d)
                # df/dr_i = 2(m-d) * 1
                
                coeff = 2 * penalty_weight * overlap[mask]
                
                # Update grad_x for i and j
                # dx = x_i - x_j
                # d(dist)/dx_i = dx / dist
                # d(overlap)/dx_i = -dx / dist
                # d(overlap^2)/dx_i = 2 * overlap * (-dx / dist)
                
                # We need to scatter these back to the arrays
                # Using bincount or scatter_add is efficient
                # Note: i_idx and j_idx are the indices of the pairs
                
                # Contribution to grad_x[i] from pair (i,j): coeff * (-dx[mask] / dist[mask])
                # Contribution to grad_x[j] from pair (i,j): coeff * (dx[mask] / dist[mask])
                
                factor = coeff / dist[mask]
                
                # Scatter to i
                np.add.at(grad_x, i_idx[mask], -factor * dx[mask])
                np.add.at(grad_y, i_idx[mask], -factor * dy[mask])
                np.add.at(grad_r, i_idx[mask], factor) # Wait, derivative of (m-d)^2 wrt r_i is 2(m-d). 
                # Let's re-verify: f = (r_i + r_j - d)^2. 
                # df/dr_i = 2(r_i + r_j - d) * 1.
                # So grad_r[i] += 2 * penalty_weight * overlap.
                # My `coeff` is already `2 * penalty_weight * overlap`.
                # So `grad_r[i] += coeff`.
                # Correct.
                
                # Scatter to j
                np.add.at(grad_x, j_idx[mask], factor * dx[mask])
                np.add.at(grad_y, j_idx[mask], factor * dy[mask])
                np.add.at(grad_r, j_idx[mask], coeff)
                
                # Note: np.add.at handles duplicate indices correctly (accumulation)
            
            # Update velocities and positions with momentum
            # Adaptive learning rate? Simple decay.
            current_lr = lr * (0.99 ** (it / 100))
            
            vx = momentum * vx - current_lr * grad_x
            vy = momentum * vy - current_lr * grad_y
            vr = momentum * vr - current_lr * grad_r
            
            current_xs += vx
            current_ys += vy
            current_rs += vr
            
            # Clamp radii to be positive
            current_rs = np.maximum(current_rs, 0.0)
            
            # Optional: Clamp positions to [0,1] to help convergence, 
            # but penalties should handle it. Clamping might cause discontinuities in gradient if not careful.
            # We'll rely on penalties.
            
        # End of iterations for this restart
        
        # Check validity of the final configuration
        # 1. Inside bounds
        valid = True
        if np.any(current_xs < current_rs - 1e-6) or np.any(current_xs > 1 - current_rs + 1e-6):
            valid = False
        if np.any(current_ys < current_rs - 1e-6) or np.any(current_ys > 1 - current_rs + 1e-6):
            valid = False
        
        # 2. No overlaps
        if valid:
            dx = current_xs[i_idx] - current_xs[j_idx]
            dy = current_ys[i_idx] - current_ys[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = current_rs[i_idx] + current_rs[j_idx]
            if np.any(dist < min_dist - 1e-6):
                valid = False
        
        if valid:
            score = np.sum(current_rs)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([current_xs, current_ys, current_rs])
        
        # Early stopping if we find a very good solution? 
        # Theoretical max for 26 circles is around 1.0-1.1? 
        # Grid 5x5 (25 circles) radius ~ 0.1. Sum ~ 2.5? No, 0.1*25 = 2.5.
        # Wait, side=5, spacing=0.2, r=0.1. Sum = 2.5.
        # If we can pack tighter, sum can be higher.
        # Let's not stop early to ensure we find the best.

    # Fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        best_result = np.array(cells[:n])
        
    return best_result

# EVOLVE-BLOCK-END
