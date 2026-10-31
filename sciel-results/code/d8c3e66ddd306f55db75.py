import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Initial configuration: Grid
    side = int(np.ceil(np.sqrt(N)))
    # Create grid points
    xs = np.linspace(0.5/side, 1 - 0.5/side, side)
    ys = np.linspace(0.5/side, 1 - 0.5/side, side)
    
    # Flatten and take first N
    grid_xs, grid_ys = np.meshgrid(xs, ys)
    grid_xs = grid_xs.flatten()
    grid_ys = grid_ys.flatten()
    
    x = grid_xs[:N].copy()
    y = grid_ys[:N].copy()
    # Initial radii: small
    r = np.full(N, 0.5/side - 0.01)
    
    def get_energy(x, y, r, lambda_o=10.0, lambda_b=10.0):
        # Calculate sum of radii
        sum_r = np.sum(r)
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                viol = r[i] + r[j] - d
                if viol > 0:
                    overlap_pen += viol * viol
        
        # Boundary penalty
        bound_pen = 0.0
        for i in range(N):
            d_min = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            viol = r[i] - d_min
            if viol > 0:
                bound_pen += viol * viol
        
        return -sum_r + lambda_o * overlap_pen + lambda_b * bound_pen

    # Simulated Annealing
    T = 0.1
    T_min = 1e-5
    cooling = 0.995
    step_size = 0.05
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_energy = get_energy(x, y, r)
    
    current_energy = best_energy
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Pick a random circle to perturb
        idx = np.random.randint(N)
        
        # Save old values
        old_xi = x[idx]
        old_yi = y[idx]
        old_ri = r[idx]
        
        # Perturb
        x[idx] += np.random.normal(0, step_size)
        y[idx] += np.random.normal(0, step_size)
        r[idx] += np.random.normal(0, step_size * 0.5)
        
        # Clamp r to positive
        if r[idx] <= 0:
            r[idx] = 1e-6
        
        # Clamp x, y to [0, 1] to keep them in square (though penalty handles it, helps convergence)
        x[idx] = np.clip(x[idx], 0, 1)
        y[idx] = np.clip(y[idx], 0, 1)
        
        new_energy = get_energy(x, y, r)
        
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
                best_y = y.copy()
                best_r = r.copy()
        else:
            # Revert
            x[idx] = old_xi
            y[idx] = old_yi
            r[idx] = old_ri
            
        T *= cooling
        
        # Adaptive step size? Maybe reduce step size as T drops
        step_size = max(0.005, 0.05 * (T / 0.1))

    # Local Refinement: Maximize radii given positions
    # Iterate: for each i, set r[i] to max possible
    for _ in range(20):
        for i in range(N):
            # Max radius due to boundaries
            r_max_bound = min(x[i], 1.0 - x[i], y[i], 1.0 - y[i])
            
            # Max radius due to neighbors
            r_max_neigh = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                # d >= r[i] + r[j] => r[i] <= d - r[j]
                r_max_neigh = min(r_max_neigh, d - r[j])
            
            new_r = max(0, min(r_max_bound, r_max_neigh))
            best_r[i] = new_r
            
    # One more pass of position optimization using gradient of "min gap"
    # We want to increase the minimum of (d_ij - r_i - r_j) and (d_wall - r_i)
    # This is equivalent to minimizing the maximum violation or maximizing the minimum slack.
    # Simple heuristic: move circles apart if they are close.
    
    for _ in range(50):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # Calculate force
            fx = 0.0
            fy = 0.0
            
            # Boundary forces
            if best_x[i] < best_r[i]:
                fx += (best_r[i] - best_x[i])
            if 1.0 - best_x[i] < best_r[i]:
                fx -= (best_r[i] - (1.0 - best_x[i]))
            if best_y[i] < best_r[i]:
                fy += (best_r[i] - best_y[i])
            if 1.0 - best_y[i] < best_r[i]:
                fy -= (best_r[i] - (1.0 - best_y[i]))
                
            # Neighbor forces
            for j in range(N):
                if i == j:
                    continue
                dx = best_x[i] - best_x[j]
                dy = best_y[i] - best_y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-9:
                    # Random nudge
                    dx = np.random.rand() - 0.5
                    dy = np.random.rand() - 0.5
                    d = np.sqrt(dx*dx + dy*dy)
                    
                overlap = best_r[i] + best_r[j] - d
                if overlap > 0:
                    # Push i away from j
                    fx += (dx / d) * overlap
                    fy += (dy / d) * overlap
            
            # Apply move
            best_x[i] += 0.01 * fx
            best_y[i] += 0.01 * fy
            best_x[i] = np.clip(best_x[i], 0, 1)
            best_y[i] = np.clip(best_y[i], 0, 1)
            improved = True
            
            # Update radii for this circle immediately? No, do it in next pass.
            
        # Update radii
        for i in range(N):
            r_max_bound = min(best_x[i], 1.0 - best_x[i], best_y[i], 1.0 - best_y[i])
            r_max_neigh = np.inf
            for j in range(N):
                if i == j:
                    continue
                dx = best_x[i] - best_x[j]
                dy = best_y[i] - best_y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_max_neigh = min(r_max_neigh, d - best_r[j])
            best_r[i] = max(0, min(r_max_bound, r_max_neigh))

    # Final check: ensure validity
    # If invalid, shrink radii slightly
    for i in range(N):
        d_min = min(best_x[i], 1.0 - best_x[i], best_y[i], 1.0 - best_y[i])
        best_r[i] = min(best_r[i], d_min)
        
    for i in range(N):
        for j in range(i+1, N):
            dx = best_x[i] - best_x[j]
            dy = best_y[i] - best_y[j]
            d = np.sqrt(dx*dx + dy*dy)
            max_r_sum = d
            # If r[i] + r[j] > d, scale down
            if best_r[i] + best_r[j] > max_r_sum:
                scale = max_r_sum / (best_r[i] + best_r[j])
                best_r[i] *= scale
                best_r[j] *= scale

    result = np.column_stack((best_x, best_y, best_r))
    return result
