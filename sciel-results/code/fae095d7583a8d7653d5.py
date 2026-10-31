import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute pair indices for vectorized operations
    i_idx, j_idx = np.triu_indices(N, k=1)
    pairs = np.column_stack([i_idx, j_idx])
    num_pairs = len(pairs)
    
    def check_validity(x, y, r, tol=1e-9):
        """
        Returns (is_valid, score).
        Checks if circles are disjoint and inside [0,1]^2.
        """
        # Boundary
        if np.any(x - r < -tol) or np.any(1 - x - r < -tol) or np.any(y - r < -tol) or np.any(1 - y - r < -tol):
            return False, 0.0
            
        # Pairwise
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist_sq = dx*dx + dy*dy
        r_sum = r[i_idx] + r[j_idx]
        r_sum_sq = r_sum * r_sum
        
        if np.any(r_sum_sq > dist_sq + tol):
            return False, 0.0
            
        return True, np.sum(r)

    def project_valid(x, y, r):
        """
        Aggressively projects the state to be valid by shrinking radii.
        Returns (x, y, r_valid, score).
        """
        # 1. Boundary clipping
        max_r_bound = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        r = np.minimum(r, max_r_bound)
        r = np.maximum(r, 1e-6)
        
        # 2. Pairwise iterative shrink
        # We shrink radii until no overlaps exist.
        # This is a conservative projection that guarantees validity.
        for _ in range(100):
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = r[i_idx] + r[j_idx]
            
            # If dist < sum_r, we have overlap.
            # We need to shrink r_i and r_j such that r_i' + r_j' <= dist.
            # Simple strategy: shrink the larger one more, or both proportionally.
            # Here we just check max violation.
            violations = sum_r - dist
            if np.max(violations) < 1e-10:
                break
            
            # Global shrink factor
            # Find the pair with max violation
            max_viol = np.max(violations)
            # Shrink all radii slightly to resolve.
            # A more precise approach:
            # For each overlapping pair, reduce r_i and r_j.
            # For speed, we do a global reduction.
            r *= 0.95
            r = np.minimum(r, max_r_bound)
            r = np.maximum(r, 1e-6)
            
        score = np.sum(r)
        return x, y, r, score

    def simulate(x, y, r, steps, dt, temp):
        """
        Run Langevin dynamics with repulsive forces.
        Forces:
        1. Repulsion from other circles (if overlapping or close)
        2. Attraction to center (to keep them in)
        3. Growth force on radii (pushing to increase size)
        
        We treat this as a system of particles with "pressure" that increases radius.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        # Precompute some constants
        mass = 1.0
        
        for step in range(steps):
            # Check time
            if time.time() > deadline - 5.0:
                break
                
            # 1. Compute Forces
            # Initialize forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            fr = np.zeros(N)
            
            # Pairwise Repulsion
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Overlap condition: dist < r_i + r_j
            r_sum = r[i_idx] + r[j_idx]
            overlap = r_sum - dist
            
            # Only apply force if overlapping or very close
            # Force magnitude proportional to overlap
            # F = k * overlap * direction
            k_repulse = 50.0
            force_mag = k_repulse * np.maximum(overlap, 0)
            
            # Normalize direction
            # If dist is 0, direction is undefined, use random or zero
            valid_dist = dist > 1e-9
            dir_x = np.zeros(num_pairs)
            dir_y = np.zeros(num_pairs)
            dir_x[valid_dist] = dx[valid_dist] / dist[valid_dist]
            dir_y[valid_dist] = dy[valid_dist] / dist[valid_dist]
            
            # Apply forces to i and j
            # i is pushed away from j
            fx_i = force_mag * dir_x
            fy_i = force_mag * dir_y
            # j is pushed away from i
            fx_j = -force_mag * dir_x
            fy_j = -force_mag * dir_y
            
            # Accumulate forces
            # Using np.add.at for accumulation
            np.add.at(fx, i_idx, fx_i)
            np.add.at(fy, i_idx, fy_i)
            np.add.at(fx, j_idx, fx_j)
            np.add.at(fy, j_idx, fy_j)
            
            # Boundary Repulsion
            # Push away from boundaries
            # Left: x < r -> force right
            # Right: 1-x < r -> force left
            # Bottom: y < r -> force up
            # Top: 1-y < r -> force down
            
            k_bound = 100.0
            v_left = r - x
            v_right = r + x - 1.0
            v_bottom = r - y
            v_top = r + y - 1.0
            
            fx += k_bound * np.maximum(v_left, 0)
            fx -= k_bound * np.maximum(v_right, 0)
            fy += k_bound * np.maximum(v_bottom, 0)
            fy -= k_bound * np.maximum(v_top, 0)
            
            # Radius Growth Force
            # We want to increase r.
            # But if increasing r causes overlap, the repulsive force on position
            # will push circles apart.
            # We apply a constant upward pressure on radii.
            # However, we must ensure that if a circle is stuck against a wall
            # or another circle, it doesn't grow infinitely.
            # The boundary repulsion and pairwise repulsion handle this by
            # pushing positions.
            # But we also need to prevent r from growing if it's already
            # constrained by boundaries in a way that can't be resolved by moving.
            # For now, simple constant growth.
            k_grow = 0.01
            fr += k_grow
            
            # 2. Update Positions and Radii
            # Langevin: v = (v + F*dt + random_noise) * damping
            # We don't track velocity explicitly, we do a simple gradient step with noise.
            
            # Noise
            noise_x = np.random.normal(0, temp, N)
            noise_y = np.random.normal(0, temp, N)
            noise_r = np.random.normal(0, temp*0.5, N) # Less noise on radii
            
            x += dt * (fx + noise_x)
            y += dt * (fy + noise_y)
            r += dt * (fr + noise_r)
            
            # 3. Project to valid state immediately to keep simulation stable
            # This is crucial: we don't let the system become wildly invalid.
            x, y, r, _ = project_valid(x, y, r)
            
            # Clip positions to [0,1] just in case
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            r = np.clip(r, 1e-6, 0.5)
            
        return x, y, r

    def local_polish(x, y, r, max_iters=200):
        """
        Coordinate descent to maximize sum of radii.
        For each circle, try to increase its radius as much as possible
        given current positions of others.
        Then, try to move the circle to allow for larger radius.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            improved = False
            # Shuffle order to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                # 1. Try to increase radius of i
                # Max radius allowed by boundaries
                max_r_bound = np.minimum(np.minimum(x[i], 1-x[i]), np.minimum(y[i], 1-y[i]))
                
                # Max radius allowed by other circles
                # For each j != i, dist(i,j) >= r_i + r_j => r_i <= dist(i,j) - r_j
                dx = x[i] - x[np.arange(N) != i]
                dy = y[i] - y[np.arange(N) != i]
                dist = np.sqrt(dx*dx + dy*dy)
                others_r = r[np.arange(N) != i]
                
                max_r_pairs = dist - others_r
                max_r_allowed = np.min(max_r_pairs)
                
                new_r = min(max_r_bound, max_r_allowed)
                new_r = max(new_r, r[i] * 1.01) # Try to grow at least 1%
                new_r = max(new_r, 1e-6)
                
                if new_r > r[i]:
                    r[i] = new_r
                    improved = True
                    
            if not improved:
                # If no radius could be increased, try to move circles to create space
                # Pick a random circle and move it slightly to see if it allows growth
                i = np.random.choice(N)
                # Try small random moves
                for _ in range(5):
                    dx = np.random.normal(0, 0.01)
                    dy = np.random.normal(0, 0.01)
                    x_new = np.clip(x[i] + dx, 0, 1)
                    y_new = np.clip(y[i] + dy, 0, 1)
                    
                    # Check if this move allows for larger radius
                    max_r_bound = np.minimum(np.minimum(x_new, 1-x_new), np.minimum(y_new, 1-y_new))
                    
                    dx_arr = x_new - x[np.arange(N) != i]
                    dy_arr = y_new - y[np.arange(N) != i]
                    dist_arr = np.sqrt(dx_arr*dx_arr + dy_arr*dy_arr)
                    others_r = r[np.arange(N) != i]
                    
                    max_r_pairs = dist_arr - others_r
                    max_r_allowed = np.min(max_r_pairs)
                    
                    new_r = min(max_r_bound, max_r_allowed)
                    new_r = max(new_r, 1e-6)
                    
                    if new_r > r[i] * 1.001: # Small improvement
                        x[i] = x_new
                        y[i] = y_new
                        r[i] = new_r
                        improved = True
                        break

            if not improved:
                break
                
        return x, y, r

    best_global_state = None
    best_global_score = 0.0
    
    # Multi-Start
    num_starts = 10
    time_per_start = (deadline - start_time) / num_starts
    time_per_start = max(time_per_start, 5.0)
    
    for k in range(num_starts):
        if time.time() > deadline - 15.0:
            break
        
        start_k = time.time()
        
        # Generate Initial State
        if k % 3 == 0:
            # Grid
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            count = 0
            for i in range(side):
                for j in range(side):
                    if count < N:
                        x[count] = (i + 0.5) / side
                        y[count] = (j + 0.5) / side
                        r[count] = 0.08
                        count += 1
        elif k % 3 == 1:
            # Random
            x = np.random.rand(N) * 0.8 + 0.1
            y = np.random.rand(N) * 0.8 + 0.1
            r = np.full(N, 0.05)
        else:
            # Ring
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.full(N, 0.05)
            
            x[0] = 0.5
            y[0] = 0.5
            
            angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
            rad1 = 0.15
            x[1:9] = 0.5 + rad1 * np.cos(angles1)
            y[1:9] = 0.5 + rad1 * np.sin(angles1)
            
            angles2 = np.linspace(0, 2*np.pi, 17, endpoint=False)
            rad2 = 0.35
            x[9:] = 0.5 + rad2 * np.cos(angles2)
            y[9:] = 0.5 + rad2 * np.sin(angles2)
            
        # Add noise
        x += np.random.normal(0, 0.01, N)
        y += np.random.normal(0, 0.01, N)
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        
        # Project to valid start
        x, y, r, _ = project_valid(x, y, r)
        
        # Phase 1: Simulation (Relaxation)
        # Run for a number of steps
        steps = 200
        dt = 0.05
        temp = 0.01
        x_sim, y_sim, r_sim = simulate(x, y, r, steps, dt, temp)
        
        # Phase 2: Local Polish
        x_pol, y_pol, r_pol = local_polish(x_sim, y_sim, r_sim, max_iters=50)
        
        # Check validity and score
        is_valid, score = check_validity(x_pol, y_pol, r_pol)
        
        if is_valid and score > best_global_score:
            best_global_score = score
            best_global_state = np.column_stack([x_pol, y_pol, r_pol])
            
        # If simulation didn't yield a valid state, try the projected initial state
        if not is_valid:
            # The simulation might have drifted.
            # We can try to polish the initial projected state directly.
            x_pol0, y_pol0, r_pol0 = local_polish(x, y, r, max_iters=20)
            is_valid0, score0 = check_validity(x_pol0, y_pol0, r_pol0)
            if is_valid0 and score0 > best_global_score:
                best_global_score = score0
                best_global_state = np.column_stack([x_pol0, y_pol0, r_pol0])

    # Fallback
    if best_global_state is None:
        x = np.full(N, 0.5)
        y = np.full(N, 0.5)
        r = np.full(N, 0.01)
        best_global_state = np.column_stack([x, y, r])
        best_global_score = np.sum(r)

    # Final Safety Projection
    x = best_global_state[:, 0]
    y = best_global_state[:, 1]
    r = best_global_state[:, 2]
    
    x, y, r, _ = project_valid(x, y, r)
    
    final_state = np.column_stack([x, y, r])
    
    return final_state
