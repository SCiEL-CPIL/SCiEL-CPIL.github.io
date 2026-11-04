"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def construct():
    np.random.seed(42)
    
    def get_wh(ar):
        # w + h = 2
        # ar = w/h => w = ar*h => h(1+ar) = 2 => h = 2/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        return w, h

    def calculate_energy(state, ar):
        """
        state: (N, 3) array [x, y, r]
        ar: aspect ratio
        Returns: Energy to minimize (Penalty - SumRadii)
        """
        xs = state[:, 0]
        ys = state[:, 1]
        rs = state[:, 2]
        
        w, h = get_wh(ar)
        
        # Boundary Penalty
        # x must be in [r, w-r], y in [r, h-r]
        # If x < r, penalty (r-x)^2. If x > w-r, penalty (x-(w-r))^2.
        # This can be vectorized.
        
        # Left/Right
        pen_x = np.maximum(0, rs - xs) ** 2 + np.maximum(0, xs - (w - rs)) ** 2
        # Bottom/Top
        pen_y = np.maximum(0, rs - ys) ** 2 + np.maximum(0, ys - (h - rs)) ** 2
        
        boundary_pen = np.sum(pen_x) + np.sum(pen_y)
        
        # Overlap Penalty
        # Vectorized distance calculation
        # dist[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        min_dists = rs[:, None] + rs[None, :]
        
        # Overlap amount
        overlap = np.maximum(0, min_dists - dists)
        
        # We only count i < j to avoid double counting and self interaction (which is 0 anyway due to max(0, 0-0))
        # But overlap[i,i] is 0. So we can just sum the upper triangle or divide by 2.
        # Sum of squares of overlaps
        overlap_pen = 0.5 * np.sum(overlap**2)
        
        total_penalty = boundary_pen + overlap_pen
        
        sum_radii = np.sum(rs)
        
        # We want to maximize sum_radii, so minimize -sum_radii + penalty
        # Scaling penalty to ensure it dominates violations
        return -sum_radii + 100.0 * total_penalty

    def propose_move(state, ar, temp):
        """
        Propose a new state and ar.
        Returns new_state, new_ar
        """
        new_state = state.copy()
        new_ar = ar
        
        # Pick a random circle
        idx = np.random.randint(N)
        
        # Determine move type
        move_type = np.random.rand()
        
        if move_type < 0.4:
            # Move position
            noise = np.random.normal(0, 0.05 * temp)
            new_state[idx, 0] += noise
            new_state[idx, 1] += np.random.normal(0, 0.05 * temp)
        elif move_type < 0.7:
            # Move radius
            noise = np.random.normal(0, 0.02 * temp)
            new_state[idx, 2] += noise
            # Ensure radius is positive
            if new_state[idx, 2] < 0.01:
                new_state[idx, 2] = 0.01
        else:
            # Move aspect ratio
            # Perturb log(ar) to keep ar positive
            log_ar = np.log(ar)
            log_ar += np.random.normal(0, 0.1 * temp)
            new_ar = np.exp(log_ar)
            # Clamp ar to reasonable range [0.2, 5.0]
            new_ar = np.clip(new_ar, 0.2, 5.0)
            
        return new_state, new_ar

    def simulated_annealing(initial_state, initial_ar, max_iters=5000, start_temp=1.0, end_temp=1e-4):
        current_state = initial_state.copy()
        current_ar = initial_ar
        current_energy = calculate_energy(current_state, current_ar)
        
        best_state = current_state.copy()
        best_ar = current_ar
        best_energy = current_energy
        
        for i in range(max_iters):
            # Cooling schedule: exponential
            t_ratio = i / max_iters
            temp = start_temp * (end_temp / start_temp) ** t_ratio
            
            new_state, new_ar = propose_move(current_state, current_ar, temp)
            
            new_energy = calculate_energy(new_state, new_ar)
            
            delta_e = new_energy - current_energy
            
            # Acceptance criterion
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temp):
                current_state = new_state
                current_ar = new_ar
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_ar = current_ar
                    best_energy = current_energy
                    
        return best_state, best_ar, best_energy

    def is_valid(state, ar, tol=1e-6):
        xs = state[:, 0]
        ys = state[:, 1]
        rs = state[:, 2]
        w, h = get_wh(ar)
        
        # Check boundaries
        for i in range(N):
            if xs[i] < rs[i] - tol or xs[i] > w - rs[i] + tol:
                return False
            if ys[i] < rs[i] - tol or ys[i] > h - rs[i] + tol:
                return False
            if rs[i] <= 0:
                return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    # Initial Configuration: Grid
    def get_initial_grid(ar):
        w, h = get_wh(ar)
        # Approximate grid
        cols = int(np.sqrt(N * ar))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        state = np.zeros((N, 3))
        for i in range(N):
            r_idx = i // cols
            c_idx = i % cols
            x = (c_idx + 0.5) * w / cols
            y = (r_idx + 0.5) * h / rows
            # Initial radius estimate
            r = min(w/cols, h/rows) * 0.4
            state[i, 0] = x
            state[i, 1] = y
            state[i, 2] = r
        return state

    best_overall_state = None
    best_overall_ar = 1.0
    best_overall_sum = -np.inf

    # Run multiple restarts
    n_restarts = 20
    
    for k in range(n_restarts):
        # Randomize initial aspect ratio slightly
        base_ar = 1.0
        init_ar = base_ar * (0.8 + 0.4 * np.random.rand())
        
        init_state = get_initial_grid(init_ar)
        
        # Add some noise to init
        init_state[:, 0] += np.random.normal(0, 0.05, N)
        init_state[:, 1] += np.random.normal(0, 0.05, N)
        
        # Run SA
        # Time budget: 120s total. 20 restarts. ~6s each.
        # 5000 iterations per restart should be fast with vectorized energy.
        res_state, res_ar, res_energy = simulated_annealing(init_state, init_ar, max_iters=5000)
        
        # Check validity
        if is_valid(res_state, res_ar):
            sum_r = np.sum(res_state[:, 2])
            if sum_r > best_overall_sum:
                best_overall_sum = sum_r
                best_overall_state = res_state
                best_overall_ar = res_ar

    # Fallback
    if best_overall_state is None:
        # Return a valid simple grid
        ar = 1.0
        w, h = get_wh(ar)
        side = int(np.ceil(np.sqrt(N)))
        state = np.zeros((N, 3))
        for i in range(N):
            r_idx = i // side
            c_idx = i % side
            x = (c_idx + 0.5) * w / side
            y = (r_idx + 0.5) * h / side
            r = min(w/side, h/side) * 0.4
            state[i, 0] = x
            state[i, 1] = y
            state[i, 2] = r
        return state

    return best_overall_state

# EVOLVE-BLOCK-END
