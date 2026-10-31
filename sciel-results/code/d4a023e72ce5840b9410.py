import numpy as np
import time

N = 26

def construct():
    """
    Packs 26 circles in a unit square maximizing the sum of radii.
    Uses Simulated Annealing with a penalty-based objective function.
    """
    
    def calculate_energy(state):
        """
        state: array of shape (N, 3) -> [x, y, r]
        Returns: energy (lower is better).
        Energy = -sum(r) + penalty * (constraint_violations)
        """
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        # Objective: Minimize -sum(r)
        obj = -np.sum(r)
        
        # Constraints
        # 1. Boundaries
        # x >= r, x <= 1-r
        # y >= r, y <= 1-r
        pen_x_low = np.maximum(0, r - x)
        pen_x_high = np.maximum(0, x - (1 - r))
        pen_y_low = np.maximum(0, r - y)
        pen_y_high = np.maximum(0, y - (1 - r))
        
        boundary_violation = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                             np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
        
        # 2. Overlaps
        # dist(i,j) >= r_i + r_j
        # Use vectorization for speed
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Overlap amount: r_sum - dist if positive
        overlap = np.maximum(0, r_sum - dist)
        # Zero out diagonal
        np.fill_diagonal(overlap, 0)
        
        overlap_violation = np.sum(overlap**2)
        
        # Total penalty
        penalty_weight = 1000.0
        total_penalty = penalty_weight * (boundary_violation + overlap_violation)
        
        return obj + total_penalty

    def is_valid(state, tol=1e-8):
        """
        Strictly checks if the state is valid.
        """
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        # Check boundaries
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        
        # Check overlaps
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # If any distance is less than sum of radii (minus tolerance), invalid
        if np.any(dist < r_sum - tol):
            return False
            
        return True

    best_sum = -1
    best_state = None
    
    # Simulated Annealing Parameters
    start_temp = 1.0
    end_temp = 0.001
    cooling_rate = 0.999
    max_iter = 5000
    
    # Run multiple trials
    for trial in range(20):
        # Initialization: Random positions, small radii
        # Start with a grid-like randomization to avoid immediate heavy overlaps
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        
        state = np.column_stack((x, y, r))
        
        current_energy = calculate_energy(state)
        best_trial_energy = current_energy
        best_trial_state = state.copy()
        
        temp = start_temp
        
        for i in range(max_iter):
            # Perturb state
            # Move positions slightly
            delta_x = np.random.normal(0, 0.01, N)
            delta_y = np.random.normal(0, 0.01, N)
            delta_r = np.random.normal(0, 0.01, N)
            
            new_state = state.copy()
            new_state[:, 0] += delta_x
            new_state[:, 1] += delta_y
            new_state[:, 2] += delta_r
            
            # Enforce basic bounds on radii to keep them positive and reasonable
            new_state[:, 2] = np.clip(new_state[:, 2], 0.01, 0.5)
            # Enforce basic bounds on positions to keep them in [0,1] roughly
            new_state[:, 0] = np.clip(new_state[:, 0], 0.0, 1.0)
            new_state[:, 1] = np.clip(new_state[:, 1], 0.0, 1.0)
            
            new_energy = calculate_energy(new_state)
            
            # Acceptance criteria
            delta_e = new_energy - current_energy
            
            if delta_e < 0:
                # Better solution
                state = new_state
                current_energy = new_energy
                if current_energy < best_trial_energy:
                    best_trial_energy = current_energy
                    best_trial_state = state.copy()
            else:
                # Worse solution, accept with probability exp(-delta_e / temp)
                if np.random.random() < np.exp(-delta_e / temp):
                    state = new_state
                    current_energy = new_energy
            
            # Cool down
            temp *= cooling_rate
            
            if temp < end_temp:
                break
        
        # After SA, check validity of the best state found in this trial
        if is_valid(best_trial_state):
            current_sum = np.sum(best_trial_state[:, 2])
            if current_sum > best_sum:
                best_sum = current_sum
                best_state = best_trial_state.copy()
                
        # If the best state from SA is invalid, we might try to repair it or just discard
        # In this simple version, we discard invalid results.
        
    if best_state is None:
        # Fallback: Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    # Max radius in cell is 0.5/side, but we must fit in square
                    # Actually, for a grid of 5x5, radius is 0.1.
                    # Let's use a safe radius.
                    r = 0.5 / side - 0.01
                    cells.append([cx, cy, r])
        return np.array(cells)
        
    return best_state
