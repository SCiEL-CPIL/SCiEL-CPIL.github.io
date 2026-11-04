import numpy as np
import random

N = 26

def construct():
    # Simulated Annealing Parameters
    T_init = 0.1
    T_end = 0.0001
    num_iterations = 50000
    cooling_rate = (T_end / T_init) ** (1.0 / num_iterations)
    
    # State: x, y, r
    # Shape: (N, 3)
    best_state = np.zeros((N, 3))
    current_state = np.zeros((N, 3))
    
    # Initialize randomly
    for i in range(N):
        current_state[i, 0] = np.random.uniform(0.1, 0.9)
        current_state[i, 1] = np.random.uniform(0.1, 0.9)
        current_state[i, 2] = np.random.uniform(0.03, 0.1)
        
    best_state = current_state.copy()
    best_score = -np.inf
    current_score = -np.inf
    
    def get_score(state):
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        
        score = np.sum(r)
        
        # Penalty for boundary
        # x must be in [r, 1-r]
        # y must be in [r, 1-r]
        boundary_penalty = 0
        for i in range(N):
            if x[i] < r[i]: boundary_penalty += (r[i] - x[i])**2
            elif x[i] > 1 - r[i]: boundary_penalty += (x[i] - (1 - r[i]))**2
            if y[i] < r[i]: boundary_penalty += (r[i] - y[i])**2
            elif y[i] > 1 - r[i]: boundary_penalty += (y[i] - (1 - r[i]))**2
            
        # Penalty for overlap
        overlap_penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    overlap_penalty += overlap*overlap
                    
        # Heavy penalty to ensure we find valid solutions
        return score - 50.0 * (boundary_penalty + overlap_penalty)

    current_score = get_score(current_state)
    best_score = current_score
    best_state = current_state.copy()
    
    T = T_init
    
    for it in range(num_iterations):
        T *= cooling_rate
        
        # Generate new state
        new_state = current_state.copy()
        idx = np.random.randint(N)
        
        # Perturb position
        new_state[idx, 0] += np.random.normal(0, 0.02)
        new_state[idx, 1] += np.random.normal(0, 0.02)
        
        # Perturb radius
        new_state[idx, 2] += np.random.normal(0, 0.005)
        
        # Clamp values to reasonable ranges
        new_state[:, 0] = np.clip(new_state[:, 0], 0.01, 0.99)
        new_state[:, 1] = np.clip(new_state[:, 1], 0.01, 0.99)
        new_state[:, 2] = np.clip(new_state[:, 2], 0.005, 0.4)
        
        new_score = get_score(new_state)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_state = new_state
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_state = current_state.copy()
                
    # Local Polishing (Gradient Ascent) on best_state
    # We try to increase radii and adjust positions to maintain validity
    # Simple local search: try small moves that increase sum(r) while keeping penalty 0
    
    polished_state = best_state.copy()
    polished_score = get_score(polished_state)
    
    # Check if current best is valid
    def is_valid(state):
        x = state[:, 0]
        y = state[:, 1]
        r = state[:, 2]
        for i in range(N):
            if x[i] < r[i] or x[i] > 1 - r[i] or y[i] < r[i] or y[i] > 1 - r[i]:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist*min_dist - 1e-9:
                    return False
        return True

    if is_valid(polished_state):
        # Try to increase radii slightly and fix positions
        for step in range(100):
            improved = False
            for i in range(N):
                # Try increasing radius
                old_r = polished_state[i, 2]
                polished_state[i, 2] = old_r * 1.01
                if is_valid(polished_state):
                    improved = True
                    polished_score = get_score(polished_state)
                else:
                    polished_state[i, 2] = old_r
                    
            if not improved:
                break
                
    # Final validation
    if is_valid(polished_state):
        return polished_state
    else:
        # Fallback to a simple grid if SA failed to produce valid output
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cx = (i + 0.5) / side
                    cy = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([cx, cy, r])
        return np.array(cells[:N])
