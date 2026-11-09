# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def squares_in_square10() -> tuple:
    """
    Place 10 squares in the unit square to maximize the sum of their side lengths.
    
    Strategy:
    Use Simulated Annealing starting from the known analytic optimum (1 large + 9 small grid).
    This allows the optimizer to explore local deformations that might yield a higher sum.
    """
    n = 10
    
    def evaluate(state):
        x = state[:n]
        y = state[n:2*n]
        s = state[2*n:3*n]
        
        penalty = 0.0
        for i in range(n):
            if x[i] < s[i]/2.0:
                penalty += (s[i]/2.0 - x[i])
            elif x[i] > 1.0 - s[i]/2.0:
                penalty += (x[i] - (1.0 - s[i]/2.0))
            if y[i] < s[i]/2.0:
                penalty += (s[i]/2.0 - y[i])
            elif y[i] > 1.0 - s[i]/2.0:
                penalty += (y[i] - (1.0 - s[i]/2.0))
            if s[i] < 1e-9:
                penalty += (1e-9 - s[i])
        
        for i in range(n):
            for j in range(i+1, n):
                dx = (s[i] + s[j]) / 2.0 - abs(x[i] - x[j])
                dy = (s[i] + s[j]) / 2.0 - abs(y[i] - y[j])
                if dx > 0 and dy > 0:
                    penalty += min(dx, dy)
        
        score = np.sum(s)
        return score - penalty

    # Initialize with the analytic solution: 9 small (s=1/3) in 3x3 grid, 1 large (L=1-3s)
    # To avoid floating point issues with exact 1/3, use a slightly smaller s.
    s_small = 0.333333
    L_large = 1.0 - 3.0 * s_small
    
    x = np.zeros(n)
    y = np.zeros(n)
    s = np.zeros(n)
    
    for i in range(9):
        row = i // 3
        col = i % 3
        x[i] = (col + 0.5) * s_small
        y[i] = (row + 0.5) * s_small
        s[i] = s_small
        
    x[9] = 1.0 - L_large / 2.0
    y[9] = 1.0 - L_large / 2.0
    s[9] = L_large
    
    initial_state = np.concatenate([x, y, s])
    current_state = initial_state.copy()
    current_score = evaluate(current_state)
    best_state = current_state.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 0.1
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iterations = 10000
    
    rng = np.random.default_rng(42)
    
    for _ in range(max_iterations):
        if T < T_min:
            break
            
        new_state = current_state.copy()
        i = rng.integers(0, n)
        var_type = rng.integers(0, 3)
        
        # Scale step size with T? Or fixed?
        # Fixed small step to stay in the neighborhood of the good solution.
        step = 0.01
        if var_type == 0:
            new_state[i] += rng.normal(0, step)
        elif var_type == 1:
            new_state[n + i] += rng.normal(0, step)
        else:
            new_state[2*n + i] += rng.normal(0, step)
            
        new_state[i] = np.clip(new_state[i], 0, 1)
        new_state[n + i] = np.clip(new_state[n + i], 0, 1)
        new_state[2*n + i] = np.clip(new_state[2*n + i], 0.01, 1.0)
        
        new_score = evaluate(new_state)
        
        delta = new_score - current_score
        if delta > 0:
            current_state = new_state
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_state = current_state.copy()
        else:
            prob = math.exp(delta / T)
            if rng.random() < prob:
                current_state = new_state
                current_score = new_score
                
        T *= cooling_rate
        
    # Final Hill Climbing
    current_state = best_state.copy()
    current_score = best_score
    for _ in range(2000):
        improved = False
        for i in range(n):
            for var_type in range(3):
                for step in [0.005, -0.005, 0.001, -0.001]:
                    trial_state = current_state.copy()
                    if var_type == 0:
                        trial_state[i] += step
                    elif var_type == 1:
                        trial_state[n + i] += step
                    else:
                        trial_state[2*n + i] += step
                        
                    trial_state[i] = np.clip(trial_state[i], 0, 1)
                    trial_state[n + i] = np.clip(trial_state[n + i], 0, 1)
                    trial_state[2*n + i] = np.clip(trial_state[2*n + i], 0.01, 1.0)
                    
                    trial_score = evaluate(trial_state)
                    if trial_score > current_score:
                        current_state = trial_state
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
            
    x = best_state[:n]
    y = best_state[n:2*n]
    s = best_state[2*n:3*n]
    
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        centers[i] = [x[i], y[i]]
        sides[i] = s[i]
        
    return centers, sides

# EVOLVE-BLOCK-END
