# EVOLVE-BLOCK-START
import numpy as np
import random

def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Uses a constructive heuristic followed by Simulated Annealing.
    """
    n = 10
    tol = 1e-7
    
    # Objective function: minimize negative sum of sides + penalties
    def objective(params):
        # params: [s0, cx0, cy0, s1, cx1, cy1, ..., s9, cx9, cy9]
        s = params[0::3]
        cx = params[1::3]
        cy = params[2::3]
        
        penalty = 0.0
        # Boundary penalty
        for i in range(n):
            if s[i] <= 0:
                penalty += 1e6
            # Check boundaries
            if cx[i] - s[i]/2 < -tol:
                penalty += 1e6 * (abs(cx[i] - s[i]/2 + tol) + 1)
            if cx[i] + s[i]/2 > 1 + tol:
                penalty += 1e6 * (abs(cx[i] + s[i]/2 - 1 - tol) + 1)
            if cy[i] - s[i]/2 < -tol:
                penalty += 1e6 * (abs(cy[i] - s[i]/2 + tol) + 1)
            if cy[i] + s[i]/2 > 1 + tol:
                penalty += 1e6 * (abs(cy[i] + s[i]/2 - 1 - tol) + 1)
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    overlap_x = max(0, t - dx)
                    overlap_y = max(0, t - dy)
                    penalty += 1e6 * (overlap_x * overlap_y + overlap_x + overlap_y)
        
        return -np.sum(s) + penalty

    def project_to_feasible(params):
        """
        Attempts to move squares to be non-overlapping and within bounds.
        Returns None if it fails (should rarely happen with small moves).
        """
        s = params[0::3].copy()
        cx = params[1::3].copy()
        cy = params[2::3].copy()
        
        # Clip to bounds
        for i in range(n):
            s[i] = max(s[i], 1e-4)
            half = s[i] / 2
            cx[i] = np.clip(cx[i], half, 1.0 - half)
            cy[i] = np.clip(cy[i], half, 1.0 - half)
            
        # Simple iterative separation
        # This is a basic heuristic, SA will refine
        return np.concatenate([s, cx, cy])

    def greedy_init():
        """
        Construct a valid packing using a greedy approach.
        We place squares in decreasing order of size.
        """
        # Initial sizes: 1 large, 4 medium, 5 small?
        # Let's try a configuration that is known to pack well.
        # 1 square of 0.5, 4 of 0.25, 5 of ~0.15?
        # Sum = 0.5 + 1.0 + 0.75 = 2.25.
        # Can we do better?
        # 2 of 0.4 (0.8), 4 of 0.3 (1.2), 4 of 0.2 (0.8) -> 2.8? 
        # 0.4*2 = 0.8 area. 0.3*4 = 0.36 area. 0.2*4 = 0.16 area. Total 1.32 > 1. Impossible.
        
        # Let's use a standard grid-like init to ensure validity.
        # 3x3 grid with one extra?
        # 10 squares.
        # Sizes: [0.33]*3 + [0.33]*3 + [0.33]*3 + [0.33] ? No, 0.33*10 = 3.3. Area 1.089 > 1.
        
        # Max area is 1. Sum of sides S. S^2/10 <= 1 => S <= sqrt(10) ~ 3.16.
        # But packing efficiency is lower.
        
        # Let's start with a valid grid.
        # 3 columns, 3 rows = 9 squares of size 1/3. Sum = 3.0.
        # 10th square? We can shrink the 9 to make room.
        
        # Better: Start with 1 large (0.5), 4 medium (0.25), 5 small (0.15).
        # 0.5 + 4*0.25 + 5*0.15 = 0.5 + 1.0 + 0.75 = 2.25.
        # This is valid.
        
        s = np.zeros(n)
        cx = np.zeros(n)
        cy = np.zeros(n)
        
        # 1 Large
        s[0] = 0.5
        cx[0] = 0.25
        cy[0] = 0.25
        
        # 4 Medium (around the large one? or in the remaining space?)
        # Remaining space is an L-shape of width 0.5.
        # Place 4 medium in the top-right 0.5x0.5 area? No, 0.25*2 = 0.5.
        # So 2x2 grid of 0.25 squares in top-right.
        s[1] = 0.25; cx[1] = 0.625; cy[1] = 0.625
        s[2] = 0.25; cx[2] = 0.875; cy[2] = 0.625
        s[3] = 0.25; cx[3] = 0.625; cy[3] = 0.875
        s[4] = 0.25; cx[4] = 0.875; cy[4] = 0.875
        
        # 5 Small. Where?
        # Bottom right 0.5x0.5 area is occupied by nothing?
        # Large is [0,0.5]x[0,0.5].
        # Mediums are [0.5,1]x[0.5,1].
        # Free space: [0.5,1]x[0,0.5] (0.5x0.5) and [0,0.5]x[0.5,1] (0.5x0.5).
        # We can fit 5 small squares of size 0.15?
        # 0.5 / 0.15 = 3.33. So we can fit 3 in a row.
        # Let's place them in the bottom-right 0.5x0.5 block.
        # 3 in bottom row, 2 in top row?
        # Size 0.15.
        # Row 1 (y=0.075): x=0.575, 0.725, 0.875
        # Row 2 (y=0.225): x=0.575, 0.725
        
        s[5] = 0.15; cx[5] = 0.575; cy[5] = 0.075
        s[6] = 0.15; cx[6] = 0.725; cy[6] = 0.075
        s[7] = 0.15; cx[7] = 0.875; cy[7] = 0.075
        s[8] = 0.15; cx[8] = 0.575; cy[8] = 0.225
        s[9] = 0.15; cx[9] = 0.725; cy[9] = 0.225
        
        # Check validity
        # Large: [0,0.5]x[0,0.5]
        # Mediums: [0.5,1]x[0.5,1]
        # Smalls: in [0.5,1]x[0,0.5]
        # No overlaps. All inside.
        
        return np.concatenate([s, cx, cy])

    # Simulated Annealing
    def simulate_annealing(x0, t_start=1.0, t_end=1e-5, steps=20000):
        x = x0.copy()
        current_cost = objective(x)
        best_x = x.copy()
        best_cost = current_cost
        
        for i in range(steps):
            t = t_start * (t_end / t_start) ** (i / steps)
            
            # Perturb
            x_new = x.copy()
            # Randomly pick a square to move
            idx = np.random.randint(n)
            
            # Change size slightly
            if np.random.random() < 0.5:
                delta_s = np.random.uniform(-0.05, 0.05)
                x_new[idx*3] = max(1e-4, x_new[idx*3] + delta_s)
            
            # Move center
            delta_x = np.random.uniform(-0.1, 0.1)
            delta_y = np.random.uniform(-0.1, 0.1)
            x_new[idx*3+1] += delta_x
            x_new[idx*3+2] += delta_y
            
            # Project to bounds
            s_new = x_new[idx*3]
            cx_new = x_new[idx*3+1]
            cy_new = x_new[idx*3+2]
            half = s_new / 2
            cx_new = np.clip(cx_new, half, 1.0 - half)
            cy_new = np.clip(cy_new, half, 1.0 - half)
            x_new[idx*3+1] = cx_new
            x_new[idx*3+2] = cy_new
            
            new_cost = objective(x_new)
            
            # Acceptance probability
            if new_cost < current_cost:
                x = x_new
                current_cost = new_cost
                if current_cost < best_cost:
                    best_x = x_new
                    best_cost = current_cost
            else:
                p = np.exp(-(new_cost - current_cost) / t)
                if np.random.random() < p:
                    x = x_new
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_x = x_new
                        best_cost = current_cost
                        
        return best_x

    # Run
    x0 = greedy_init()
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Run SA
    result_params = simulate_annealing(x0, steps=30000)
    
    s = result_params[0::3]
    cx = result_params[1::3]
    cy = result_params[2::3]
    
    # Final cleanup
    for i in range(n):
        s[i] = max(s[i], 1e-4)
        half = s[i] / 2
        cx[i] = np.clip(cx[i], half, 1.0 - half)
        cy[i] = np.clip(cy[i], half, 1.0 - half)
        
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
