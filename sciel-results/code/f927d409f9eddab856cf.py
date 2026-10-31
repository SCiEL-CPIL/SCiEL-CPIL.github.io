"""
Strategy: Simulated Annealing for Flexible Circle Packing
Instead of rigid lattice structures, we use a stochastic optimization approach
(Simulated Annealing) to find a dense packing. 
1. Initialize with a valid random packing (e.g., grid-based with small jitter).
2. Define a cost function based on perimeter constraint violation and overlap.
   Actually, we want to MAXIMIZE sum of radii.
   We can reformulate this as: 
   Fix the rectangle perimeter to 4 (W+H=2).
   We want to maximize sum(r_i).
   This is equivalent to minimizing the "slack" in the packing.
   
   A more robust approach for SA:
   - Variables: positions (x, y) and radii (r) for all N circles, plus W, H.
   - Constraints:
     1. Non-overlap: dist(i, j) >= r_i + r_j
     2. Boundary: r_i <= x_i <= W - r_i, r_i <= y_i <= H - r_i
     3. Perimeter: 2(W + H) <= 4 -> W + H <= 2
   
   Objective: Maximize Sum(r_i).
   
   To handle constraints in SA, we use a penalty method or repair heuristics.
   A common effective heuristic for circle packing is:
   1. Keep radii fixed or slowly growing? No, radii are variables.
   2. Better: 
      - Start with small radii.
      - In each step, propose a move:
        a) Move a circle randomly.
        b) Change the radius of a circle slightly.
        c) Change W or H slightly.
      - Calculate "Energy" = Penalty(overlaps) + Penalty(boundary violations) - Lambda * Sum(radii).
      - Accept if Energy decreases or with probability exp(-delta_E / T).
      
   However, maximizing sum of radii with variable W, H is tricky because we can just make W, H huge and radii huge?
   No, W+H <= 2.
   
   Let's refine the objective:
   We want to maximize S = sum(r_i).
   Subject to:
   - Circles inside W x H
   - No overlaps
   - W + H <= 2
   
   This is a non-convex optimization.
   
   Alternative Formulation:
   For a fixed configuration of relative positions (topology), the optimal radii and W, H can be found?
   Not easily.
   
   Let's stick to direct SA on (x, y, r, W, H).
   Penalty function P:
   P = sum( max(0, (r_i + r_j) - dist(i,j))^2 )  # Overlap penalty
     + sum( max(0, r_i - x_i)^2 + max(0, x_i - (W - r_i))^2 + ... ) # Boundary penalty
     + max(0, (W + H - 2))^2 # Perimeter penalty
   
   Score = Sum(r_i) - mu * P.
   We want to maximize Score.
   
   To ensure validity at the end, we need P=0.
   If P > 0, the solution is invalid.
   We can run SA to minimize P first (finding a valid packing), then maximize Sum(r_i) while keeping P=0?
   Or combine them: Maximize Sum(r_i) - Large_Penalty * P.
   
   Let's use a two-phase approach or a strong penalty.
   Given the time limit (120s), a single SA run with careful tuning might be enough, or a few restarts.
   
   Key insight: The optimal packing for 21 circles in a rectangle likely has a specific structure.
   Random initialization + SA can find it if the temperature schedule is right.
   
   Implementation details:
   - N = 21
   - State: X (N, 2), R (N,), W, H
   - Init: Grid layout with small random jitter, small radii.
   - Move:
     - Pick a random circle i.
     - Propose delta_x, delta_y ~ N(0, sigma_pos)
     - Propose delta_r ~ N(0, sigma_r)
     - Optionally perturb W, H.
   - Energy Calculation:
     - Use cKDTree for fast overlap detection? Or just O(N^2) since N=21 is small.
     - O(N^2) is 210 pairs, very fast.
   
   Let's implement the SA.
"""

import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Constants
    PERIMETER_LIMIT = 4.0
    MAX_W_H_SUM = PERIMETER_LIMIT / 2.0  # W + H <= 2
    
    best_score = -np.inf
    best_config = None
    
    def calculate_energy(state, mu_over, mu_bound, mu_perim):
        """
        state: (x, y, r, W, H)
        Returns energy to be MINIMIZED.
        Energy = -Sum(r) + mu_over * OverlapPenalty + mu_bound * BoundaryPenalty + mu_perim * PerimPenalty
        """
        x, y, r, W, H = state
        
        # 1. Overlap Penalty
        overlap_pen = 0.0
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2
        
        # 2. Boundary Penalty
        bound_pen = 0.0
        for i in range(n):
            # x bounds
            if x[i] < r[i]:
                bound_pen += (r[i] - x[i])**2
            if x[i] > W - r[i]:
                bound_pen += (x[i] - (W - r[i]))**2
            # y bounds
            if y[i] < r[i]:
                bound_pen += (r[i] - y[i])**2
            if y[i] > H - r[i]:
                bound_pen += (y[i] - (H - r[i]))**2
        
        # 3. Perimeter Penalty
        perim_pen = 0.0
        if W + H > MAX_W_H_SUM:
            perim_pen = (W + H - MAX_W_H_SUM)**2
        
        # Objective: We want to maximize Sum(r).
        # So minimize: -Sum(r) + penalties
        score = -np.sum(r)
        energy = score + mu_over * overlap_pen + mu_bound * bound_pen + mu_perim * perim_pen
        
        return energy

    def simulate_annealing(init_state, t_start, t_end, steps, mu_over, mu_bound, mu_perim):
        x, y, r, W, H = init_state
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        current_energy = calculate_energy((x, y, r, W, H), mu_over, mu_bound, mu_perim)
        best_local_energy = current_energy
        best_local_state = (x.copy(), y.copy(), r.copy(), W, H)
        
        # Temperature schedule: Exponential cooling
        # T(t) = T_start * (T_end/T_start)^(t/steps)
        
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            t = t_start * (t_end / t_start) ** (step / steps)
            
            # Propose move
            # 1. Move a random circle
            i = random.randint(0, N - 1)
            
            # Random step size, scaled by temperature?
            # Use fixed step sizes for simplicity, or adapt.
            dx = random.gauss(0, 0.05)
            dy = random.gauss(0, 0.05)
            dr = random.gauss(0, 0.01)
            
            new_x = x[i] + dx
            new_y = y[i] + dy
            new_r = max(0.01, r[i] + dr) # Radii must be positive
            
            # Occasionally perturb W and H
            if random.random() < 0.1:
                dW = random.gauss(0, 0.02)
                dH = random.gauss(0, 0.02)
                new_W = max(0.1, W + dW)
                new_H = max(0.1, H + dH)
            else:
                new_W = W
                new_H = H
            
            # Calculate new energy
            new_energy = calculate_energy((x, y, r, new_W, new_H), mu_over, mu_bound, mu_perim)
            # Wait, I passed old x,y,r but new W,H. I need to pass the new state.
            # Let's create a temporary state for calculation to avoid modifying x,y,r yet?
            # Or just modify, calculate, and revert if rejected.
            
            old_x, old_y, old_r = x[i], y[i], r[i]
            x[i] = new_x
            y[i] = new_y
            r[i] = new_r
            old_W, old_H = W, H
            W, H = new_W, new_H
            
            new_energy = calculate_energy((x, y, r, W, H), mu_over, mu_bound, mu_perim)
            
            delta_e = new_energy - current_energy
            
            # Acceptance criterion
            if delta_e < 0 or random.random() < np.exp(-delta_e / t):
                current_energy = new_energy
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local_state = (x.copy(), y.copy(), r.copy(), W, H)
            else:
                # Revert
                x[i] = old_x
                y[i] = old_y
                r[i] = old_r
                W, H = old_W, old_H
        
        return best_local_state, best_local_energy

    # Initialization
    def init_state():
        # Start with a square grid in a 1x1 box (W=1, H=1, sum=2)
        # This is a valid starting point.
        W = 1.0
        H = 1.0
        
        # 5x5 grid = 25 spots. We need 21.
        # Use 5x4 grid = 20 spots + 1 extra? Or just 5x5 and remove 4?
        # Let's use a 5x5 grid, take first 21, and add jitter.
        # Or better, 4x5 grid (20 circles) + 1 in center?
        # Simplest: 5x5 grid, remove last 4, add small random jitter to break symmetry.
        
        xs = []
        ys = []
        rs = []
        
        # Grid spacing
        # If W=1, H=1, and we have 5 cols, 5 rows.
        # Spacing = 1/5 = 0.2.
        # Radius max = 0.1 (if touching).
        # Let's start with r = 0.05 (small, allows movement).
        
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                x = (i + 0.5) * (1.0 / 5.0)
                y = (j + 0.5) * (1.0 / 5.0)
                # Add jitter
                x += random.gauss(0, 0.01)
                y += random.gauss(0, 0.01)
                # Clip to box to ensure initial validity roughly
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                
                xs.append(x)
                ys.append(y)
                rs.append(0.05)
                count += 1
            if count >= N:
                break
        
        return np.array(xs), np.array(ys), np.array(rs), W, H

    # Run multiple SA attempts with different seeds/parameters
    # We have ~140s. Each SA step is fast (O(N^2) ~ 200 ops).
    # 100,000 steps should take < 1s in Python? Maybe 1-2s.
    # So we can do many restarts.
    
    random.seed(42)
    np.random.seed(42)
    
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        init = init_state()
        
        # Parameters
        # We want to maximize sum(r).
        # Penalties must be strong enough to enforce constraints.
        # mu_over, mu_bound, mu_perim
        # As we progress, we might want to reduce penalties to allow "illegal" states to escape local minima?
        # No, better to keep them strong and let SA explore.
        
        mu_over = 10.0
        mu_bound = 10.0
        mu_perim = 10.0
        
        t_start = 1.0
        t_end = 0.01
        steps = 20000  # Adjust based on speed
        
        state, energy = simulate_annealing(init, t_start, t_end, steps, mu_over, mu_bound, mu_perim)
        
        x, y, r, W, H = state
        
        # Check validity
        # If valid, calculate raw score
        valid = True
        # Boundary
        for i in range(N):
            if x[i] < r[i] - 1e-5 or x[i] > W - r[i] + 1e-5 or y[i] < r[i] - 1e-5 or y[i] > H - r[i] + 1e-5:
                valid = False
                break
            if W + H > MAX_W_H_SUM + 1e-5:
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if dist < r[i] + r[j] - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                # Store result
                config = np.zeros((N, 3))
                config[:, 0] = x
                config[:, 1] = y
                config[:, 2] = r
                best_config = config
                
                # If we found a very good score, we might want to refine it
                # But for now, just keep the best valid one.
                
    if best_config is not None:
        return best_config
    
    # Fallback: Return a valid but suboptimal packing
    # Grid packing
    W = 1.0
    H = 1.0
    config = np.zeros((N, 3))
    count = 0
    for i in range(5):
        for j in range(5):
            if count >= N:
                break
            x = (i + 0.5) * 0.2
            y = (j + 0.5) * 0.2
            r = 0.1
            config[count] = [x, y, r]
            count += 1
        if count >= N:
            break
            
    return config

# EVOLVE-BLOCK-END
