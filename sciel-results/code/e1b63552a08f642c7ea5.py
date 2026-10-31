"""
Strategy: Simulated Annealing for circle packing.
We model the problem as minimizing an energy function:
E = (sum of overlap penalties) + (wall violation penalties) - (beta * sum of radii)
or more simply, we maximize sum(r_i) subject to constraints.
Using SA, we can explore the landscape more globally than gradient descent.

We use a fixed aspect ratio search or optimize aspect ratio as part of the state?
Optimizing W and H adds 2 dimensions. Total 2*21 + 1 (radius scale) + 2 (W,H) = 45 dims.
However, perimeter constraint W+H=2 means H = 2-W. So 1 dim for W.
Total dims: 42 (pos) + 1 (r_scale) + 1 (W) = 44.

Actually, it's safer to fix W and H in a loop and optimize positions + r_scale.
Or just optimize everything. Let's optimize positions and a single radius value R (all circles same radius).
Wait, the task says "maximize sum of their radii". It doesn't force them to be equal.
However, for 21 circles in a rectangle, optimal packings often have equal radii or a few distinct sizes.
To keep it simple and effective, let's assume all radii are equal (r_i = r).
This reduces the problem to finding the max r such that 21 circles of radius r fit.
This is a standard "packing equal circles" problem.
If we allow different radii, the search space is larger.
Given N=21, equal radius is a very strong baseline.
Let's stick to equal radii for the SA approach, as it reduces the problem to:
Maximize r subject to:
1. x_i in [r, W-r]
2. y_i in [r, H-r]
3. dist(i,j) >= 2r
4. W+H = 2

State: [x_0...x_20, y_0...y_20, r, W]
H is derived: H = 2 - W.
Dimensions: 42 + 1 + 1 = 44.

We will run SA with multiple restarts.
"""

import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0

    # Random seed for reproducibility in debugging, but random for diversity in evolution?
    # Usually, fixed seed is better for consistent scoring, but random can explore.
    # Let's use a fixed seed to ensure the "best" is found reliably, 
    # but vary the number of iterations based on time.
    
    rng = np.random.default_rng(42)

    # Define the energy function
    def energy(state, W, H):
        """
        state: [x0...xN-1, y0...yN-1, r]
        Returns a penalty value. Lower is better.
        We want to maximize r, so we can minimize -r + penalties.
        Or simply check validity and return 0 if valid, else penalty.
        For SA, a smooth penalty is better.
        
        Penalty = sum(max(0, (r_i + r_j - dist_ij)^2)) + sum(wall_violations^2)
        Objective to minimize: Penalty - lambda * r
        """
        xs = state[:N]
        ys = state[N:2*N]
        r = state[2*N]
        
        if r <= 0:
            return 1e6
        if W <= 0 or H <= 0:
            return 1e6
            
        # Wall penalties
        # We want x in [r, W-r]
        # Violation if x < r or x > W-r
        pen_x = np.sum(np.maximum(0, r - xs)**2) + np.sum(np.maximum(0, xs - (W - r))**2)
        pen_y = np.sum(np.maximum(0, r - ys)**2) + np.sum(np.maximum(0, ys - (H - r))**2)
        
        # Pairwise penalties
        # dist_ij >= 2r
        # Violation if dist_ij < 2r
        pen_pair = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx*dx + dy*dy
                min_d = 2*r
                if d2 < min_d*min_d:
                    d = np.sqrt(d2) if d2 > 1e-10 else 0
                    overlap = min_d - d
                    pen_pair += overlap * overlap
                    
        # We want to maximize r. 
        # Let's define the cost as: Penalty - r * large_weight
        # Or better, use a constraint satisfaction approach in SA:
        # If valid, cost = -r (so we minimize -r, i.e., maximize r)
        # If invalid, cost = -r + penalty * penalty_weight
        
        penalty_weight = 10.0
        total_penalty = pen_x + pen_y + pen_pair
        cost = total_penalty * penalty_weight - r
        
        return cost

    def check_validity(xs, ys, r, W, H):
        if r <= 0: return False
        if W <= 0 or H <= 0: return False
        # Walls
        if np.any(xs < r - 1e-9) or np.any(xs > W - r + 1e-9): return False
        if np.any(ys < r - 1e-9) or np.any(ys > H - r + 1e-9): return False
        # Pairs
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d2 = dx*dx + dy*dy
                min_d = 2*r
                if d2 < min_d*min_d - 1e-9:
                    return False
        return True

    def run_sa(W, H, init_state, max_iters, T_start, T_end):
        """
        Run SA for fixed W, H.
        init_state: [xs, ys, r]
        """
        state = init_state.copy()
        current_cost = energy(state, W, H)
        best_state = state.copy()
        best_cost = current_cost
        
        # SA parameters
        # Geometric cooling
        alpha = (T_end / T_start) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            T = T_start * (alpha ** i)
            if T < 1e-9:
                break
                
            # Perturb state
            new_state = state.copy()
            # Random walk in position and radius
            # Step size should be adaptive or fixed small
            step_pos = 0.05 * max(0.1, W) # Relative to box size
            step_r = 0.01 * max(0.1, W)
            
            # Pick random dimensions to perturb
            for _ in range(3): # Perturb 3 random coordinates
                dim = rng.integers(0, 2*N + 1)
                if dim < 2*N:
                    # Position
                    delta = rng.normal(0, step_pos)
                    new_state[dim] += delta
                else:
                    # Radius
                    delta = rng.normal(0, step_r)
                    new_state[dim] = max(0.01, new_state[dim] + delta)
            
            # Keep within bounds roughly to avoid huge penalties
            # But let the penalty handle it
            
            new_cost = energy(new_state, W, H)
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
                state = new_state
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_state = state.copy()
                    
        return best_state, best_cost

    # Search over aspect ratios
    # W + H = 2. W in [0.5, 1.5]
    # We'll try a set of W values
    W_values = np.linspace(0.6, 1.4, 17) # 17 values
    
    for W in W_values:
        if time.time() > deadline - 10:
            break
            
        H = 2.0 - W
        
        # Initialize state
        # Grid initialization
        cols = int(np.sqrt(N * W / H))
        if cols < 1: cols = 1
        rows = int(np.ceil(N / cols))
        
        xs = np.zeros(N)
        ys = np.zeros(N)
        for i in range(N):
            c = i % cols
            r_idx = i // cols
            xs[i] = (c + 0.5) * W / cols
            ys[i] = (r_idx + 0.5) * H / rows
            
        # Initial radius estimate
        # Area packing density ~0.9
        r_est = np.sqrt(0.9 * W * H / (N * np.pi))
        
        # State: [xs, ys, r]
        init_state = np.concatenate([xs, ys, [r_est]])
        
        # Run SA
        # Time budget per W: ~5s?
        # Total time 145s. 17 W values. ~8s each.
        # Let's allocate time dynamically.
        time_per_w = (deadline - time.time() - 5) / (len(W_values) - int(np.where(W_values == W)[0][0]))
        if time_per_w < 1.0:
            time_per_w = 1.0
            
        # Estimate iterations
        # Each iteration is ~210 pair checks. 210 ops. Very fast.
        # 10k iterations should take < 1s in Python? 
        # Let's benchmark roughly.
        # 10000 iterations * 210 pairs = 2.1M ops. In Python loop, maybe 0.5-1s.
        # Let's aim for 20000 iterations.
        
        max_iters = 20000
        T_start = 1.0
        T_end = 0.01
        
        best_state, best_cost = run_sa(W, H, init_state, max_iters, T_start, T_end)
        
        xs_best = best_state[:N]
        ys_best = best_state[N:2*N]
        r_best = best_state[2*N]
        
        # Verify and update global best
        if check_validity(xs_best, ys_best, r_best, W, H):
            sum_r = N * r_best
            if sum_r > best_r:
                best_r = sum_r
                best_config = np.column_stack((xs_best, ys_best, np.full(N, r_best)))
                best_W = W
                best_H = H
                # print(f"New best: r={r_best:.4f}, W={W:.2f}, H={H:.2f}, SumR={sum_r:.4f}")

    # If no valid config found, return a safe grid
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Final check
    xs = best_config[:, 0]
    ys = best_config[:, 1]
    rs = best_config[:, 2]
    
    # We already checked in loop, but double check just in case
    if check_validity(xs, ys, rs[0], best_W, best_H):
        return best_config
    else:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
