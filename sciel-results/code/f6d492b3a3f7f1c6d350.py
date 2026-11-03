import numpy as np
import time

N = 200

def construct():
    # Set random seed for reproducibility if needed, but SA usually benefits from randomness
    # We will use a fixed seed for the "program" to be deterministic in evaluation?
    # The task says "score is value/seed". It doesn't specify if we should be deterministic.
    # Usually, for these challenges, deterministic is safer, but SA is stochastic.
    # We can run multiple restarts if time permits.
    
    def calculate_energy(P):
        # P is (N, 2)
        # Calculate pairwise distances
        # To avoid O(N^2) memory blowup for large N, but N=200 is fine (200x200 = 40k floats)
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        
        # Avoid division by zero
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        
        # Sum of inverse squared distances
        # E_pair = sum(1/d^2)
        E_pair = np.sum(1.0 / (dists**2))
        
        # Boundary energy
        # Distance to walls
        d_left = 0.5 - P[:, 0]
        d_right = 0.5 + P[:, 0]
        d_bottom = 0.5 - P[:, 1]
        d_top = 0.5 + P[:, 1]
        
        # Combine boundary distances
        # We want to penalize being close to walls
        # d_wall[i] = min(d_left, d_right, d_bottom, d_top)
        d_walls = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0)
        
        # If any point is outside, energy should be huge
        # But we constrain moves to stay inside? Or just high penalty?
        # High penalty is easier for SA.
        # If d_wall < 0, it's outside.
        # Let's use 1/d^2 for walls too, but handle d=0 or d<0.
        # If d <= 0, set to a very small positive number to create huge energy.
        d_walls_safe = np.maximum(d_walls, 1e-9)
        
        E_wall = np.sum(1.0 / (d_walls_safe**2))
        
        return E_pair + E_wall

    def get_radius(P):
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        d_left = 0.5 - P[:, 0]
        d_right = 0.5 + P[:, 0]
        d_bottom = 0.5 - P[:, 1]
        d_top = 0.5 + P[:, 1]
        d_walls = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0)
        min_wall = np.min(d_walls)
        
        return min(min_pair, min_wall) / 2.0

    # Initial State
    # Start with a random configuration, but slightly jittered grid to avoid immediate high energy overlaps
    # A loose grid is a good start
    m = 14 # approx sqrt(200)
    base = N // m
    rem = N % m
    row_counts = [base + 1] * rem + [base] * (m - rem)
    
    P = np.zeros((N, 2))
    idx = 0
    y_start = -0.4
    y_step = 0.8 / max(m - 1, 1)
    for j, n in enumerate(row_counts):
        y = y_start + j * y_step
        x_start = -0.4
        x_step = 0.8 / max(n - 1, 1) if n > 1 else 0
        for i in range(n):
            x = x_start + i * x_step
            P[idx, 0] = x
            P[idx, 1] = y
            idx += 1
            
    # Add small random noise to break symmetry
    P += np.random.randn(N, 2) * 0.01
    
    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.9995 # cooling rate
    max_steps = 200000 # Adjust based on time
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    current_E = calculate_energy(P)
    best_E = current_E
    best_P = P.copy()
    
    # Precompute indices for speed? No, N is small.
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random particle
        i = np.random.randint(N)
        
        # Propose move
        # Scale of move depends on T
        scale = T_start * (alpha ** step)
        # Ensure scale doesn't get too small too fast
        # Actually, let's use T directly as scale
        T_curr = T_start * (alpha ** step)
        
        delta = np.random.randn(2) * T_curr
        P_new = P.copy()
        P_new[i] = P[i] + delta
        
        # Optional: Keep within bounds to reduce wall energy calculation errors?
        # No, let energy function handle it.
        
        E_new = calculate_energy(P_new)
        
        if E_new < current_E:
            current_E = E_new
            P = P_new
            if E_new < best_E:
                best_E = E_new
                best_P = P.copy()
        else:
            # Accept with probability exp(-(E_new - current_E) / T_curr)
            # If T_curr is very small, this will be 0
            if T_curr > 1e-12:
                prob = np.exp(-(E_new - current_E) / T_curr)
                if np.random.rand() < prob:
                    current_E = E_new
                    P = P_new
                    if E_new < best_E:
                        best_E = E_new
                        best_P = P.copy()
                    
    return best_P
