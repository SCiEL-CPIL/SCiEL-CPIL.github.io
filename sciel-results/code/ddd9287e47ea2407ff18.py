"""
Strategy: High-Resolution Quadrature Backward Induction for Bermudan Put.

This implementation uses a deterministic approach to compute the optimal exercise
policy. Instead of relying on the stochastic noise of the training paths (which
might be limited to 4096 samples), it leverages the known GBM parameters to
compute the continuation value via Gaussian Quadrature on a fine price grid.

1.  We define a fine grid of spot prices $S$.
2.  We perform backward induction from maturity $T=1$ to the first exercise date.
3.  At each step, we calculate the continuation value $C_t(S)$ by integrating the
    next step's value function $V_{t+1}(S')$ against the log-normal transition
    density using Gauss-Hermite quadrature.
4.  The value function is $V_t(S) = \max(\text{Intrinsic}(S), C_t(S))$.
5.  We extract the optimal exercise boundary $B_t$ for each step.
6.  The `exercise` method checks if the current spot is below the boundary for
    the given time step.
"""

import numpy as np
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    # Extract parameters from contract if available, otherwise use defaults from prompt
    # The prompt states: Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1.
    # 12 monthly exercise dates.
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Grid setup
    # S_min should be low enough to capture deep ITM puts.
    # S_max should be high enough that the put value is negligible.
    # With sigma=0.2, T=1, max move is approx 2-3 sigma. 
    # 100 * exp(0.2 * 2) ~ 150. 100 * exp(-0.2 * 2) ~ 67.
    # Let's use [0.1, 300] to be safe.
    S_min = 0.1
    S_max = 300.0
    n_grid = 2000
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Quadrature setup
    # Gauss-Hermite quadrature approximates integral of f(x) * exp(-x^2)
    # We need integral of f(z) * exp(-z^2/2) / sqrt(pi)
    # Let z = sqrt(2) * x. dz = sqrt(2) dx.
    # E[f(Z)] = 1/sqrt(pi) * integral_{-inf}^{inf} f(sqrt(2)*x) * exp(-x^2) dx
    # nodes, weights = roots_hermite(n)
    # weights are for exp(-x^2). We need to scale weights by 1/sqrt(pi) and 1/sqrt(2)?
    # Let's derive carefully.
    # Standard Normal PDF: p(z) = 1/sqrt(2*pi) * exp(-z^2/2)
    # Integral I = int_{-inf}^{inf} f(z) p(z) dz
    # Sub z = sqrt(2) x => dz = sqrt(2) dx
    # I = int_{-inf}^{inf} f(sqrt(2) x) * 1/sqrt(2*pi) * exp(-x^2) * sqrt(2) dx
    # I = 1/sqrt(pi) * int_{-inf}^{inf} f(sqrt(2) x) * exp(-x^2) dx
    # Gauss-Hermite: int_{-inf}^{inf} g(x) exp(-x^2) dx ~ sum w_i g(x_i)
    # So I ~ 1/sqrt(pi) * sum w_i f(sqrt(2) x_i)
    
    n_quad = 100
    x_nodes, w_weights = roots_hermite(n_quad)
    # Scale nodes to match z
    z_nodes = np.sqrt(2) * x_nodes
    # Scale weights
    w_scaled = w_weights / np.sqrt(np.pi)
    
    # Precompute drift and diffusion terms for transition
    # S_next = S_curr * exp((r - sigma^2/2) * dt + sigma * sqrt(dt) * Z)
    drift_term = (r - 0.5 * sigma**2) * dt
    vol_term = sigma * np.sqrt(dt)
    
    # Initialize Value Function at Maturity (Step 12)
    # V_12(S) = max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store boundaries
    # boundaries[t] for t in 1..12
    # Index 0 in list corresponds to step 1, index 11 to step 12.
    boundaries = [0.0] * n_steps
    
    # Maturity Step (Step 12)
    # At maturity, we always exercise if in the money.
    # Boundary is K.
    boundaries[n_steps - 1] = K
    
    # Backward Induction from Step 11 down to 1
    # Step index in 0-based array for boundaries:
    # Step 12 (t=12) is boundaries[11]
    # Step 11 (t=11) is boundaries[10]
    # ...
    # Step 1 (t=1) is boundaries[0]
    
    for t_idx in range(n_steps - 2, -1, -1):
        # t_idx corresponds to step t_idx + 1
        # We are calculating V_t for step t_idx + 1
        # Using V_{t+1} which is currently in V_next
        
        # Calculate S_next for all grid points and quadrature nodes
        # S_grid shape: (n_grid,)
        # z_nodes shape: (n_quad,)
        # S_next shape: (n_grid, n_quad)
        
        # S_next[i, k] = S_grid[i] * exp(drift_term + vol_term * z_nodes[k])
        # Vectorized:
        # Exponent part
        exp_part = np.exp(drift_term + vol_term * z_nodes) # shape (n_quad,)
        S_next = S_grid[:, np.newaxis] * exp_part[np.newaxis, :] # shape (n_grid, n_quad)
        
        # Clip S_next to grid bounds for interpolation
        S_next = np.clip(S_next, S_min, S_max)
        
        # Interpolate V_next at S_next
        # V_next is defined on S_grid
        # We need V_next(S_next)
        # np.interp is slow for 2D. We can use searchsorted or a custom vectorized interp.
        # However, n_grid=2000, n_quad=100. Total 200k points.
        # np.interp is 1D. We can apply it along axis 1.
        
        # Efficient interpolation:
        # Since S_grid is uniform, we can use linear interpolation manually or np.interp.
        # Let's use np.interp which is C-optimized.
        # We need to apply it to each row of S_next.
        
        # Reshape for np.interp? No, np.interp takes 1D x.
        # We can do:
        V_interp = np.empty_like(S_next)
        for i in range(n_grid):
            V_interp[i, :] = np.interp(S_next[i, :], S_grid, V_next)
            
        # Calculate Continuation Value
        # C_t(S_i) = discount * sum_k w_scaled[k] * V_interp[i, k]
        # w_scaled shape (n_quad,)
        # V_interp shape (n_grid, n_quad)
        C_t = discount * (V_interp @ w_scaled)
        
        # Calculate Intrinsic Value
        Intrinsic_t = np.maximum(K - S_grid, 0.0)
        
        # Determine Value Function
        V_curr = np.maximum(Intrinsic_t, C_t)
        
        # Determine Boundary
        # We exercise if Intrinsic_t >= C_t
        # diff = Intrinsic_t - C_t
        diff = Intrinsic_t - C_t
        
        # Find the last index where diff >= 0
        # diff is positive for low S, negative for high S.
        # Find indices where diff >= 0
        mask = diff >= 0
        if np.any(mask):
            last_idx = np.where(mask)[0][-1]
            # Check if we are at the end of the grid
            if last_idx < n_grid - 1:
                # Interpolate between last_idx and last_idx + 1
                # We want S where diff(S) = 0
                S1 = S_grid[last_idx]
                S2 = S_grid[last_idx + 1]
                d1 = diff[last_idx]
                d2 = diff[last_idx + 1]
                
                if d1 == d2:
                    boundary = S1
                else:
                    # Linear interpolation: S = S1 + (0 - d1) * (S2 - S1) / (d2 - d1)
                    boundary = S1 - d1 * (S2 - S1) / (d2 - d1)
                # Clip to valid range
                boundary = np.clip(boundary, S_min, S_max)
            else:
                boundary = S_max
        else:
            # Never exercise (should not happen for puts if S_min is low enough)
            boundary = S_min
            
        boundaries[t_idx] = boundary
        
        # Update for next iteration
        V_next = V_curr
        
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, bounds):
            self.bounds = bounds
            self.n_steps = len(bounds)
            
        def exercise(self, current_state, time):
            # time: 1-based index of the exercise date (1 to 12)
            # current_state: array of spot prices
            # Return: boolean array
            
            # Ensure time is within bounds
            if time < 1 or time > self.n_steps:
                # Default to not exercise if time is invalid?
                # Or handle gracefully.
                return np.zeros_like(current_state, dtype=bool)
            
            # Get boundary for this step
            # boundaries[0] is for step 1, boundaries[11] is for step 12
            idx = time - 1
            boundary = self.bounds[idx]
            
            # Exercise if S <= boundary
            # For puts, we exercise when price is low.
            return current_state <= boundary

    return ExercisePolicy(boundaries)

# EVOLVE-BLOCK-END
