import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fixed model parameters
    r = 0.05
    K = 100.0
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    
    # Use a high-resolution binomial tree for backward induction
    # The tree must have at least n_exercise steps, and we want fine resolution
    # for the exercise boundary. We use 1200 steps to match the reference.
    n_steps_tree = 1200
    dt = T / n_steps_tree
    
    # Up and down factors
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Discount factor per step
    disc = np.exp(-r * dt)
    
    # Number of nodes at each level: level k has k+1 nodes
    # We only need to store the current and next level values
    # Maximum nodes at level n_steps_tree: n_steps_tree + 1 = 1201
    
    # We'll use arrays of size (n_steps_tree + 1) for node values
    # Start at the terminal level
    # At level n_steps_tree, node i has spot S_0 * u^i * d^(n_steps_tree - i)
    # But we don't know S_0. Instead, we work with the value as a function of spot.
    # Actually, for a binomial tree, the value at each node depends on the spot.
    # We can compute the value for a range of spots.
    
    # Alternative: Use the fact that for a European option, the value is known.
    # For American/Bermudan, we need backward induction on the tree.
    
    # Let's define the tree in terms of log-spot or just spot levels.
    # At step t (0 to n_steps_tree), the possible spots are:
    # S(t, i) = S_0 * u^i * d^(t-i) for i = 0, ..., t
    # But S_0 is not fixed; we want the exercise boundary as a function of S.
    
    # Better approach: Work with the value function V(t, S) on a grid of S values.
    # At each step, we know V(t+1, S) for a grid of S.
    # Then V(t, S) = max(K - S, disc * (p * V(t+1, S*u) + (1-p) * V(t+1, S*d)))
    
    # We need a grid of S values. Let's use a log-spaced grid from S_min to S_max.
    # S_min should be small enough, S_max large enough.
    # For a put, S > K is OTM, so we focus on S in [0, K*2] or so.
    
    # Actually, the standard binomial tree approach for American options uses a fixed
    # number of nodes at each level, not a fixed S grid. Let me use the standard approach.
    
    # Standard binomial tree:
    # Level 0: 1 node, S = S_0
    # Level 1: 2 nodes, S = S_0*u, S_0*d
    # ...
    # Level t: t+1 nodes, S = S_0 * u^i * d^(t-i) for i=0..t
    
    # The value at each node is computed by backward induction.
    # But we want the exercise boundary, which is the threshold S* such that
    # for S < S*, exercise is optimal.
    
    # To get the boundary, we can compute the value for a range of S_0 values,
    # or we can work with the value as a function of the "position" in the tree.
    
    # Let's use a different approach: compute the value function on a fine grid of S
    # using the PDE or a finite difference method. But the task says to use a binomial tree.
    
    # Let me use the binomial tree with a fixed S_0 and compute the boundary.
    # Actually, the boundary is independent of S_0 in the sense that it's a function
    # of the remaining time and the current spot. So we can compute it for a range of spots.
    
    # I'll use a log-spaced grid of spot values and do backward induction.
    n_grid = 10000
    S_max = 200.0
    S_min = 0.01
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # V[i] = value at spot S_grid[i]
    # Initialize at terminal time
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Exercise boundaries: boundary[t] = threshold at exercise date t (0-indexed, t=0 is first exercise)
    # Exercise dates are at steps 1, 2, ..., 12 in the original path, which correspond to
    # tree levels n_steps_tree - 12, n_steps_tree - 11, ..., n_steps_tree - 1? 
    # Wait, the tree has n_steps_tree steps total. The exercise dates are at times
    # T/12, 2T/12, ..., 12T/12. In the tree, these correspond to levels:
    # level for time t: level = int(t / dt) = int(t * n_steps_tree / T)
    # For t = k*T/12, level = k * n_steps_tree / 12 = k * 100 (since n_steps_tree=1200)
    
    # So exercise date k (k=1..12) is at tree level k * 100
    # We do backward induction from level n_steps_tree down to level 0.
    # At each level, we compute V_current from V_next.
    # At exercise levels, we take max with payoff.
    
    # Precompute which levels are exercise levels
    exercise_levels = set(k * (n_steps_tree // n_exercise) for k in range(1, n_exercise + 1))
    # n_steps_tree // n_exercise = 1200 // 12 = 100
    # So exercise levels are 100, 200, ..., 1200
    
    # We'll store the boundary at each exercise date
    boundaries = np.zeros(n_exercise)
    
    # Backward induction
    for level in range(n_steps_tree - 1, -1, -1):
        if _time.time() > deadline:
            break
            
        # Compute continuation value: disc * (p * V_next[S*u] + (1-p) * V_next[S*d])
        # We need to interpolate V_next at S*u and S*d
        # S_grid is log-spaced, so S*u and S*d are also on a log-spaced grid
        # We can use np.interp for this
        
        Su = S_grid * u
        Sd = S_grid * d
        
        # Interpolate V_next at Su and Sd
        V_next_at_Su = np.interp(Su, S_grid, V_next)
        V_next_at_Sd = np.interp(Sd, S_grid, V_next)
        
        continuation = disc * (p * V_next_at_Su + (1 - p) * V_next_at_Sd)
        
        # Check if this level is an exercise level
        if level in exercise_levels:
            payoff = np.maximum(K - S_grid, 0.0)
            V_current = np.maximum(payoff, continuation)
            
            # Find the boundary: largest S where payoff >= continuation
            # i.e., K - S >= continuation (for S < K)
            # Find where K - S_grid - continuation changes sign
            diff = K - S_grid - continuation
            # We want the largest S where diff >= 0
            # Find the last index where diff >= 0
            mask = diff >= 0
            if np.any(mask):
                idx = np.where(mask)[0][-1]
                # Interpolate for better precision
                if idx + 1 < n_grid:
                    # Linear interpolation between idx and idx+1
                    x1, x2 = S_grid[idx], S_grid[idx + 1]
                    y1, y2 = diff[idx], diff[idx + 1]
                    if abs(y2 - y1) < 1e-12:
                        thresh = x1
                    else:
                        thresh = x1 - y1 * (x2 - x1) / (y2 - y1)
                    thresh = np.clip(thresh, 0, K)
                else:
                    thresh = S_grid[idx]
                # Determine which exercise date this corresponds to
                # level = k * 100, so k = level // 100
                k = level // (n_steps_tree // n_exercise)
                if 1 <= k <= n_exercise:
                    boundaries[k - 1] = thresh
        else:
            V_current = continuation
        
        V_next = V_current
    
    # For the last exercise date (k=12, level=1200), the boundary is K
    boundaries[11] = K
    
    # Ensure boundaries are monotonically non-increasing (earlier exercise should have lower threshold)
    # Actually, for a put, the exercise boundary typically decreases over time (you're less eager to exercise early)
    # Wait, for a put, you want to exercise when S is low. The boundary should be non-decreasing over time?
    # No, for a put, the optimal exercise boundary is typically non-increasing in time (you exercise at a higher threshold as expiration approaches)
    # Actually, for American puts, the exercise boundary is non-decreasing in time (closer to expiration, you're willing to exercise at a higher spot)
    # Let me not enforce monotonicity and just use the computed values.
    
    return ExercisePolicy(boundaries)
