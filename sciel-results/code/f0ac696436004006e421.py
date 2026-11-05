"""Bermudan exercise-policy research: ML-based stopping rule with antithetic training."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric stopping boundary
    learned from training paths via backward induction.
    
    Parameters:
    -----------
    training_paths : ndarray, shape (n_paths, n_steps+1) or similar
        Training price paths.
    contract : dict or object
        Contract specification with fields like K, r, sigma, T, n_steps, etc.
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector.
    """
    # Extract contract parameters
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    n_steps = getattr(contract, 'n_steps', 12)
    
    # Ensure numeric types
    K = float(K)
    r = float(r)
    sigma = float(sigma)
    T = float(T)
    n_steps = int(n_steps)
    
    # Determine path shape
    # training_paths might be (n_paths, n_times) where n_times includes t=0
    if isinstance(training_paths, np.ndarray):
        paths = training_paths.copy()
    else:
        paths = np.array(training_paths)
    
    # Ensure 2D
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    n_paths, n_times = paths.shape
    
    # The exercise dates are at t_1, t_2, ..., t_n (no t_0)
    # So we have n_steps exercise opportunities
    # paths has n_times = n_steps + 1 entries (including t=0)
    
    # We'll use a parametric model for the stopping boundary:
    # Stop if S <= B(t) where B(t) = a * exp(-b * t) or similar
    # Actually for puts, we stop when S is low. The boundary should be decreasing over time
    # (as maturity approaches, the optimal stopping boundary for a put decreases to 0)
    # Wait, for a put, you want to exercise when S is low. As time to maturity decreases,
    # the value of waiting decreases, so you're more willing to exercise.
    # The critical price S* increases as time to maturity decreases.
    # At t=T, S* = K. As t->0, S* -> 0.
    # So S*(t) is an increasing function of time-to-maturity, or decreasing function of t.
    
    # Let's parameterize: S*(t) = K * (1 - exp(-alpha * (T - t))) / (1 - exp(-alpha * T))
    # At t=T: S* = K * (1 - 1) / ... = 0? No.
    # Let me think again.
    
    # Standard result: for a Bermudan put, the optimal exercise boundary is a function
    # of time. Let's use a simple parametric form:
    # B(t) = K * (1 - (t/T)^gamma) for gamma > 0
    # At t=T: B(T) = K * (1 - 1) = 0. Hmm, that's not right either.
    
    # Actually, let's just use a linear or exponential form and learn the parameters.
    # Let's use: B(t) = a + b * t
    # Or better, let's use the approach of learning the continuation value via
    # backward induction on a discretized state space.
    
    # Approach: Use the training paths to estimate the distribution of future payoffs
    # and compute the continuation value at each exercise date.
    
    # Step 1: Backward induction using the training paths
    # For each exercise date i (from last to first), compute the continuation value
    # as the discounted expected value of the optimal stopping rule from i+1 onwards.
    
    # We'll discretize the state space using quantiles from the training data.
    
    # Get the exercise dates (excluding t=0)
    # paths[:, i] gives the price at time i-1? Or time i?
    # Let's assume paths[:, j] is the price at time j*dt where j=0,...,n_steps
    # Exercise can happen at times 1, 2, ..., n_steps (indices 1 through n_steps in paths)
    
    # Actually, let me reconsider the structure. The problem says "12 monthly exercise dates (no time-0 exercise)".
    # So exercise dates are at t_1, t_2, ..., t_12.
    # paths likely has 13 columns: t_0, t_1, ..., t_12.
    
    # For backward induction:
    # At the last exercise date (t_12), the continuation value is 0 (no more exercise opportunities after).
    # So at t_12, we exercise if payoff > 0, i.e., if S < K.
    # Value at t_12: max(K - S, 0)
    
    # At t_11, continuation value = exp(-r*dt) * E[V_12 | S_11]
    # where V_12 is the value at t_12 under optimal stopping.
    # We exercise at t_11 if (K - S_11) > continuation_value_11
    
    # Let's implement this with a grid-based approach.
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # For each exercise date, we need to compute the value function.
    # We'll use the training paths to estimate the conditional distribution.
    
    # Number of exercise dates: n_steps (from 1 to n_steps)
    # Let's index exercise dates as 0 to n_steps-1, corresponding to times dt, 2dt, ..., T
    
    # Create a grid of S values for discretization
    # Use quantiles from all training data at relevant times
    all_prices = paths[:, 1:].flatten()  # Exclude t=0
    # Use a reasonable grid
    grid_min = max(0.01, np.percentile(all_prices, 0.1))
    grid_max = np.percentile(all_prices, 99.9)
    n_grid = 200
    S_grid = np.linspace(grid_min, grid_max, n_grid)
    
    # We'll store the value function at each exercise date on the grid
    # V[i][j] = value at exercise date i (0-indexed, so i=0 is t_1, i=n_steps-1 is t_12)
    # when current price is S_grid[j]
    
    # Initialize: at the last exercise date (i = n_steps - 1), 
    # V = max(K - S, 0) since no further exercise opportunities
    V = np.zeros((n_steps, n_grid))
    V[n_steps - 1, :] = np.maximum(K - S_grid, 0.0)
    
    # Backward induction
    for i in range(n_steps - 2, -1, -1):
        # At exercise date i, the continuation value is the discounted expected value
        # of V[i+1] given the current state S_i.
        # We need to estimate E[V[i+1](S_{i+1}) | S_i = s]
        
        # Use the training paths to estimate this conditional expectation.
        # For each training path, at time i (0-indexed exercise date, so paths[:, i+1] is S at t_{i+1}),
        # we have S_i = paths[:, i] and S_{i+1} = paths[:, i+1].
        # Wait, I need to be careful with indexing.
        
        # Let's say paths[:, k] is the price at time k*dt.
        # Exercise date i (0-indexed) corresponds to time (i+1)*dt, so paths[:, i+1].
        # The next exercise date is i+1 (0-indexed), corresponding to time (i+2)*dt, so paths[:, i+2].
        
        # So for exercise date i, current price is paths[:, i+1], next price is paths[:, i+2].
        
        S_current = paths[:, i + 1]  # Price at current exercise date
        S_next = paths[:, i + 2]     # Price at next exercise date
        
        # For each grid point s in S_grid, estimate E[V[i+1](S_next) | S_current = s]
        # Use kernel regression or simple binning
        
        # Simple approach: for each grid point, find training paths where S_current is close to s,
        # and average V[i+1](S_next) for those paths.
        
        # First, compute V[i+1] for each training path's S_next
        V_next_at_paths = np.interp(S_next, S_grid, V[i + 1], left=V[i+1, 0], right=V[i+1, -1])
        
        # Now, for each grid point, compute the conditional expectation
        # Use a simple approach: sort by S_current and use local averaging
        order = np.argsort(S_current)
        S_current_sorted = S_current[order]
        V_next_sorted = V_next_at_paths[order]
        
        # For each grid point, find the nearest neighbors in S_current_sorted
        # and average their V_next values
        cont_values = np.zeros(n_grid)
        for j in range(n_grid):
            s = S_grid[j]
            # Find paths where S_current is within a small window
            # Use a bandwidth based on the grid spacing
            bandwidth = (grid_max - grid_min) / n_grid * 2
            mask = np.abs(S_current_sorted - s) < bandwidth
            if np.sum(mask) > 5:
                cont_values[j] = np.mean(V_next_sorted[mask])
            else:
                # Fallback: use all paths or nearest
                if np.sum(mask) > 0:
                    cont_values[j] = np.mean(V_next_sorted[mask])
                else:
                    # Use the overall mean
                    cont_values[j] = np.mean(V_next_sorted)
        
        # Continuation value at exercise date i
        continuation = discount * cont_values
        
        # Optimal value: max(immediate payoff, continuation)
        immediate = np.maximum(K - S_grid, 0.0)
        V[i, :] = np.maximum(immediate, continuation)
        
        # Also store the exercise decision for this grid point
        # (not strictly needed for the policy, but useful for diagnostics)
    
    # Now we have the value function V at each exercise date.
    # The exercise policy: at exercise date i, exercise if immediate payoff > continuation value.
    # But we can also just use: exercise if S <= B_i where B_i is the boundary.
    # Let's extract the boundary for each exercise date.
    
    # For each exercise date i, find the largest S where we exercise
    # (i.e., where immediate >= continuation)
    boundaries = np.zeros(n_steps)
    for i in range(n_steps):
        immediate = np.maximum(K - S_grid, 0.0)
        # Recompute continuation for this date
        if i == n_steps - 1:
            cont = np.zeros(n_grid)
        else:
            S_current = paths[:, i + 1]
            S_next = paths[:, i + 2]
            V_next_at_paths = np.interp(S_next, S_grid, V[i + 1], left=V[i+1, 0], right=V[i+1, -1])
            order = np.argsort(S_current)
            S_current_sorted = S_current[order]
            V_next_sorted = V_next_at_paths[order]
            cont_values = np.zeros(n_grid)
            for j in range(n_grid):
                s = S_grid[j]
                bandwidth = (grid_max - grid_min) / n_grid * 2
                mask = np.abs(S_current_sorted - s) < bandwidth
                if np.sum(mask) > 5:
                    cont_values[j] = np.mean(V_next_sorted[mask])
                else:
                    if np.sum(mask) > 0:
                        cont_values[j] = np.mean(V_next_sorted[mask])
                    else:
                        cont_values[j] = np.mean(V_next_sorted)
            cont = discount * cont_values
        
        # Exercise where immediate > continuation
        exercise_mask = immediate >= cont
        # Find the largest S where we exercise
        if np.any(exercise_mask):
            exercise_indices = np.where(exercise_mask)[0]
            boundaries[i] = S_grid[exercise_indices[-1]]
        else:
            boundaries[i] = 0.0
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, S_grid, n_steps):
            self.boundaries = boundaries
            self.S_grid = S_grid
            self.n_steps = n_steps
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise.
            
            Parameters:
            -----------
            current_state : ndarray of shape (n_eval_paths,)
                Current prices.
            time : int or float
                Time index or time value.
            
            Returns:
            --------
            ndarray of bools, shape (n_eval_paths,)
            """
            # Convert time to exercise date index
            if isinstance(time, (int, np.integer)):
                # Assume time is the exercise date index (1-based)
                idx = int(time) - 1
            else:
                # Assume time is the actual time value
                idx = int(round(float(time) / dt)) - 1
            
            # Clamp index
            idx = max(0, min(idx, self.n_steps - 1))
            
            boundary = self.boundaries[idx]
            return np.array(current_state) <= boundary
    
    return ExercisePolicy(boundaries, S_grid, n_steps)
# EVOLVE-BLOCK-END
