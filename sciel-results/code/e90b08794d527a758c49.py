"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class BermudanExercisePolicy:
    """Bermudan put exercise policy using binomial tree backward induction."""
    
    def __init__(self, K, r, sigma, T, n_exercise_dates, exercise_times):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_exercise_dates = n_exercise_dates
        self.exercise_times = exercise_times  # array of exercise times (sorted)
        
        # Compute critical thresholds via backward induction
        self._compute_thresholds()
    
    def _compute_thresholds(self):
        """Compute critical stock price thresholds for each exercise date."""
        n_ex = self.n_exercise_dates
        # We need to handle the discrete exercise dates
        # For backward induction, we work from the last exercise date back to the first
        
        # Use a fine grid for the binomial tree
        # Number of steps in the tree: use a reasonable number
        # The exercise dates are at t_1, t_2, ..., t_n
        # We need to compute continuation values at each exercise date
        
        # For simplicity, use a binomial tree with steps matching the exercise intervals
        # Exercise dates: exercise_times[0], exercise_times[1], ..., exercise_times[n-1]
        
        # Build tree from last exercise date backwards
        # At the last exercise date, continuation value = 0 (no more exercise opportunities)
        # So we exercise if K - S > 0, i.e., always exercise if in the money
        
        # Actually, for Bermudan options, we need to be more careful.
        # Let's use a standard approach: discretize time and use backward induction.
        
        # The exercise dates are at specific times. Between exercise dates,
        # the option is held (no exercise). At exercise dates, we can choose to exercise or hold.
        
        # For a put: payoff if exercised = max(K - S, 0)
        # Continuation value = discounted expected value of future optimal policy
        
        # Use a log-spaced grid for stock prices
        # S ranges roughly from 0 to 2*S0
        # Let's use a grid of stock prices
        
        n_grid = 2001
        S_max = 200.0
        S_min = 0.01
        S_grid = np.linspace(S_min, S_max, n_grid)
        
        # Discount factors between exercise dates
        # Exercise dates: self.exercise_times[0] to self.exercise_times[-1]
        # Also need to handle the terminal condition
        
        # For the last exercise date, the continuation value after that is 0
        # (since there are no more exercise opportunities)
        
        # Backward induction:
        # V_n(S) = max(K - S, 0)  at the last exercise date (no continuation)
        # V_i(S) = max(K - S, 0, discount * E[V_{i+1}(S_{next}) | S])
        
        # For GBM, given S at time t_i, S at time t_{i+1} is:
        # S_{i+1} = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        
        # We'll use a simple approach: for each S on the grid, compute the expected
        # continuation value using the GBM transition
        
        # Start from the last exercise date
        # V at last exercise date: since no more exercise, V = max(K-S, 0)
        # But wait - if it's the last exercise date, and we don't exercise, the option
        # expires worthless. So V_last(S) = max(K-S, 0)
        
        V = np.maximum(self.K - S_grid, 0.0)
        
        # Now go backwards through exercise dates
        for i in range(n_ex - 1, 0, -1):
            t_curr = self.exercise_times[i - 1]
            t_next = self.exercise_times[i]
            dt = t_next - t_curr
            
            if dt <= 0:
                # Shouldn't happen, but handle gracefully
                continue
            
            # For each S on the grid, compute E[V(S_next)] where S_next ~ GBM
            # S_next = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            # E[V(S_next)] can be computed analytically or numerically
            
            # Use quadrature: integrate over Z
            # E[V(S * exp((r-sigma^2/2)*dt + sigma*sqrt(dt)*Z))]
            # = (1/sqrt(2*pi)) * integral V(S * exp((r-sigma^2/2)*dt + sigma*sqrt(dt)*z)) * exp(-z^2/2) dz
            
            # Use Gauss-Hermite quadrature for efficiency
            # Or use a simple numerical integration
            
            # Let's use a simple approach: for each S, compute the expected value
            # using a discrete approximation
            
            # Actually, let's use a cleaner approach with log-normal expectation
            # For a put option continuation value, we can use the Black-Scholes-like formula
            # but since V is not linear, we need numerical integration
            
            # Use trapezoidal rule on a grid of Z values
            n_z = 101
            z_max = 4.0
            z_vals = np.linspace(-z_max, z_max, n_z)
            # Standard normal PDF weights
            z_weights = np.exp(-z_vals**2 / 2.0) / np.sqrt(2.0 * np.pi)
            dz = z_vals[1] - z_vals[0]
            
            # For each S on the grid, compute E[V(S_next)]
            # S_next = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*z)
            drift = (self.r - 0.5 * self.sigma**2) * dt
            vol_term = self.sigma * np.sqrt(dt)
            
            # S_next for each (S, z) pair
            # S_grid has n_grid elements, z_vals has n_z elements
            # S_next[i, j] = S_grid[i] * exp(drift + vol_term * z_vals[j])
            
            # This could be memory intensive: 2001 * 101 = ~200k, which is fine
            S_next = S_grid[:, np.newaxis] * np.exp(drift + vol_term * z_vals[np.newaxis, :])
            
            # Interpolate V at S_next
            # V is defined on S_grid, we need V(S_next)
            # Use np.interp
            V_next = np.interp(S_next, S_grid, V, left=0.0, right=0.0)
            
            # Expected continuation value for each S
            # E[V(S_next)] = sum_j V_next[i, j] * z_weights[j] * dz
            cont_val = np.sum(V_next * z_weights[np.newaxis, :] * dz, axis=1)
            
            # Discount
            discount = np.exp(-self.r * dt)
            cont_val = discount * cont_val
            
            # Exercise value
            exercise_val = np.maximum(self.K - S_grid, 0.0)
            
            # Optimal value: max(exercise, continuation)
            V = np.maximum(exercise_val, cont_val)
        
        # Now V contains the optimal value at the first exercise date
        # But we need thresholds for each exercise date, not just the first
        
        # Let's redo this to store thresholds at each exercise date
        thresholds = np.zeros(n_ex)
        
        # Recompute, storing the threshold at each step
        V = np.maximum(self.K - S_grid, 0.0)
        thresholds[n_ex - 1] = self.K  # At last date, exercise if S < K
        
        for i in range(n_ex - 1, 0, -1):
            t_curr = self.exercise_times[i - 1]
            t_next = self.exercise_times[i]
            dt = t_next - t_curr
            
            if dt <= 0:
                thresholds[i - 1] = thresholds[i]
                continue
            
            drift = (self.r - 0.5 * self.sigma**2) * dt
            vol_term = self.sigma * np.sqrt(dt)
            
            S_next = S_grid[:, np.newaxis] * np.exp(drift + vol_term * z_vals[np.newaxis, :])
            V_next = np.interp(S_next, S_grid, V, left=0.0, right=0.0)
            cont_val = np.sum(V_next * z_weights[np.newaxis, :] * dz, axis=1)
            discount = np.exp(-self.r * dt)
            cont_val = discount * cont_val
            
            exercise_val = np.maximum(self.K - S_grid, 0.0)
            V = np.maximum(exercise_val, cont_val)
            
            # Find the threshold: the largest S where exercise_val >= cont_val
            # i.e., where max(K-S, 0) >= cont_val
            # For S <= K: K - S >= cont_val
            # For S > K: 0 >= cont_val (which is false since cont_val >= 0)
            # So threshold is where K - S = cont_val, i.e., S = K - cont_val
            
            # Find the crossover point
            diff = exercise_val - cont_val
            # Find where diff changes from positive to negative
            # For small S, exercise_val is large, cont_val is small -> diff > 0
            # For large S, exercise_val is 0, cont_val is positive -> diff < 0
            
            # Find the last index where diff >= 0
            positive_mask = diff >= 0
            if np.any(positive_mask):
                last_pos = np.where(positive_mask)[0][-1]
                # Interpolate between last_pos and last_pos+1
                if last_pos < len(diff) - 1:
                    d0 = diff[last_pos]
                    d1 = diff[last_pos + 1]
                    if d0 - d1 != 0:
                        frac = d0 / (d0 - d1)
                        threshold = S_grid[last_pos] + frac * (S_grid[last_pos + 1] - S_grid[last_pos])
                    else:
                        threshold = S_grid[last_pos]
                else:
                    threshold = S_grid[last_pos]
            else:
                threshold = 0.0
            
            thresholds[i - 1] = threshold
        
        self.thresholds = thresholds
        self.S_grid = S_grid
        self.V_final = V
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time for the given states.
        
        current_state: array of spot prices
        time: current time
        
        Returns: boolean array, True if should exercise
        """
        # Find which exercise date we're at
        # The exercise times are self.exercise_times
        # We need to find the closest exercise date
        
        # Find the index of the exercise date closest to time
        # But we should be at an exercise date
        idx = np.argmin(np.abs(self.exercise_times - time))
        
        threshold = self.thresholds[idx]
        
        # For a put, exercise if S < threshold
        return current_state < threshold


def fit_exercise_policy(training_paths, contract):
    """Fit a Bermudan exercise policy."""
    start_time = time.time()
    deadline = start_time + 110.0  # 110s deadline, 40s safety margin
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    
    # Exercise dates: 12 monthly, no time-0
    # So at T/12, 2T/12, ..., 12T/12 = T
    exercise_times = np.linspace(T / n_exercise_dates, T, n_exercise_dates)
    
    # Create the policy
    policy = BermudanExercisePolicy(K, r, sigma, T, n_exercise_dates, exercise_times)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 10.0:
        # Not enough time, but we already have the policy
        pass
    
    return policy
# EVOLVE-BLOCK-END
