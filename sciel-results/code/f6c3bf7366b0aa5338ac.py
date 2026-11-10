"""Monte Carlo optimization of Bermudan put exercise thresholds."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy by optimizing 12 exercise thresholds
    using Monte Carlo backward induction and stochastic gradient ascent.
    
    Policy: exercise at date i if S_i < threshold_i.
    """
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: t = 1/12, 2/12, ..., 12/12
    exercise_times = np.array([(i + 1) / n_exercise for i in range(n_exercise)])
    
    n_paths = training_paths.shape[0]
    n_time_steps = training_paths.shape[1]
    
    # Map exercise dates to column indices
    if n_time_steps > 1:
        dt = T / (n_time_steps - 1)
    else:
        dt = T
    
    exercise_cols = []
    for t_ex in exercise_times:
        col = int(round(t_ex / dt))
        col = min(col, n_time_steps - 1)
        exercise_cols.append(col)
    
    # Initialize thresholds: start slightly below K (typical for puts)
    thresholds = np.full(n_exercise, K * 0.95)
    
    def compute_value_and_gradients(thresholds, paths, n_samples=None):
        """
        Compute the policy value and gradient estimates using backward induction.
        
        Returns: (value, gradients)
        """
        if n_samples is not None and n_samples < paths.shape[0]:
            idx = np.random.choice(paths.shape[0], n_samples, replace=False)
            paths = paths[idx]
        
        n_p = paths.shape[0]
        
        # Backward induction: compute continuation values
        # cont[i] = value of continuing at exercise date i (before decision)
        cont = np.zeros(n_exercise)
        
        # Last exercise date: if we don't exercise, option expires worthless
        # So cont[n-1] = 0 for all paths
        
        # For i from n-2 down to 0:
        # At date i+1, value = max(exercise, continue)
        # Exercise payoff = max(K - S_{i+1}, 0)
        # Continue value = cont[i+1]
        # So value_at_i+1 = where(S_{i+1} < thresh[i+1], payoff, cont[i+1])
        # cont[i] = disc(i->i+1) * value_at_i+1
        
        for i in range(n_exercise - 2, -1, -1):
            S_next = paths[:, exercise_cols[i + 1]]
            payoff_next = np.maximum(K - S_next, 0.0)
            
            # Decision at i+1
            exercise_mask = S_next < thresholds[i + 1]
            value_next = np.where(exercise_mask, payoff_next, cont[i + 1])
            
            # Discount back
            dt_i = exercise_times[i + 1] - exercise_times[i]
            disc_i = np.exp(-r * dt_i)
            cont[i] = disc_i * value_next
        
        # Value at first exercise date
        S_first = paths[:, exercise_cols[0]]
        payoff_first = np.maximum(K - S_first, 0.0)
        exercise_mask_0 = S_first < thresholds[0]
        value_first = np.where(exercise_mask_0, payoff_first, cont[0])
        
        # Discount to time 0
        disc_0 = np.exp(-r * exercise_times[0])
        value = disc_0 * np.mean(value_first)
        
        # Gradient estimation: for each threshold_i, the gradient is
        # dV/dtheta_i ≈ E[ (payoff_i - cont_i) * d(indicator(S_i < theta_i))/dtheta_i ]
        # Using a smoothed indicator: d/dtheta P(S < theta) ≈ density at theta
        # In practice, we can estimate by looking at paths near the threshold
        
        gradients = np.zeros(n_exercise)
        
        # For each exercise date, estimate the gradient
        for i in range(n_exercise):
            S_i = paths[:, exercise_cols[i]]
            payoff_i = np.maximum(K - S_i, 0.0)
            
            # The marginal benefit of exercising vs continuing at date i
            # = payoff_i - cont[i] (for i < n-1), or payoff_i for i = n-1
            if i < n_exercise - 1:
                marginal_benefit = payoff_i - cont[i]
            else:
                marginal_benefit = payoff_i
            
            # We want to increase threshold if marginal_benefit > 0 for paths near threshold
            # Estimate gradient using paths within a small window around threshold
            window = 0.5  # width of the window
            near_mask = np.abs(S_i - thresholds[i]) < window
            
            if np.sum(near_mask) > 0:
                # Average marginal benefit of paths near the threshold
                # If positive, we should increase threshold (exercise more)
                # If negative, we should decrease threshold (exercise less)
                gradients[i] = np.mean(marginal_benefit[near_mask])
            else:
                gradients[i] = 0.0
        
        return value, gradients
    
    # Stochastic gradient ascent
    n_iter = 150
    batch_size = min(1024, n_paths)
    lr_schedule = lambda it: 0.3 * (1.0 - it / n_iter) + 0.05
    
    best_value = -np.inf
    best_thresholds = thresholds.copy()
    
    for it in range(n_iter):
        # Take a random batch
        idx = np.random.choice(n_paths, batch_size, replace=False)
        batch = training_paths[idx]
        
        value, grads = compute_value_and_gradients(thresholds, batch)
        
        # Update thresholds
        lr = lr_schedule(it)
        for i in range(n_exercise):
            thresholds[i] += lr * grads[i]
            # Keep thresholds in reasonable range
            thresholds[i] = np.clip(thresholds[i], 0.0, K * 2.0)
        
        # Track best
        if value > best_value:
            best_value = value
            best_thresholds = thresholds.copy()
    
    # Use the best thresholds found
    thresholds = best_thresholds
    
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times, K):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
            self.K = K
            self.n_exercise = len(thresholds)
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            - current_state: array of current spot prices
            - time: current time (scalar)
            
            Returns:
            - boolean vector
            """
            t_val = float(time)
            # Find the exercise date index closest to time
            idx = np.argmin(np.abs(self.exercise_times - t_val))
            
            # Check if it's an exercise date
            if abs(self.exercise_times[idx] - t_val) > 0.01:
                return np.zeros(len(current_state), dtype=bool)
            
            # Exercise if S < threshold
            return current_state < self.thresholds[idx]
    
    return ExercisePolicy(thresholds, exercise_times, K)
# EVOLVE-BLOCK-END
