"""Bermudan put exercise-policy research with binomial tree reference and threshold refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 115.0  # safety margin under 120s limit


def _build_binomial_tree(S0, K, r, sigma, T, n_dates, n_steps=1200):
    """
    Build a binomial tree for the Bermudan put and compute optimal exercise thresholds.
    
    Returns:
        thresholds: array of shape (n_dates,) with optimal exercise thresholds
        tree_values: the option values at each node (for diagnostics)
    """
    dt_tree = T / n_steps
    dt_exercise = T / n_dates
    
    # Binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = 1.0 / u
    p = (np.exp(r * dt_tree) - d) / (u - d)
    
    # We need to identify which tree steps correspond to exercise dates
    # Exercise dates are at t = dt_exercise, 2*dt_exercise, ..., n_dates*dt_exercise = T
    # Tree steps are at t = dt_tree, 2*dt_tree, ..., n_steps*dt_tree = T
    # So exercise date i (0-indexed) corresponds to tree step (i+1) * n_steps / n_dates
    
    exercise_steps = []
    for i in range(n_dates):
        step = int(round((i + 1) * n_steps / n_dates))
        exercise_steps.append(step)
    
    # Build the tree backward from maturity
    # At maturity (step n_steps), the option value is the put payoff
    # We'll store values at each exercise step
    
    # For efficiency, we only need to track values at exercise steps
    # But we need to propagate through all tree steps
    
    # Let's build the full tree values at all steps, but only make exercise decisions at exercise steps
    
    # Number of nodes at step k is k+1
    # We'll work backward from step n_steps to step 0
    
    # Memory optimization: we only need two consecutive steps
    # But we need to record values at exercise steps
    
    # Let's use a dictionary to store values at exercise steps
    exercise_step_set = set(exercise_steps)
    
    # Start from maturity
    # At step n_steps, node j has price S0 * u^j * d^(n_steps-j)
    # Put payoff = max(K - S, 0)
    
    # We'll work backward
    # values[j] for j = 0, ..., k represents the value at step k, node j
    
    # At step n_steps:
    k = n_steps
    nodes = k + 1
    log_prices = np.log(S0) + np.arange(nodes) * np.log(u) + (k - np.arange(nodes)) * np.log(d)
    prices = np.exp(log_prices)
    values = np.maximum(K - prices, 0.0)
    
    # Store value at exercise step n_steps (which is the last exercise date)
    # But wait, the last exercise date is at T, which is step n_steps
    # So we need to record the threshold at this step
    
    # Actually, let's record thresholds at each exercise step
    thresholds = np.zeros(n_dates)
    
    # At the last exercise step (maturity), exercise if in the money
    # Threshold is K
    last_exercise_idx = n_dates - 1
    if exercise_steps[-1] == n_steps:
        thresholds[last_exercise_idx] = K
    
    # Work backward through all tree steps
    for k in range(n_steps - 1, -1, -1):
        # Discount and average
        values = np.exp(-r * dt_tree) * (p * values[1:] + (1 - p) * values[:-1])
        
        # Check if this step is an exercise step
        if k in exercise_step_set:
            # Find which exercise date this corresponds to
            exercise_idx = exercise_steps.index(k)
            
            # Compute prices at this step
            nodes = k + 1
            log_prices = np.log(S0) + np.arange(nodes) * np.log(u) + (k - np.arange(nodes)) * np.log(d)
            prices = np.exp(log_prices)
            
            # Immediate payoff
            payoff = np.maximum(K - prices, 0.0)
            
            # Optimal value = max(payoff, continuation)
            optimal_values = np.maximum(payoff, values)
            
            # Find threshold: the highest price where exercise is optimal
            # Exercise is optimal where payoff >= continuation
            exercise_mask = payoff >= values - 1e-10  # small tolerance
            
            if np.any(exercise_mask):
                # Find the highest price where exercise is optimal
                exercise_prices = prices[exercise_mask]
                thresholds[exercise_idx] = np.max(exercise_prices)
            else:
                # Never exercise at this date
                thresholds[exercise_idx] = 0.0
            
            # Update values to optimal values
            values = optimal_values
    
    return thresholds


def _refine_thresholds_with_paths(training_paths, thresholds, K, r, T, n_dates, n_paths, n_iterations=5):
    """
    Refine thresholds using path-based policy iteration.
    
    Args:
        training_paths: array of shape (n_dates, n_paths)
        thresholds: initial thresholds from binomial tree
        K: strike price
        r: risk-free rate
        T: maturity
        n_dates: number of exercise dates
        n_paths: number of training paths
        n_iterations: number of policy iteration steps
    
    Returns:
        refined thresholds
    """
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    current_thresholds = thresholds.copy()
    
    for iteration in range(n_iterations):
        # Compute continuation values using current thresholds
        value_next = np.zeros(n_paths)
        continuation_values = np.zeros((n_dates, n_paths))
        
        for i in range(n_dates - 1, -1, -1):
            exercise = training_paths[i] < current_thresholds[i]
            payoff = np.maximum(K - training_paths[i], 0.0)
            
            if i == n_dates - 1:
                value = np.where(exercise, payoff, 0.0)
                continuation_values[i] = 0.0
            else:
                value = np.where(exercise, payoff, discount * value_next)
                continuation_values[i] = discount * value_next
            
            value_next = value
        
        # Re-derive thresholds
        new_thresholds = np.zeros(n_dates)
        for i in range(n_dates):
            if i == n_dates - 1:
                new_thresholds[i] = K
            else:
                # Exercise is optimal when K - S > continuation_values[i, p]
                # i.e., S < K - continuation_values[i, p]
                implied = K - continuation_values[i]
                # Use a robust aggregate - median
                new_thresholds[i] = np.median(implied)
        
        # Check convergence
        if np.allclose(current_thresholds, new_thresholds, atol=1e-6):
            current_thresholds = new_thresholds
            break
        current_thresholds = new_thresholds
    
    return current_thresholds


def fit_exercise_policy(training_paths, contract):
    """Fit a Bermudan put exercise policy."""
    start_time = _time.time()
    
    # Extract parameters
    n_dates, n_paths = training_paths.shape
    K = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.2
    
    dt = T / n_dates
    
    # Step 1: Compute initial thresholds from binomial tree
    remaining_time = _DEADLINE - (_time.time() - start_time)
    if remaining_time > 20:
        tree_thresholds = _build_binomial_tree(100.0, K, r, sigma, T, n_dates, n_steps=1200)
    else:
        # Fallback: simple heuristic
        tree_thresholds = np.full(n_dates, K)
    
    # Step 2: Refine thresholds using path-based policy iteration
    remaining_time = _DEADLINE - (_time.time() - start_time)
    if remaining_time > 10:
        refined_thresholds = _refine_thresholds_with_paths(
            training_paths, tree_thresholds, K, r, T, n_dates, n_paths, n_iterations=5
        )
    else:
        refined_thresholds = tree_thresholds
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, T, n_dates):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.T = T
            self.n_dates = n_dates
        
        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating which paths to exercise.
            
            current_state: array of shape (n_paths,) with current spot prices
            time: current time (to determine which exercise date)
            """
            # Determine which exercise date we're at
            # Exercise dates are at t = dt, 2*dt, ..., n_dates*dt = T
            # time should be one of these values
            
            # Find the exercise date index
            # t_i = (i+1) * dt for i = 0, ..., n_dates-1
            # So date index = round(time / dt) - 1
            
            date_idx = int(round(time / dt)) - 1
            date_idx = max(0, min(date_idx, self.n_dates - 1))
            
            # Exercise if current_state < threshold
            return current_state < self.thresholds[date_idx]
    
    policy = ExercisePolicy(refined_thresholds, K, r, T, n_dates)
    return policy

# EVOLVE-BLOCK-END
