"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on training paths.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_time_steps)
        Simulated spot price paths. n_time_steps should be 13 (t=0 to t=12).
    contract : dict
        Contract parameters with keys 'K', 'r', 'sigma', 'T', 'n_steps'.
        
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # Convert to numpy array
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_time = paths.shape
    
    # Monthly step size
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Exercise dates are t=1, 2, ..., 12 (no exercise at t=0)
    # We need to compute optimal exercise decisions for each exercise date
    # using backward induction
    
    # For each exercise date j (from 12 down to 1), compute:
    # - Immediate payoff: max(K - S_j, 0)
    # - Continuation value: discount * E[max(K - S_{j+1}, 0) | exercise at j+1 or later]
    # Exercise if immediate payoff >= continuation value
    
    # We'll store the exercise thresholds or decisions for each date
    # Since we only have discrete paths, we'll use the empirical distribution
    
    # Backward induction: compute value function at each exercise date
    # V_j(S) = max(K - S, discount * E[V_{j+1}(S_{j+1}) | S_j = S])
    
    # At the last exercise date (j=12), V_12(S) = K - S (must exercise if in the money)
    # Actually at j=12, there's no future, so V_12(S) = max(K - S, 0)
    
    # We'll compute for each exercise date j, the continuation value as a function of S
    # using the training paths
    
    # Strategy: For each exercise date j, group paths by their state at time j,
    # and compute the average continuation value from future exercise dates
    
    # More efficiently: for each path, we can compute the optimal exercise decision
    # using backward induction along each path, but this is path-dependent
    
    # Better approach: Use the fact that for a Bermudan option with GBM, the optimal
    # exercise region at each date is S <= threshold_j
    # We can estimate thresholds by backward induction on the training paths
    
    # Let's do proper backward induction:
    # Start from the last exercise date and work backwards
    
    # At exercise date j, for a path with state S_j:
    # - If j == 12 (last date): exercise if S_12 < K
    # - If j < 12: exercise if K - S_j >= discount * E[continuation_value(S_{j+1}) | S_j]
    
    # The continuation value at date j+1 depends on whether we exercise at j+1 or later
    
    # Let's compute this iteratively:
    # value[j, path] = optimal value from date j onwards, given we're at date j
    
    n_exercise_dates = n_steps  # dates 1 through 12
    
    # value[j, i] = optimal continuation value at exercise date j for path i
    # We'll use 0-indexed: value[0] corresponds to exercise date 1, value[11] to date 12
    
    value = np.zeros((n_exercise_dates, n_paths), dtype=np.float64)
    
    # Last exercise date (index n_exercise_dates - 1, which is date 12)
    # At the last date, you must take the payoff if in the money
    last_idx = n_exercise_dates - 1
    S_last = paths[:, last_idx + 1]  # time index 12
    value[last_idx] = np.maximum(K - S_last, 0.0)
    
    # Work backwards from date 11 to date 1
    for j in range(n_exercise_dates - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
        S_j = paths[:, j + 1]  # time index j+1
        S_next = paths[:, j + 2]  # time index j+2
        
        # Immediate payoff
        immediate = np.maximum(K - S_j, 0.0)
        
        # Continuation value: discount * value at next exercise date
        # But we need to account for the fact that at the next date, we might exercise
        # value[j+1] already includes the optimal decision at j+1
        continuation = discount * value[j + 1]
        
        # Optimal value at date j
        value[j] = np.maximum(immediate, continuation)
    
    # Now we have value[j, i] for each exercise date j and path i
    # The exercise decision at date j for path i is:
    # exercise if immediate_payoff >= continuation_value
    # i.e., if max(K - S_j, 0) >= discount * value[j+1, i]
    
    # For the exercise policy, we need to determine: given current state S at time t,
    # should we exercise?
    
    # The training paths give us empirical distributions. We can estimate the
    # exercise boundary as a function of S at each date.
    
    # For each exercise date j, find the threshold such that paths with S_j <= threshold
    # should exercise.
    
    # Actually, the decision depends on both S_j and the future distribution.
    # With GBM, the continuation value is a smooth function of S_j.
    
    # Let's estimate the continuation value as a function of S_j for each date.
    # We can use the training paths to fit a simple model.
    
    # For each exercise date j (0 to n_exercise_dates-2):
    #   continuation_value_j(S) = discount * E[value[j+1] | S_j = S]
    #   Exercise if K - S >= continuation_value_j(S)
    
    # We'll use the empirical relationship from the training paths.
    # For each date j, sort paths by S_j and compute the average continuation value.
    
    # Store exercise boundaries
    # For each exercise date j, we'll store the threshold S_j* such that
    # we exercise if S_j <= S_j*
    
    exercise_thresholds = np.zeros(n_exercise_dates, dtype=np.float64)
    
    for j in range(n_exercise_dates - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
        
        S_j = paths[:, j + 1]
        immediate = np.maximum(K - S_j, 0.0)
        
        if j == n_exercise_dates - 1:
            # Last date: exercise if in the money
            exercise_mask = S_j < K
            if np.any(exercise_mask):
                exercise_thresholds[j] = np.max(S_j[exercise_mask])
            else:
                exercise_thresholds[j] = K
        else:
            continuation = discount * value[j + 1]
            exercise_mask = immediate >= continuation
            
            if np.any(exercise_mask):
                exercise_thresholds[j] = np.max(S_j[exercise_mask])
            else:
                exercise_thresholds[j] = 0.0
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K_val):
            self.thresholds = thresholds
            self.K = K_val
            
        def exercise(self, current_state, time_idx):
            """
            Determine whether to exercise the option.
            
            Parameters:
            -----------
            current_state : array-like, shape (n,)
                Current spot prices for each path.
            time_idx : int
                Current time index (1 to 12 for exercise dates).
                
            Returns:
            --------
            boolean array, shape (n,)
                True if the option should be exercised.
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            # time_idx is the exercise date (1 to 12)
            # Our thresholds are indexed 0 to 11 for dates 1 to 12
            if time_idx < 1 or time_idx > n_exercise_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            j = time_idx - 1  # Convert to 0-indexed
            threshold = self.thresholds[j]
            
            # Exercise if S <= threshold
            return current_state <= threshold
    
    policy = ExercisePolicy(exercise_thresholds, K)
    return policy
# EVOLVE-BLOCK-END
