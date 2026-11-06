"""Bermudan exercise policy using Longstaff-Schwarz Monte Carlo with Least Squares Regression."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 130.0  # safety margin
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # training_paths shape: (num_paths, num_steps) or (num_steps, num_paths)
    # Contract says "4096 training paths". Usually shape is (num_paths, num_time_steps).
    # Let's verify shape. If training_paths[0] is a path, then shape is (num_paths, steps).
    # If training_paths[0] is a time slice, shape is (steps, num_paths).
    # Standard convention in many tasks: (num_paths, num_steps).
    # Let's assume (num_paths, num_steps).
    # Time steps: 12 monthly. So likely 13 points (0 to 12) or 12 points (1 to 12).
    # "12 monthly exercise dates (no time-0 exercise)".
    # So exercise at t=1/12, ..., 12/12.
    # Paths likely include t=0 to t=T.
    
    paths = np.array(training_paths, dtype=np.float64)
    if paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
    
    # Check orientation. 
    # If shape[0] is 4096, it's (paths, steps).
    # If shape[1] is 4096, it's (steps, paths).
    num_paths_expected = 4096
    if paths.shape[0] == num_paths_expected:
        # (paths, steps)
        num_paths, num_steps = paths.shape
    elif paths.shape[1] == num_paths_expected:
        # (steps, paths) -> transpose
        paths = paths.T
        num_paths, num_steps = paths.shape
    else:
        # Fallback: assume first dimension is paths
        num_paths, num_steps = paths.shape
        if num_paths < num_steps:
             paths = paths.T
             num_paths, num_steps = paths.shape

    # We need to identify the time indices corresponding to exercise dates.
    # T=1.0, 12 monthly dates.
    # dt = T / 12
    # Exercise at i=1, ..., 12.
    # If num_steps == 13, indices are 1..12.
    # If num_steps == 12, indices are 0..11? No, "no time-0".
    # Usually paths include t=0. So steps are 0, 1, ..., 12.
    # Let's assume the last column is t=T.
    # Exercise dates are at the last 12 time points?
    # If num_steps is 13, indices 1 to 12 are exercise dates.
    # If num_steps is 12, maybe indices 0 to 11 correspond to t=1/12 to t=1?
    # Let's check the values. S0 is around 90-110.
    # Let's assume standard GBM path generation: S_0, S_1, ..., S_N.
    # N=12. Exercise at 1..12.
    
    # Determine exercise indices
    # We assume the time grid is uniform.
    # If num_steps == 13, exercise_indices = np.arange(1, 13)
    # If num_steps == 12, exercise_indices = np.arange(0, 12) ? 
    # "12 monthly exercise dates (no time-0 exercise)"
    # This implies the dates are t_1, ..., t_12.
    # If the array has 13 columns, col 0 is t=0, col 12 is t=1.
    # If the array has 12 columns, col 0 is t=1/12? Unlikely.
    # Let's assume 13 columns (t=0 to t=1).
    
    if num_steps >= 13:
        exercise_indices = np.arange(1, 13)
    elif num_steps == 12:
        # Maybe it starts at t=1/12?
        exercise_indices = np.arange(0, 12)
    else:
        # Fallback: last 12 steps
        exercise_indices = np.arange(num_steps - 12, num_steps)
        
    # Ensure indices are within bounds
    exercise_indices = exercise_indices[exercise_indices < num_steps]
    
    # Backward induction
    # We store regression coefficients for each exercise date.
    # coeffs[i] corresponds to exercise date exercise_indices[i]
    # coeffs[i] is a vector of weights for features.
    
    # Features: S, log(S), 1 (bias)
    # Maybe add S^2 or (S-K)^2?
    # Let's start with S, log(S), 1.
    
    def get_features(S):
        S = np.asarray(S).flatten()
        f1 = S
        f2 = np.log(S + 1e-10) # avoid log(0)
        f0 = np.ones_like(S)
        return np.column_stack([f1, f2, f0])
    
    # Discount factors
    dt = T / 12.0
    discount_step = np.exp(-r * dt)
    
    # Initialize continuation values at maturity (after last exercise)
    # At the last exercise date, if not exercised, value is 0 (European put at maturity is max(K-S,0), but here it's Bermudan. 
    # Wait, at the LAST exercise date (t=1), if you don't exercise, the option expires. 
    # So continuation value at t=1 is 0? 
    # No, the "continuation value" at t=1 is the value of holding the option. 
    # But it expires at t=1. So if you don't exercise at t=1, you get 0.
    # So the decision at t=1 is: max(Intrinsic, 0).
    # So we can treat the "future payoff" for regression at t=1 as 0.
    
    # Let's iterate backward from the last exercise date to the first.
    # exercise_indices are e.g., [1, 2, ..., 12]
    # We need to process them in reverse order: 12, 11, ..., 1
    
    # Store coefficients for each exercise date
    # coeffs_by_index: dict mapping index to coefficients
    coeffs_by_index = {}
    
    # We need the "future payoff" for regression.
    # Let V_future[i] be the value of the option at time i, assuming optimal exercise from i+1 to end.
    # At the last exercise date (idx = 12), V_future is 0 (if we don't exercise, it's worth 0).
    # So at idx 12, we regress 0? No.
    # The regression at step i predicts the continuation value C_i(S_i).
    # C_i(S_i) = E[ V_{i+1}(S_{i+1}) | S_i ]
    # Where V_{i+1} is the value at step i+1 (max(Intrinsic, Continuation)).
    
    # Let's define the values at each step.
    # Values at step 12 (t=1): 
    #   Intrinsic = max(K - S_12, 0)
    #   Continuation = 0
    #   Value_12 = Intrinsic
    #   Exercise_12 = True (since Intrinsic >= 0)
    
    # Values at step 11 (t=11/12):
    #   Intrinsic = max(K - S_11, 0)
    #   Continuation = E[ Value_12 | S_11 ]
    #   We regress Value_12 on S_11 to estimate Continuation.
    #   Value_11 = max(Intrinsic, Continuation)
    #   Exercise_11 = (Intrinsic > Continuation)
    
    # So we need to store the "Value" at each step for the paths that haven't exercised yet?
    # Actually, for the regression, we only use paths that are still "alive" (not exercised).
    # But for the policy, we need to know the continuation value for ANY state.
    # The regression gives us a function C_i(S).
    
    # Algorithm:
    # 1. Start at the last exercise date (idx = 12).
    #    Value_12 = Intrinsic_12.
    #    (We don't need to regress here because Continuation_12 = 0).
    #    But we need to know which paths exercised at 12? 
    #    For the next step (11), we regress Value_12 on S_11.
    #    Do we use ALL paths or only those that didn't exercise at 12?
    #    Standard LSM: Use all paths for the regression, but the "target" is the realized value.
    #    Wait, if a path exercised at 12, its value is Intrinsic_12.
    #    If it didn't, its value is 0.
    #    So Value_12 = Intrinsic_12 for all paths (since it's the last date, you always exercise if Intrinsic > 0).
    #    So we regress Intrinsic_12 on S_11.
    
    # 2. Move to idx = 11.
    #    Regress Value_12 on S_11 -> Get Coeffs_11.
    #    Continuation_11 = S_11 @ Coeffs_11.
    #    Intrinsic_11 = max(K - S_11, 0).
    #    Value_11 = max(Intrinsic_11, Continuation_11).
    #    Exercise_11 = (Intrinsic_11 > Continuation_11).
    #    Note: For the next regression (at 10), we need the "realized" Value_11.
    #    However, the "realized" value depends on the exercise decision.
    #    Standard LSM uses the "Americanized" values.
    #    Value_11_realized = Intrinsic_11 if Exercise_11 else Continuation_11.
    #    Actually, Value_11 = max(Intrinsic, Continuation) IS the Americanized value.
    #    So we regress Value_11 on S_10.
    
    # 3. Repeat backward.
    
    # We need to store the coefficients for the policy.
    # The policy at time t (index i) exercises if Intrinsic_i > Continuation_i(S_i).
    # Continuation_i(S) is given by the regression coefficients learned at step i (predicting Value_{i+1}).
    # Wait, the regression at step i predicts Value_{i+1} using S_i.
    # So Coeffs_i are learned from data (S_i, Value_{i+1}).
    
    # Let's trace indices:
    # exercise_indices: [1, 2, ..., 12]
    # 
    # Step 12 (Last):
    #   S_12, Intrinsic_12.
    #   Value_12 = Intrinsic_12.
    #   (No regression needed for policy at 12, as Continuation is 0. Policy: Exercise if Intrinsic > 0).
    #   But we need Coeffs for step 11 to predict Value_12 from S_11.
    #   So we regress Value_12 on S_11.
    #   Let's call these coeffs "coeffs_for_11".
    
    # Step 11:
    #   S_11, Intrinsic_11.
    #   Continuation_11 = S_11 @ coeffs_for_11.
    #   Value_11 = max(Intrinsic_11, Continuation_11).
    #   Regress Value_11 on S_10 -> coeffs_for_10.
    
    # ...
    # Step 1:
    #   S_1, Intrinsic_1.
    #   Continuation_1 = S_1 @ coeffs_for_1.
    #   Value_1 = max(Intrinsic_1, Continuation_1).
    #   (No next step, so no regression needed for policy, but we would have learned coeffs_for_1 from Value_2 on S_1).
    
    # So we need to store coeffs_for_i for i=1 to 11.
    # For i=12, the policy is simply Intrinsic > 0.
    
    # Implementation:
    # current_values = Intrinsic at last step (12)
    # For i from 11 down to 1:
    #   S_i = paths[:, i]
    #   S_next = paths[:, i+1] # Not needed for regression, we regress current_values on S_i
    #   Wait, the regression is: current_values (which is Value_{i+1}) ~ S_i.
    #   So we regress Value_{i+1} on S_i.
    #   
    #   Let's refine:
    #   We have an array `future_values` of shape (num_paths,).
    #   Initially, future_values = Intrinsic at step 12.
    #   
    #   Loop i from 11 down to 1:
    #       S_i = paths[:, i]
    #       Features_i = get_features(S_i)
    #       Coeffs_i = lstsq(Features_i, future_values)
    #       coeffs_by_index[i] = Coeffs_i
    #       
    #       Continuation_i = Features_i @ Coeffs_i
    #       Intrinsic_i = np.maximum(K - S_i, 0)
    #       Value_i = np.maximum(Intrinsic_i, Continuation_i)
    #       
    #       # Update future_values for the next iteration (i-1)
    #       future_values = Value_i
    #       
    #       # Check deadline
    #       if time.time() > DEADLINE:
    #           break
    #
    #   After loop, coeffs_by_index has keys 1..11.
    #   For index 12, we know the policy is Exercise if Intrinsic > 0.
    
    # Let's execute this.
    
    # Get last exercise index
    last_idx = exercise_indices[-1]
    
    # Initialize future values at the last exercise date
    # Value at last date is Intrinsic (since continuation is 0)
    S_last = paths[:, last_idx]
    future_values = np.maximum(K - S_last, 0.0)
    
    # Store policy for last index
    # Policy: Exercise if Intrinsic > 0
    # We can store a special flag or just handle it in the class.
    # Let's store coeffs as None for the last index, indicating "Always exercise if intrinsic > 0".
    coeffs_by_index[last_idx] = None
    
    # Iterate backward
    # We need to map exercise_indices to the actual column indices in paths.
    # exercise_indices are [1, 2, ..., 12]
    # We iterate from second-to-last down to first.
    
    # Get the list of indices in reverse, excluding the last one
    indices_to_process = exercise_indices[:-1][::-1]
    
    for idx in indices_to_process:
        if time.time() > DEADLINE:
            break
            
        S_i = paths[:, idx]
        
        # Create features
        F_i = get_features(S_i)
        
        # Regression: future_values ~ F_i
        # Use lstsq
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(F_i, future_values, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback: simple mean
            coeffs = np.array([0.0, 0.0, np.mean(future_values)])
            
        coeffs_by_index[idx] = coeffs
        
        # Calculate continuation value
        cont_val = F_i @ coeffs
        
        # Calculate intrinsic value
        intr_val = np.maximum(K - S_i, 0.0)
        
        # Value at this step
        value_i = np.maximum(intr_val, cont_val)
        
        # Update future values for the next (earlier) step
        future_values = value_i
        
    # Create policy object
    class ExercisePolicy:
        def __init__(self):
            self.coeffs_by_index = coeffs_by_index
            self.exercise_indices = exercise_indices
            self.K = K
            self.num_steps = num_steps
            
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: current time
            Returns: boolean array
            """
            # Determine the index
            # time is continuous. We need to map it to the discrete index.
            # T=1.0, 12 steps.
            # t_k = k * T / 12.
            # index = round(time * 12)
            # But we must ensure it matches one of our exercise_indices.
            
            # Calculate step
            step_float = time * 12.0
            step_int = int(round(step_float))
            
            # Clamp to valid range
            step_int = max(0, min(step_int, 12))
            
            # Check if this is an exercise date
            # Our exercise_indices are [1, ..., 12]
            if step_int not in self.coeffs_by_index:
                # Not an exercise date
                return np.zeros(len(current_state), dtype=bool)
            
            coeffs = self.coeffs_by_index[step_int]
            S = np.asarray(current_state).flatten()
            
            if coeffs is None:
                # Last exercise date
                # Exercise if Intrinsic > 0
                intr_val = np.maximum(self.K - S, 0.0)
                return intr_val > 0.0
            else:
                # Calculate continuation
                F = get_features(S)
                cont_val = F @ coeffs
                
                # Calculate intrinsic
                intr_val = np.maximum(self.K - S, 0.0)
                
                # Exercise if Intrinsic > Continuation
                return intr_val > cont_val
                
    policy = ExercisePolicy()
    return policy

# EVOLVE-BLOCK-END
