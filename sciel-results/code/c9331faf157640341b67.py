"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo regression.
    
    Args:
        training_paths: array of shape (n_paths, n_times+1) with simulated stock price paths
        contract: dict with contract parameters
        
    Returns:
        An object with exercise(current_state, time) method returning boolean vector.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Ensure training_paths is a numpy array
    training_paths = np.asarray(training_paths)
    n_paths, n_time_points = training_paths.shape
    
    # Determine exercise indices in the training paths
    # Assume standard monthly spacing if not explicitly defined in contract
    # training_paths columns: 0..n_time_points-1
    # Exercise dates are at 1/12, 2/12, ..., 12/12 of T
    # If n_time_points == 13, indices 1..12 correspond to exercise dates.
    # General case: map time to index.
    # The contract says "12 monthly exercise dates (no time-0 exercise)".
    # Typically, simulation paths are generated at these discrete time steps.
    
    # Heuristic for indices:
    # If n_time_points is 13, indices 1 to 12 are the exercise dates.
    # If n_time_points is different, we assume uniform spacing and pick the closest indices.
    exercise_times = np.linspace(1.0/n_exercise, T, n_exercise)
    
    # Map exercise times to indices in training_paths
    # Assuming training_paths time grid is uniform from 0 to T with n_time_points-1 steps?
    # Or maybe it just has the exercise dates + time 0?
    # Given "4096 training paths" and "12 monthly", it's highly likely shape is (4096, 13).
    if n_time_points == n_exercise + 1:
        exercise_indices = list(range(1, n_exercise + 1))
    else:
        # Fallback: linear interpolation of indices
        # Total steps in path data
        steps_per_unit_time = (n_time_points - 1) / T
        exercise_indices = [int(round(t * steps_per_unit_time)) for t in exercise_times]
        # Ensure indices are within bounds and unique
        # If duplicates or out of bounds, clamp
        exercise_indices = [max(1, min(n_time_points - 1, idx)) for idx in exercise_indices]

    # Discount factors between exercise dates
    # dt is uniform for monthly dates
    dt = T / n_exercise
    discount_factor = np.exp(-r * dt)

    # Polynomial degree for regression
    # Higher degree fits better but can overfit. 4-6 is standard for 4096 paths.
    poly_degree = 5 

    # Backward induction
    # V_next_coeffs: coefficients of the polynomial approximating the value at the NEXT exercise date
    # We start from the last exercise date.
    # At the last exercise date (j = n_exercise - 1, 0-indexed), the option expires.
    # The value is the intrinsic value: max(K - S, 0).
    # This is not a polynomial, but for the step before it, we need the expected value of this payoff.
    # Actually, the standard LSM works backwards:
    # 1. Start at the last date. The value is intrinsic.
    # 2. For the second-to-last date, compute the continuation value based on the value at the last date.
    #    Continuation Value_i = discount * V_last(S_{i, last})
    #    Fit a polynomial to (S_{i, second_last}, Continuation Value_i)
    #    V_second_last(S) = max(K-S, Polynomial(S))
    # 3. Repeat for previous dates.
    
    # We need to store the fitted polynomial for the "current" date to be used as V_next in the previous step.
    # Let's store the coefficients of the polynomial that represents the OPTIMAL VALUE at the next date?
    # No, the regression is on the CONTINUATION value.
    # The value at date j is V_j(S) = max(Intrinsic, Regressed_Continuation).
    # To compute the continuation for date j-1, we need V_j(S_{j}) for each path.
    # So we need to be able to evaluate V_j(S) for any S.
    # Since V_j is the max of intrinsic and a polynomial, we can store the polynomial coefficients
    # and evaluate V_j(S) = max(K-S, Poly_j(S)).
    
    # Store polynomial coefficients for the value function at each exercise date
    # poly_coeffs[j] corresponds to the polynomial part of V_j(S)
    # Note: For the last date, V_last(S) = max(K-S, 0). We can represent this as a polynomial?
    # No, max(K-S, 0) is not a polynomial.
    # However, for the regression at j-1, we need the realized values V_last(S_last).
    # We can just calculate these directly from the intrinsic formula.
    
    # Let's iterate from last to first.
    # j = n_exercise - 1 (last)
    # V_j is intrinsic.
    # j = n_exercise - 2
    # Continuation = disc * V_{j+1}(S_{j+1})
    # Fit Poly_j to (S_j, Continuation)
    # V_j(S) = max(K-S, Poly_j(S))
    
    # We need to store the polynomial coefficients for V_j(S) for j = 1..n_exercise-1
    # And for j = n_exercise-1, we know it's intrinsic.
    
    # Actually, it's easier to store the decision boundary or the polynomial itself.
    # Let's store the polynomial coefficients for the continuation value fit at each step.
    # But the exercise decision depends on V_j(S) = max(Intrinsic, Cont_j(S)).
    # So we need Cont_j(S) to make the decision.
    
    # Let's store `cont_poly_coeffs[j]` for j=0..n_exercise-1.
    # For j = n_exercise-1 (last), Cont is 0? No, the value is intrinsic.
    # The continuation from the last date is 0 (no future).
    # So V_last(S) = max(K-S, 0).
    
    # Let's define the loop backwards.
    
    # Initialize for the last exercise date
    # The value at the last date is intrinsic.
    # We don't need a polynomial for the last date's decision, it's always exercise if ITM.
    # But for the previous date, we need the values at the last date.
    
    # Let's store the polynomial coefficients for the CONTINUATION value at each date j (j=0 to n_exercise-2)
    # For the last date (j=n_exercise-1), the continuation is 0.
    
    # We'll store a list of polynomial coefficients.
    # coeffs[j] is the polynomial approximating E[disc * V_{j+1} | S_j]
    
    # Start from the end
    # V_{last}(S) = max(K-S, 0)
    
    # We need to evaluate V_{j+1}(S) for each path to compute the target for regression at j.
    # V_{j+1}(S) = max(K-S, Poly_{j+1}(S))  (for j+1 < last)
    # V_{last}(S) = max(K-S, 0)
    
    # Let's store the polynomials for V_j(S) for j = 1..n_exercise-1?
    # No, V_j is defined by the max. We can store the polynomial that fits the continuation.
    
    # Let's use a list `polys` where `polys[j]` is the numpy array of coefficients for the polynomial
    # that approximates the continuation value at exercise date j.
    # For the last date, continuation is 0, so no polynomial needed (or zero poly).
    
    # We'll work backwards from j = n_exercise - 1 down to 0.
    
    # Initialize: 
    # At j = n_exercise - 1, V(S) = max(K-S, 0).
    # We don't fit a polynomial here.
    
    # For j = n_exercise - 2 down to 0:
    # 1. Get S_j and S_{j+1} from training_paths.
    # 2. Compute V_{j+1}(S_{j+1}) using the stored info from step j+1.
    # 3. Target = discount * V_{j+1}(S_{j+1})
    # 4. Filter paths where S_j < K (in the money) to fit the polynomial.
    # 5. Fit polynomial of degree `poly_degree` to (S_j, Target) for ITM paths.
    # 6. Store the coefficients.
    
    # Data structures:
    # `stored_polys[j]` will hold the coefficients for the continuation value at date j.
    # `stored_is_last[j]` indicates if j is the last date (where V is intrinsic).
    
    n_ex = n_exercise
    # coeffs[j] for j in 0..n_ex-1
    # coeffs[j] is None if j is the last date (value is intrinsic)
    coeffs = [None] * n_ex
    
    # We need to be able to evaluate V_j(S) for any S.
    # If coeffs[j] is None: V_j(S) = max(K-S, 0)
    # Else: V_j(S) = max(K-S, np.polyval(coeffs[j], S))
    
    # Start from the second to last date (j = n_ex - 2) down to 0.
    # The last date (j = n_ex - 1) has coeffs[n_ex-1] = None.
    
    # Loop backwards
    for j in range(n_ex - 2, -1, -1):
        if time.time() - start_time > DEADLINE:
            # Fallback: use simple intrinsic exercise for remaining dates
            # This is a conservative fallback
            break
        
        idx_j = exercise_indices[j]
        idx_next = exercise_indices[j + 1]
        
        S_j = training_paths[:, idx_j]
        S_next = training_paths[:, idx_next]
        
        # Evaluate V_next(S_next)
        if coeffs[j + 1] is None:
            # Last date
            V_next = np.maximum(K - S_next, 0.0)
        else:
            # Evaluate polynomial
            V_next_poly = np.polyval(coeffs[j + 1], S_next)
            V_next = np.maximum(K - S_next, V_next_poly)
        
        # Continuation value target for regression
        target = discount_factor * V_next
        
        # Filter for In-The-Money paths at date j
        # For a put, ITM is S < K
        itm_mask = S_j < K
        
        # If no ITM paths, we can't fit a polynomial. 
        # In this case, the continuation value is likely very low or zero.
        # We can set the polynomial to zero or a small constant.
        if np.sum(itm_mask) < poly_degree + 1:
            # Not enough points for regression.
            # Use a simple average or zero.
            # If very few ITM, the option is likely out of the money or deep OTM.
            # Setting to 0 is safe (exercise if ITM).
            coeffs[j] = np.zeros(poly_degree + 1)
            continue
        
        S_itm = S_j[itm_mask]
        target_itm = target[itm_mask]
        
        # Fit polynomial
        # Use np.polyfit
        # Ensure stability
        try:
            c = np.polyfit(S_itm, target_itm, poly_degree)
            coeffs[j] = c
        except Exception:
            coeffs[j] = np.zeros(poly_degree + 1)

    # If we broke out of the loop early due to deadline, fill remaining with zeros (exercise if ITM)
    for j in range(n_ex - 2, -1, -1):
        if coeffs[j] is None:
            coeffs[j] = np.zeros(poly_degree + 1)
        else:
            break

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, coeffs, exercise_indices, exercise_times, K, n_exercise):
            self.coeffs = coeffs
            self.exercise_indices = exercise_indices
            self.exercise_times = exercise_times
            self.K = K
            self.n_exercise = n_exercise
        
        def _get_threshold_or_value(self, S, time_val):
            """
            Determine exercise decision for a given state S and time.
            """
            # Find the closest exercise index
            # time_val is a scalar or array
            if np.isscalar(time_val):
                # Find closest index
                idx = np.argmin(np.abs(self.exercise_times - time_val))
                # Get coefficients for this date
                c = self.coeffs[idx]
                if c is None:
                    # Last date: exercise if ITM
                    return S < self.K
                else:
                    # Evaluate continuation value
                    cont_val = np.polyval(c, S)
                    # Exercise if Intrinsic > Continuation
                    # Intrinsic = max(K - S, 0)
                    # Since we only care about ITM for puts in the regression context,
                    # but for OTM, Intrinsic is 0, and Cont is >= 0 (usually), so we don't exercise.
                    # So: Exercise if (K - S) > cont_val AND S < K
                    # Which is equivalent to: S < K AND (K - S) > cont_val
                    return (S < self.K) & (self.K - S > cont_val)
            else:
                # Vectorized
                # time_val is an array
                # Find closest index for each time
                # Use broadcasting
                # self.exercise_times shape (n_ex,)
                # time_val shape (N,)
                # diff shape (N, n_ex)
                diffs = np.abs(self.exercise_times[None, :] - time_val[:, None])
                idxs = np.argmin(diffs, axis=1)
                
                # Initialize result
                exercise = np.zeros_like(S, dtype=bool)
                
                # Process each unique index? Or just vectorize per index?
                # Since n_ex is small (12), we can loop over indices
                for k in range(self.n_exercise):
                    mask = idxs == k
                    if not np.any(mask):
                        continue
                    
                    S_subset = S[mask]
                    c = self.coeffs[k]
                    
                    if c is None:
                        exercise[mask] = S_subset < self.K
                    else:
                        cont_val = np.polyval(c, S_subset)
                        exercise[mask] = (S_subset < self.K) & (self.K - S_subset > cont_val)
                
                return exercise

        def exercise(self, current_state, time_val):
            """
            Determine whether to exercise at the given time.
            
            Args:
                current_state: array of stock prices
                time_val: current time (scalar or array)
            
            Returns:
                Boolean array indicating whether to exercise
            """
            if isinstance(current_state, (list, tuple)):
                current_state = np.array(current_state)
            
            return self._get_threshold_or_value(current_state, time_val)
    
    return ExercisePolicy(coeffs, exercise_indices, exercise_times, K, n_exercise)

# EVOLVE-BLOCK-END
