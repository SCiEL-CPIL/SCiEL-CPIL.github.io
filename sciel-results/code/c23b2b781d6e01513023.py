import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0

    # training_paths shape: (N_paths, n_time_points)
    # n_time_points = n_exercise + 1 = 13 (t=0, 1/12, ..., 1)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # V[i, path] = value at exercise date i for path
    # Indices: 0..n_exercise
    # Exercise dates correspond to indices 1..n_exercise in training_paths
    # But in our V array, let's align indices such that V[j] is value at exercise date j (1..12)
    # Or simpler: V is size n_exercise+1. V[n_exercise] is maturity.
    # V[i] is value at step i.
    # Step i corresponds to training_paths column i+1?
    # Let's stick to the previous indexing:
    # V[k] is the value at the k-th exercise date?
    # Previous code:
    # V[n_exercise] = payoff at maturity (col n_exercise)
    # Loop i from n_exercise - 1 down to 0.
    # S_i = training_paths[:, i+1]
    # This implies V[i] is value at time step i+1 (1..12).
    # So V[11] is value at t=1 (maturity).
    # V[10] is value at t=11/12.
    # ...
    # V[0] is value at t=1/12.
    
    # Let's re-verify:
    # training_paths has 13 columns: 0, 1, ..., 12.
    # Col 0: t=0
    # Col 1: t=1/12 (Exercise 1)
    # ...
    # Col 12: t=1 (Exercise 12, Maturity)
    
    # Previous code:
    # V = np.zeros((n_exercise + 1, N)) # Size 13
    # V[n_exercise] = payoff at col n_exercise (12). Correct.
    # Loop i in range(n_exercise - 1, -1, -1): # 11 down to 0
    # S_i = training_paths[:, i+1]
    # If i=11, S_i is col 12. Wait.
    # If i=11, S_i = col 12. But V[11] is already set to payoff at col 12.
    # The loop calculates V[i] based on V[i+1].
    # If i=11, it calculates V[11] based on V[12].
    # But V[12] doesn't exist in the previous logic?
    # Previous code: V = np.zeros((n_exercise + 1, N)) -> indices 0..12.
    # V[n_exercise] is V[12].
    # V[12] = payoff at col 12.
    # Loop i = 11: S_i = col 12? No, `training_paths[:, i+1]` -> col 12.
    # So at i=11, we are at Maturity?
    # But we already set V[12] to the payoff.
    # Then we calculate V[11] using V[12].
    # S_i for i=11 is col 12.
    # This implies V[11] is the value *just before* maturity? No, that's not right.
    
    # Let's trace carefully.
    # Exercise dates: 1..12.
    # Maturity is at date 12.
    # At date 12, you must exercise (or value is payoff).
    # At date 11, you can exercise or continue to date 12.
    
    # If we define V[k] as the value *at* exercise date k:
    # V[12] = Payoff(S_12)
    # V[11] = max(Payoff(S_11), E[discount * V[12] | S_11])
    # ...
    # V[1] = max(Payoff(S_1), E[discount * V[2] | S_1])
    
    # In the previous code:
    # V[n_exercise] (V[12]) = Payoff at col 12.
    # Loop i = 11:
    #   S_i = training_paths[:, 12] ?? No, `training_paths[:, i+1]` is `training_paths[:, 12]`.
    #   This is WRONG.
    #   If i=11, we are calculating V[11].
    #   The current state for V[11] is at time 11/12, which is column 11.
    #   So S_i should be `training_paths[:, 11]`.
    #   The formula `training_paths[:, i+1]` gives col 12 for i=11.
    #   So the previous code was calculating V[11] using the state at Maturity?
    #   That would be wrong.
    
    # Let's check the previous code again.
    # `V[n_exercise] = np.maximum(K - training_paths[:, n_exercise], 0)`
    # `n_exercise` is 12. So V[12] uses col 12.
    # Loop `for i in range(n_exercise - 1, -1, -1):` -> i = 11, 10, ..., 0.
    # `S_i = training_paths[:, i + 1]`
    # If i=11, S_i is col 12.
    # `exercise_payoff = max(K - S_i, 0)` -> Payoff at Maturity.
    # `cont_values` estimated from V[12].
    # `V[11] = max(exercise_payoff, cont_values)`
    # This calculates the value at "Date 12" again?
    # No, V[11] is stored in index 11.
    # But the state used is from col 12.
    # This means V[11] is actually the value at Maturity, not at 11/12.
    # This looks like an off-by-one error in the previous code.
    
    # Correct mapping:
    # Exercise Date k (1-indexed) corresponds to column k.
    # Value at Date k, V[k], depends on State S_k (col k) and Future Value V[k+1].
    # V[12] = Payoff(S_12).
    # V[11] = max(Payoff(S_11), E[disc * V[12] | S_11]).
    # ...
    # V[1] = max(Payoff(S_1), E[disc * V[2] | S_1]).
    
    # So we need V array of size 13 (0..12).
    # V[12] = Payoff(col 12).
    # For k from 11 down to 1:
    #   S_k = col k.
    #   V[k] = max(Payoff(S_k), E[disc * V[k+1] | S_k]).
    #
    # The policy is applied at dates 1..12.
    # The `exercise` function receives `time`.
    # If `time` is date index (1..12) or continuous time.
    # We need to return decision for that date.
    # Decision at date k: Exercise if Payoff(S_k) >= E[disc * V[k+1] | S_k].
    # This is equivalent to: Exercise if S_k <= Threshold_k.
    # Or using regression: Exercise if Payoff(S_k) >= Regressed_Cont_Value(S_k).

    # Let's implement this correctly with Polynomial Regression.

    V = np.zeros((n_exercise + 1, N))
    # Maturity
    V[n_exercise] = np.maximum(K - training_paths[:, n_exercise], 0)
    
    # We will store the regression coefficients for each step to use in the policy
    # Or simply store the thresholds.
    # Thresholds are simpler and faster for evaluation.
    
    thresholds = np.zeros(n_exercise) # For dates 1..12, store in 0..11
    
    # Degree of polynomial
    deg = 4
    
    for k in range(n_exercise - 1, 0, -1): # k from 11 down to 1
        S_k = training_paths[:, k]
        # V[k+1] is the value at the next step
        # We need to estimate C(S) = E[discount * V[k+1] | S_k]
        # Fit polynomial P(S) of degree deg to (S_k, discount * V[k+1])
        
        y = discount * V[k+1]
        
        # Normalize S to improve conditioning? 
        # S is around 100. S^4 is 100,000,000. 
        # Better to normalize S to [0, 1] or center it.
        S_norm = (S_k - np.mean(S_k)) / (np.std(S_k) + 1e-8)
        
        # Build Vandermonde matrix
        X = np.vander(S_norm, N=deg+1, increasing=True)
        
        # Solve least squares
        # np.linalg.lstsq(X, y, rcond=None)
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(deg+1)
            
        # Evaluate continuation value
        cont_values = np.polyval(coeffs, S_norm)
        
        # Ensure non-negative? Continuation value of a put is non-negative.
        # But regression might go negative.
        # max(0, cont) is a good idea.
        cont_values = np.maximum(cont_values, 0)
        
        exercise_payoff = np.maximum(K - S_k, 0)
        
        V[k] = np.maximum(exercise_payoff, cont_values)
        
        # Determine threshold
        # We want to find S such that K - S = P(S)
        # Since P(S) is convex (usually) and K-S is linear, they cross once.
        # We can find the threshold by scanning the sorted S_k.
        
        # Alternative: Just store the decision logic?
        # The policy object needs to be fast.
        # Threshold is O(1).
        
        order = np.argsort(S_k)
        S_sorted = S_k[order]
        ex_sorted = exercise_payoff[order]
        cont_sorted = cont_values[order]
        
        # Exercise if ex >= cont
        # Since ex is decreasing and cont is increasing (for puts, cont increases with S),
        # there is a single crossing.
        # Find the largest S where ex >= cont.
        
        should_exercise = ex_sorted >= cont_sorted
        if np.any(should_exercise):
            last_true = np.where(should_exercise)[0][-1]
            thresholds[k-1] = S_sorted[last_true]
        else:
            thresholds[k-1] = 0.0

        if time.time() > deadline - 5:
            break

    class Policy:
        def __init__(self, thresholds, n_exercise):
            self.thresholds = thresholds
            self.n_exercise = n_exercise

        def exercise(self, current_state, time):
            if isinstance(time, (int, np.integer)):
                idx = int(time)
                # If time is 1..12, map to 0..11
                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                idx = idx - 1
            else:
                # Continuous time
                # Map to date index 1..12
                # t = k/12
                # k = round(t * 12) ?
                # If t=1/12, k=1.
                # If t=1.0, k=12.
                # Use floor(t*12 + 0.5) to round?
                # Or just floor(t*12) if we assume exact steps?
                # Task says "12 monthly exercise dates".
                # If time is continuous, we assume it hits the date exactly or close.
                # Let's round to nearest integer month.
                k = int(np.round(time * 12))
                if k < 1 or k > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)
                idx = k - 1

            thresh = self.thresholds[idx]
            return current_state <= thresh

    return Policy(thresholds, n_exercise)
