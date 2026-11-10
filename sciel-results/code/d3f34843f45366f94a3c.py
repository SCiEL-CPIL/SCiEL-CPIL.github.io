import numpy as np
import time
import math
from numpy.polynomial import chebyshev as C

class ExercisePolicy:
    def __init__(self, coeffs, exercise_times):
        self.coeffs = coeffs
        self.exercise_times = exercise_times
        self.n_ex = len(exercise_times)
        self.T = exercise_times[-1] if len(exercise_times) > 0 else 1.0
        
    def exercise(self, current_state, time):
        T = self.T
        # Map time to exercise index
        # Exercise times are at T/12, 2T/12, ..., T
        # We need to find which exercise window we are in.
        # The policy is defined *at* the exercise dates.
        # If time is between exercise dates, we don't exercise.
        # However, the interface asks for exercise(current_state, time).
        # Usually, Bermudan options can only be exercised at specific dates.
        # If `time` is not exactly an exercise date, we should return False.
        # But floating point comparison is tricky.
        
        # Let's assume the evaluator calls this *at* the exercise dates or close to them.
        # We find the closest exercise date index.
        diffs = np.abs(self.exercise_times - time)
        k = np.argmin(diffs)
        
        # If the time is not close to an exercise date, do not exercise.
        if diffs[k] > 1e-4:
            return np.zeros_like(current_state, dtype=bool)
            
        if k >= self.n_ex:
            return np.zeros_like(current_state, dtype=bool)

        # Evaluate the continuation value using Chebyshev polynomials
        # We need to map S to the Chebyshev domain [-1, 1].
        # Let's assume a reasonable range for S, e.g., [0, 200] or based on training data min/max.
        # Since we don't have training data min/max in the class, we'll use a fixed reasonable range for GBM with S0=100, sigma=0.2, T=1.
        # 95% interval for S_T is approx 100 * exp(-0.5*0.04) * exp(+-1.96*0.2) ~ [76, 133].
        # Let's use [0, 250] to be safe.
        a, b = 0.0, 250.0
        # Map S to [-1, 1]
        x = 2.0 * (current_state - a) / (b - a) - 1.0
        
        # Evaluate polynomial
        # C.chebval(x, coeffs)
        # coeffs[k] contains the coefficients for the k-th exercise date
        cont_val = C.chebval(x, self.coeffs[k])
        
        # Payoff
        K = 100.0
        payoff = np.maximum(K - current_state, 0.0)
        
        # Exercise if payoff > continuation value
        return payoff > cont_val

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_ex = 12
    
    if hasattr(contract, 'params'):
        params = contract.params
        if isinstance(params, dict):
            K = params.get('K', K)
            r = params.get('r', r)
            sigma = params.get('sigma', sigma)
            T = params.get('T', T)
            n_ex = params.get('n_ex', n_ex)
    elif isinstance(contract, dict):
        K = contract.get('K', K)
        r = contract.get('r', r)
        sigma = contract.get('sigma', sigma)
        T = contract.get('T', T)
        n_ex = contract.get('n_ex', n_ex)

    # Exercise times
    exercise_times = np.linspace(T/n_ex, T, n_ex)
    
    # Assume training_paths shape is (n_paths, n_steps + 1) or similar
    # The contract says "4096 training paths".
    # We need to align the paths with the exercise times.
    # If paths are simulated at every month, there are 13 time points (0 to 12).
    # Exercise dates are indices 1 to 12.
    
    n_paths = training_paths.shape[0]
    # Determine number of time steps in training paths
    # If T=1 and n_ex=12, likely 13 columns (t=0, 1/12, ..., 1)
    n_steps = training_paths.shape[1] - 1
    
    # If n_steps doesn't match n_ex, we might need to interpolate or assume monthly steps.
    # Given "12 monthly exercise dates", it is highly likely n_steps == 12.
    
    # Discount factor per step
    dt = T / n_steps
    disc = np.exp(-r * dt)
    
    # Initialize value vector at the final time step (T)
    # At T, value is payoff
    S_T = training_paths[:, -1]
    V = np.maximum(K - S_T, 0.0).copy()
    
    # Store coefficients for each exercise date
    # coeffs[k] will be the coefficients for the polynomial at exercise date k
    # We iterate backwards from k = n_ex - 1 (last date) to k = 0 (first date)
    coeffs = [None] * n_ex
    
    # Chebyshev domain mapping
    # We need to map S to [-1, 1].
    # Let's use the min and max of the training paths at each step to define the domain, 
    # or a fixed domain. Fixed domain is safer for generalization to evaluation paths.
    # Let's use [0, 250] as a reasonable bound for S0=100, sigma=0.2, T=1.
    a, b = 0.0, 250.0
    
    # Number of Chebyshev terms
    # Too many terms leads to overfitting. 
    # With 4096 paths, we can use a moderate number of terms.
    # Let's try 10-20 terms.
    n_terms = 15
    
    for k in range(n_ex - 1, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Current time step index in the array
        # Exercise date k corresponds to time (k+1)*T/n_ex
        # In the array, index is k+1
        idx = k + 1
        
        S_current = training_paths[:, idx]
        
        # Identify paths that have not been exercised yet
        # We need to track which paths are "alive".
        # Initialize alive mask as all True
        # We will update this mask as we go backwards.
        # Wait, we need to know which paths were exercised at later dates to exclude them from regression at earlier dates.
        # Let's maintain an `alive` mask.
        
        # Actually, we can just use the V vector which holds the optimal value for each path at the *next* step.
        # But we need to regress on the *current* step's S for the paths that are alive.
        
        # Let's maintain an `alive` mask.
        # Start with all alive at T.
        # At step k, the paths that are alive are those that were NOT exercised at steps > k.
        
        # We need to propagate the `alive` mask backwards.
        # Let's initialize `alive` as all True at the end.
        # We'll store the exercise decisions to update the mask.
        
        # This simple loop doesn't easily maintain the mask backwards without storing decisions.
        # Let's store decisions in a list.
        
        # Alternative:
        # V_current holds the value at time t_{k+1}.
        # We regress V_current on S_current for paths that are "alive" at t_k.
        # A path is alive at t_k if it was not exercised at any t_j for j > k.
        
        # Let's re-structure:
        # 1. Initialize V at T.
        # 2. Initialize alive mask as all True.
        # 3. Iterate k from n_ex-1 down to 0.
        # 4. At step k, the "next" value is V (which is value at t_{k+1}).
        # 5. The "current" S is S_current.
        # 6. The paths to consider are those in `alive`.
        # 7. Regress V[alive] on S_current[alive].
        # 8. Compute cont_val = polynomial(S_current[alive]).
        # 9. Compute payoff = max(K - S_current[alive], 0).
        # 10. Exercise if payoff > cont_val.
        # 11. Update V[alive] = max(payoff, cont_val).
        # 12. Update `alive` mask: paths that exercised are no longer alive for previous steps.
        #     alive[alive_mask] = alive[alive_mask] & ~exercise_decision
        
        pass # Placeholder for logic below

    # Let's implement the loop properly with the alive mask.
    V = np.maximum(K - training_paths[:, -1], 0.0).copy()
    alive = np.ones(n_paths, dtype=bool)
    coeffs = [None] * n_ex
    
    for k in range(n_ex - 1, -1, -1):
        if time.time() > deadline - 5.0:
            # Fill remaining coeffs with zeros or simple heuristic
            for i in range(k, -1, -1):
                if coeffs[i] is None:
                    coeffs[i] = np.zeros(n_terms)
            break
            
        idx = k + 1
        S_current = training_paths[:, idx]
        
        # Mask of paths that are currently alive
        current_alive = alive.copy()
        
        if np.sum(current_alive) < 10:
            # Not enough data, use simple boundary
            # Estimate boundary as median of S where payoff > 0?
            # Or just don't exercise
            coeffs[k] = np.zeros(n_terms)
            continue
            
        S_alive = S_current[current_alive]
        V_next = V[current_alive]
        
        # Map S to Chebyshev domain [-1, 1]
        x = 2.0 * (S_alive - a) / (b - a) - 1.0
        
        # Build Chebyshev basis matrix
        # T_0(x), T_1(x), ..., T_{n_terms-1}(x)
        # Use numpy.polynomial.chebyshev.chebvander
        X = C.chebvander(x, n_terms - 1)
        
        # Solve least squares: X @ coeff = V_next
        # Use np.linalg.lstsq
        try:
            coeff, residuals, rank, s = np.linalg.lstsq(X, V_next, rcond=None)
            coeffs[k] = coeff
        except np.linalg.LinAlgError:
            coeffs[k] = np.zeros(n_terms)
            
        # Evaluate continuation value for all alive paths
        # Note: We regress on V_next which is the value at t_{k+1}.
        # The continuation value at t_k is disc * E[V_{t_{k+1}} | S_{t_k}]
        # So we must discount the regression result.
        cont_val = C.chebval(x, coeffs[k]) * disc
        
        # Payoff
        payoff = np.maximum(K - S_alive, 0.0)
        
        # Exercise decision
        exercise_mask = payoff > cont_val
        
        # Update V for these paths
        V_current_vals = np.maximum(payoff, cont_val)
        
        # Update the main V vector
        V[current_alive] = V_current_vals
        
        # Update alive mask for the previous step
        # Paths that exercised are no longer alive
        alive[current_alive] = alive[current_alive] & ~exercise_mask

    return ExercisePolicy(coeffs, exercise_times)
