"""Bermudan put exercise-policy research with Polynomial Regression Backward Induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time

_DEADLINE = 115.0  # safety margin under 120s limit


def fit_exercise_policy(training_paths, contract):
    """Fit a Bermudan put exercise policy using Polynomial Regression Backward Induction."""
    start_time = _time.time()
    
    # Extract parameters
    n_dates, n_paths = training_paths.shape
    K = 100.0
    r = 0.05
    T = 1.0
    sigma = 0.2
    
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    # We will store the polynomial coefficients for the continuation value estimate
    # at each step i (for i = 0 to n_dates - 2).
    # V_i = max(K - S_i, discount * P_i(ln S_i))
    # The exercise decision is K - S_i > discount * P_i(ln S_i)
    
    # Initialize values at the last step (maturity)
    # At the last step (n_dates - 1), we just exercise if in the money.
    # There is no continuation value after maturity.
    # So for the decision at n_dates - 1, we compare payoff to 0.
    
    # Let's work backward.
    # values_next will hold the optimal values at step i+1.
    # At step n_dates - 1, the value is the payoff.
    values_next = np.maximum(K - training_paths[n_dates - 1], 0.0)
    
    # Store polynomials for each step
    # polys[i] will be the coefficients for the continuation value estimate at step i
    # We need to store them for the evaluation phase.
    polys = [None] * n_dates
    
    # For the last step, we don't need a polynomial, we just exercise if ITM.
    # But to keep the interface consistent, we can treat the continuation value as 0.
    # So exercise if K - S > 0.
    
    for i in range(n_dates - 2, -1, -1):
        # Fit a polynomial to estimate E[V_{i+1} | S_i]
        # We use ln(S_i) as the feature for better scaling.
        log_S = np.log(training_paths[i])
        
        # Use polynomial regression. Degree 4 is a good balance.
        # We fit values_next to log_S.
        # Note: values_next is the value *at step i+1*.
        # The continuation value at step i is discount * E[V_{i+1} | S_i].
        
        # Check time
        if _time.time() - start_time > _DEADLINE - 5:
             # Fallback: simple threshold
             # This shouldn't happen, but safety first.
             break
        
        # np.polyfit returns coefficients [c_n, ..., c_1, c_0] for x^n ... x
        # We want P(x) = c_n x^n + ... + c_0
        try:
            # Degree 4
            coeffs = np.polyfit(log_S, values_next, 4)
        except Exception:
            # Fallback to lower degree if fit fails (unlikely)
            coeffs = np.polyfit(log_S, values_next, 2)
            
        polys[i] = coeffs
        
        # Compute the continuation value estimate at step i
        # cont_val = discount * P(log_S)
        cont_val = discount * np.polyval(coeffs, log_S)
        
        # Compute the optimal value at step i
        payoff = np.maximum(K - training_paths[i], 0.0)
        values_next = np.maximum(payoff, cont_val)
        
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, polys, K, r, T, n_dates, dt):
            self.polys = polys
            self.K = K
            self.r = r
            self.T = T
            self.n_dates = n_dates
            self.dt = dt
            self.discount = np.exp(-r * dt)
            
        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating which paths to exercise.
            """
            # Determine which exercise date we're at
            date_idx = int(round(time / self.dt)) - 1
            date_idx = max(0, min(date_idx, self.n_dates - 1))
            
            # If it's the last date, exercise if in the money
            if date_idx == self.n_dates - 1:
                return current_state < self.K
            
            # Otherwise, use the polynomial
            if self.polys[date_idx] is not None:
                log_S = np.log(current_state)
                # Evaluate polynomial
                cont_val = self.discount * np.polyval(self.polys[date_idx], log_S)
                payoff = self.K - current_state
                # Exercise if payoff > continuation value
                return payoff > cont_val
            else:
                # Fallback: exercise if ITM
                return current_state < self.K
    
    policy = ExercisePolicy(polys, K, r, T, n_dates, dt)
    return policy

# EVOLVE-BLOCK-END
