"""Bermudan exercise policy using Longstaff-Schwartz regression-based backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P

def fit_exercise_policy(training_paths, contract):
    """
    Fits a Bermudan exercise policy using Longstaff-Schwartz method.
    
    Args:
        training_paths: Array of shape (N, n_steps) containing simulated spot prices.
                        Typically n_steps includes the initial time or just exercise dates.
                        Based on "12 monthly exercise dates (no time-0 exercise)", 
                        we assume training_paths has 12 columns corresponding to t_1..t_12.
                        If it has 13, we ignore the first.
        contract: Dictionary containing contract parameters (S0, K, r, sigma, T, n_dates).
    
    Returns:
        An object with an `exercise(current_state, time)` method.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Determine time step
    dt = T / n_dates
    discount_factor = np.exp(-r * dt)

    # Prepare data
    # training_paths is expected to be (N, n_dates) or (N, n_dates+1).
    # The prompt says "12 monthly exercise dates (no time-0 exercise)".
    # Let's assume the columns correspond to the exercise dates t_1 to t_12.
    # If the input has 13 columns, we drop the first (t_0).
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        paths = paths.reshape(-1, n_dates)
        
    if paths.shape[1] > n_dates:
        # If there are more columns than exercise dates, assume the first is t=0
        paths = paths[:, 1:n_dates+1]
    elif paths.shape[1] < n_dates:
        # Pad if necessary? Unlikely. Assume it matches.
        pass

    N_paths = paths.shape[0]
    
    # Storage for regression coefficients for each time step
    # coeffs[t] will be the polynomial coefficients for the continuation value at time t
    # We fit for t = n_dates-1 down to 0 (indexing 0-based for exercise dates 1..12)
    # Let index i correspond to exercise date i+1.
    # We need continuation values for dates 1..11. Date 12 is maturity.
    
    # Initialize continuation values for the last step (maturity)
    # Value at maturity is the payoff
    # We will store the fitted polynomial coefficients for the continuation value 
    # which is the expected discounted future value.
    
    # coeffs[i] corresponds to the continuation value at exercise date i+1
    # We fit continuation value for i = n_dates-2 down to 0.
    # For i = n_dates-1 (last date), continuation value is 0 (maturity).
    
    poly_coeffs = [None] * n_dates
    
    # Backward induction
    # Start from the second to last date
    # Value at last date (index n_dates-1) is the payoff
    # We need to fit the continuation value at date i based on the realized value at date i+1.
    
    # Let V_realized[i] be the value of the option at date i given optimal exercise from i to end.
    # We compute this backward.
    
    # Step 1: Last date (index n_dates-1)
    # The value is the immediate payoff, as there is no future.
    # But for the previous step, we need to regress the *discounted* value of the next step.
    
    # Let's define:
    # C_i(S_i) = E[ V_{i+1} | S_i ] * discount_factor
    # V_i(S_i) = max( Payoff_i(S_i), C_i(S_i) )
    
    # We need to fit C_i for i = 0 to n_dates-2.
    
    # Value at the last exercise date (index n_dates-1)
    # V_last = max(K - S_last, 0)
    S_last = paths[:, n_dates-1]
    V_last = np.maximum(K - S_last, 0.0)
    
    # We will store the realized values V_i for each path to use in the previous regression
    # Actually, we only need the values at i+1 to regress for i.
    
    current_values = V_last.copy()
    
    # Loop backward from n_dates-2 to 0
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            # Emergency break, use simple heuristic for remaining steps
            break
            
        S_i = paths[:, i]
        
        # The target for regression is the discounted value of the next step
        # The next step's value is current_values (which is V_{i+1})
        # Discounted target = discount_factor * current_values
        
        target = discount_factor * current_values
        
        # Fit polynomial to S_i
        # Use degree 4 for good balance of fit and stability
        # We need to handle potential numerical issues
        try:
            # polyfit returns coefficients from lowest degree to highest
            # We fit target = c0 + c1*S + c2*S^2 + ...
            coeffs = P.polyfit(S_i, target, deg=4)
        except Exception:
            # Fallback to linear if polyfit fails
            coeffs = P.polyfit(S_i, target, deg=1)
            
        poly_coeffs[i] = coeffs
        
        # Calculate estimated continuation value for all paths at step i
        C_i = P.polyval(S_i, coeffs)
        
        # Immediate payoff at step i
        Payoff_i = np.maximum(K - S_i, 0.0)
        
        # Optimal value at step i is max(Payoff_i, C_i)
        V_i = np.maximum(Payoff_i, C_i)
        
        # Set current_values to V_i for the next iteration (i-1)
        current_values = V_i
        
    # Create the policy object
    class BermudanExercisePolicy:
        def __init__(self, coeffs, K, r, dt, n_dates):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.dt = dt
            self.n_dates = n_dates
            self.discount_factor = np.exp(-r * dt)

        def exercise(self, current_state, time):
            """
            Determines whether to exercise the option.
            
            Args:
                current_state: Array of spot prices at the current time.
                time: The index of the current exercise date (0-based, 0..n_dates-1).
                       Corresponds to t_{time+1}.
            
            Returns:
                Boolean array indicating whether to exercise (True) or hold (False).
            """
            # If it's the last date, always exercise if in the money
            # Actually, at the last date, continuation value is 0.
            # So exercise if Payoff > 0.
            if time >= self.n_dates - 1:
                return np.maximum(self.K - current_state, 0.0) > 0.0
            
            # Get coefficients for this time step
            if self.coeffs[time] is None:
                # If we didn't fit this step (e.g., due to timeout), fallback to simple rule
                # Simple rule: Exercise if S < K * 0.9 (rough heuristic)
                return current_state < self.K * 0.9
            
            # Calculate estimated continuation value
            C = P.polyval(current_state, self.coeffs[time])
            
            # Calculate immediate payoff
            Payoff = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if Payoff > Continuation Value
            # Add a small epsilon to avoid floating point issues where they are equal
            return Payoff > C + 1e-9

    return BermudanExercisePolicy(poly_coeffs, K, r, dt, n_dates)

# EVOLVE-BLOCK-END
