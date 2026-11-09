"""Bermudan put exercise policy via AOMC (Longstaff-Schwartz) with polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # Get training paths
    # training_paths is expected to be an array of shape (n_paths, n_steps+1) or similar
    # We need to figure out the structure. Let's assume it's (n_paths, n_time_steps)
    # where n_time_steps includes time 0 through T.
    
    # Let's inspect the training_paths structure
    # The problem says "4096 training paths"
    # Let's assume training_paths is a 2D array: (n_paths, n_dates)
    # n_dates should be 13 (time 0 to 12) or 12 (time 1 to 12)
    
    tp = np.asarray(training_paths)
    
    # Determine dimensions
    # If tp.shape[1] == 13, it includes time 0
    # If tp.shape[1] == 12, it starts at time 1
    
    n_paths, n_cols = tp.shape
    
    # Check if first column is S0 (all ~S0)
    if n_cols >= 13:
        # Assume first column is time 0
        # Exercise dates are columns 1 through 12
        exercise_cols = np.arange(1, 13)
    elif n_cols == 12:
        # Exercise dates are columns 0 through 11
        exercise_cols = np.arange(0, 12)
    else:
        # Fallback: assume last 12 columns are exercise dates
        exercise_cols = np.arange(n_cols - 12, n_cols)
    
    # Extract states at exercise dates
    # States at exercise dates: shape (n_paths, 12)
    states = tp[:, exercise_cols]
    
    # Payoff at each exercise date for a put: max(K - S, 0)
    payoffs = np.maximum(K - states, 0.0)
    
    # Discount factors
    # Discount from time t to time 0: exp(-r * t)
    # But in AOMC, we work backwards from the last exercise date
    # The value at time t is the expected discounted future value
    
    # AOMC algorithm:
    # 1. Start from the last exercise date (t=12)
    # 2. At t=12, continuation value = 0 (no future exercise)
    #    So V_12 = payoff_12
    # 3. For t=11 down to 1:
    #    - Regress the discounted continuation value (V_{t+1} discounted) on states at time t
    #    - Continuation value at t = max(payoff_t, E[discounted V_{t+1}])
    #    - But in AOMC, we use the regression to estimate the continuation value
    
    # More precisely:
    # Let V_t be the value at time t
    # V_12 = payoff_12
    # For t = 11, ..., 1:
    #   Discounted future value = exp(-r*dt) * V_{t+1}
    #   Regress exp(-r*dt) * V_{t+1} on states at time t using a basis
    #   Continuation value at t = max(payoff_t, regressed value)
    #   V_t = continuation value at t
    
    # Wait, that's not quite right. Let me think again.
    
    # In AOMC:
    # At each exercise date t, the holder decides whether to exercise or continue.
    # If they continue, the value is the expected discounted value of V_{t+1}.
    # So: V_t = max(payoff_t, E_t[exp(-r*dt) * V_{t+1}])
    
    # We estimate E_t[exp(-r*dt) * V_{t+1}] using regression on the state at time t.
    
    # Basis functions: use polynomials in log(S) or S
    # Let's use a polynomial basis in S
    # Degree 5 polynomial: [1, S, S^2, S^3, S^4, S^5]
    
    def make_basis(S, degree=5):
        """Create polynomial basis for S."""
        S = np.asarray(S)
        if S.ndim == 1:
            S = S[:, np.newaxis]
        basis = np.ones((S.shape[0], 1))
        for d in range(1, degree + 1):
            basis = np.column_stack([basis, S**d])
        return basis
    
    # AOMC with polynomial regression
    n_basis = 6  # degree 5 polynomial: 6 basis functions
    
    # V at last exercise date
    V_next = payoffs[:, -1].copy()  # shape (n_paths,)
    
    # Exercise decisions: shape (n_paths, 12)
    exercise_decisions = np.zeros((n_paths, 12), dtype=bool)
    exercise_decisions[:, -1] = payoffs[:, -1] > 0  # Always exercise if in the money at last date
    
    # Work backwards
    discount = np.exp(-r * dt)
    
    for t in range(n_exercise - 2, -1, -1):
        if time.time() > deadline:
            break
        
        # Regress discount * V_next on states at time t
        S_t = states[:, t]  # shape (n_paths,)
        
        # Only use in-the-money paths for regression (standard AOMC practice)
        in_the_money = payoffs[:, t] > 0
        if np.sum(in_the_money) < n_basis + 1:
            # Not enough in-the-money paths, use all paths
            in_the_money = np.ones(n_paths, dtype=bool)
        
        S_t_itm = S_t[in_the_money]
        V_next_itm = V_next[in_the_money]
        
        if len(S_t_itm) < n_basis:
            # Fallback: just use the mean
            cont_value = np.mean(discount * V_next)
            exercise_decisions[:, t] = payoffs[:, t] > cont_value
            V_next = np.maximum(payoffs[:, t], discount * V_next)
            continue
        
        # Create basis
        basis = make_basis(S_t_itm, degree=5)
        
        # Regress discount * V_next on basis
        target = discount * V_next_itm
        try:
            # Use least squares
            coef, residuals, rank, sv = np.linalg.lstsq(basis, target, rcond=None)
        except np.linalg.LinAlgError:
            coef = np.zeros(n_basis)
        
        # Compute continuation value for all paths at time t
        basis_all = make_basis(S_t, degree=5)
        cont_value_all = basis_all @ coef
        
        # Exercise decision
        exercise_decisions[:, t] = payoffs[:, t] >= cont_value_all
        
        # Value at time t
        V_next = np.maximum(payoffs[:, t], cont_value_all)
    
    # Now we have exercise_decisions for each path at each exercise date
    # We need to create a policy that can be applied to new states
    
    # The policy is: exercise at time t if payoff >= cont_value
    # But we need to estimate cont_value as a function of S
    
    # Let's re-do the regression to get the coefficients for each time step
    # so we can evaluate the continuation value for any state
    
    coef_by_time = [None] * n_exercise
    coef_by_time[-1] = None  # At last date, always exercise if ITM
    
    for t in range(n_exercise - 2, -1, -1):
        if time.time() > deadline:
            break
        
        S_t = states[:, t]
        in_the_money = payoffs[:, t] > 0
        if np.sum(in_the_money) < n_basis + 1:
            in_the_money = np.ones(n_paths, dtype=bool)
        
        S_t_itm = S_t[in_the_money]
        
        if len(S_t_itm) < n_basis:
            coef_by_time[t] = None
            continue
        
        # We need V_next from the previous iteration
        # Let's re-compute V_next properly
        
    # Actually, let me restructure to store the coefficients properly
    
    # Re-do AOMC storing coefficients
    V_next = payoffs[:, -1].copy()
    coef_by_time = [None] * n_exercise
    
    for t in range(n_exercise - 2, -1, -1):
        if time.time() > deadline:
            break
        
        S_t = states[:, t]
        in_the_money = payoffs[:, t] > 0
        if np.sum(in_the_money) < n_basis + 1:
            in_the_money = np.ones(n_paths, dtype=bool)
        
        S_t_itm = S_t[in_the_money]
        V_next_itm = V_next[in_the_money]
        
        if len(S_t_itm) < n_basis:
            coef_by_time[t] = None
            cont_value_all = np.mean(discount * V_next)
        else:
            basis = make_basis(S_t_itm, degree=5)
            target = discount * V_next_itm
            try:
                coef, residuals, rank, sv = np.linalg.lstsq(basis, target, rcond=None)
            except np.linalg.LinAlgError:
                coef = np.zeros(n_basis)
            coef_by_time[t] = coef
            
            basis_all = make_basis(S_t, degree=5)
            cont_value_all = basis_all @ coef
        
        V_next = np.maximum(payoffs[:, t], cont_value_all)
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, coef_by_time, K, r, dt, n_exercise):
            self.coef_by_time = coef_by_time
            self.K = K
            self.r = r
            self.dt = dt
            self.n_exercise = n_exercise
            self.discount = np.exp(-r * dt)
        
        def exercise(self, current_state, time):
            """
            current_state: array of current stock prices, shape (n_eval_paths,)
            time: current time index (0-indexed, so time 0 is first exercise date)
            Returns: boolean vector indicating whether to exercise
            """
            if time >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            if time == self.n_exercise - 1:
                # Last exercise date: exercise if in the money
                return current_state < self.K
            
            coef = self.coef_by_time[time]
            if coef is None:
                # Fallback: exercise if in the money
                return current_state < self.K
            
            # Compute continuation value
            basis = make_basis(current_state, degree=5)
            cont_value = basis @ coef
            
            # Exercise if payoff >= continuation value
            payoff = np.maximum(self.K - current_state, 0.0)
            return payoff >= cont_value
    
    return ExercisePolicy(coef_by_time, K, r, dt, n_exercise)

# EVOLVE-BLOCK-END
