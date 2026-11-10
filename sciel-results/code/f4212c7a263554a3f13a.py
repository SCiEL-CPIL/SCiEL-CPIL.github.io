"""Bermudan exercise-policy research: binomial tree boundaries with interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a high-resolution binomial tree.
    
    Parameters:
    -----------
    training_paths : ndarray of shape (N, 13) where columns are S_0, S_1, ..., S_12
    contract : dict with keys 'K', 'r', 'sigma', 'T', 'num_steps' (optional)
    
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean vector
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin out of 150s

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = contract.get('num_steps', 12)
    
    # Use a fine binomial tree for accurate boundaries
    # 1200 steps gives 100 fine steps per monthly exercise date
    n_fine = 1200
    dt_fine = T / n_fine
    u = math.exp(sigma * math.sqrt(dt_fine))
    d = 1.0 / u
    p = (math.exp(r * dt_fine) - d) / (u - d)
    disc = math.exp(-r * dt_fine)
    
    # Exercise dates in terms of fine tree steps
    # Exercise at steps 100, 200, ..., 1200 (i.e., every 100 fine steps)
    ex_steps = [int(round(i * n_fine / n_exercises)) for i in range(1, n_exercises + 1)]
    
    # We'll work in log-space for the tree
    # S0 reference for tree construction
    S0_ref = 100.0
    log_S0 = math.log(S0_ref)
    log_u = math.log(u)
    
    # Initialize at maturity (step n_fine)
    # Number of nodes at step j is j+1
    # Node i at step j: log_S = log_S0 + (2*i - j) * log_u
    i_terminal = np.arange(n_fine + 1)
    exponents = 2 * i_terminal - n_fine
    log_S_terminal = log_S0 + exponents * log_u
    S_terminal = np.exp(log_S_terminal)
    
    # Payoff at maturity
    V_next = np.maximum(K - S_terminal, 0.0)
    
    # Store critical prices for each exercise date
    # s_crit[k] is the maximum S at which we exercise at time t_k
    s_crit = [0.0] * (n_exercises + 1)
    
    # Track which exercise steps we've passed
    ex_step_set = set(ex_steps)
    
    # Iterate backwards from j = n_fine - 1 down to 0
    for j in range(n_fine - 1, -1, -1):
        # Check deadline
        if time.time() > deadline:
            # Fallback: fill remaining s_crit with K (exercise if ITM)
            for k in range(1, n_exercises + 1):
                if s_crit[k] == 0.0:
                    s_crit[k] = K
            break
        
        # Current step j has nodes i = 0 .. j
        i_curr = np.arange(j + 1)
        exponents_curr = 2 * i_curr - j
        log_S_curr = log_S0 + exponents_curr * log_u
        S_curr = np.exp(log_S_curr)
        
        # Intrinsic Value at current step
        V_int = np.maximum(K - S_curr, 0.0)
        
        # Continuation Value
        # V_cont[i] = disc * (p * V_next[i+1] + (1-p) * V_next[i])
        V_cont = disc * (p * V_next[1:] + (1 - p) * V_next[:-1])
        
        # Determine if this step is an exercise date
        if j in ex_step_set:
            # Find which exercise index k this corresponds to
            k = ex_steps.index(j) + 1  # 1-based
            
            # Find the boundary where V_int >= V_cont
            diff = V_int - V_cont
            exercise_mask = diff >= 0
            
            if np.any(exercise_mask):
                # Find the last index where exercise is optimal
                true_indices = np.where(exercise_mask)[0]
                last_true_idx = true_indices[-1]
                
                # Interpolate the boundary for higher precision
                if last_true_idx < j:
                    d1 = diff[last_true_idx]
                    d2 = diff[last_true_idx + 1]
                    s1 = S_curr[last_true_idx]
                    s2 = S_curr[last_true_idx + 1]
                    
                    if abs(d2 - d1) > 1e-12:
                        alpha = d1 / (d1 - d2)
                        s_crit[k] = s1 + alpha * (s2 - s1)
                    else:
                        s_crit[k] = s1
                else:
                    s_crit[k] = S_curr[last_true_idx]
            else:
                # Never exercise at this date (boundary below lowest node)
                s_crit[k] = 0.0
        
        # Update V for this step
        V_curr = np.where(V_int > V_cont, V_int, V_cont)
        V_next = V_curr
    
    # Fallback for any uncomputed boundaries
    for k in range(1, n_exercises + 1):
        if s_crit[k] == 0.0:
            s_crit[k] = K
    
    # Build the policy object
    class ExercisePolicy:
        def __init__(self, s_crits, n_exercises, T):
            self.s_crits = s_crits
            self.n_exercises = n_exercises
            self.T = T
            self.dt_exercise = T / n_exercises
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the Bermudan put.
            
            Parameters:
            -----------
            current_state : ndarray of shape (M,) - current spot prices
            time : int or float - current time step (1-based, 1 to 12) or time in years
            
            Returns:
            --------
            boolean vector of shape (M,) - True if should exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            M = len(current_state)
            
            if time is None:
                return np.zeros(M, dtype=bool)
            
            # Handle both integer step indices (1-12) and continuous time
            t = float(time)
            
            # If time looks like an integer step (1-12), use it directly
            # Otherwise, convert from continuous time
            if t <= self.n_exercises and abs(t - round(t)) < 1e-9:
                k = int(round(t))
            else:
                # Continuous time: k = ceil(t / dt_exercise)
                k = int(math.ceil(t / self.dt_exercise - 1e-9))
            
            if k < 1:
                return np.zeros(M, dtype=bool)
            if k > self.n_exercises:
                k = self.n_exercises
            
            crit = self.s_crits[k]
            
            if crit <= 0.0:
                return np.zeros(M, dtype=bool)
            
            # Exercise if S <= critical price
            return current_state <= crit
    
    return ExercisePolicy(s_crit, n_exercises, T)

# EVOLVE-BLOCK-END
