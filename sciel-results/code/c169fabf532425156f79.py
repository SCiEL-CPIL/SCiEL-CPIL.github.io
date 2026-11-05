"""Bermudan put exercise policy via refined binomial tree backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using binomial tree backward induction
    with linear interpolation for boundary refinement.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 100.0  # Safety margin before 120s kill

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Build a CRR binomial tree
    # 2000 steps is a good balance of accuracy and speed
    n_tree_steps = 2000
    dt_tree = T / n_tree_steps
    
    # CRR parameters
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = np.exp(-sigma * np.sqrt(dt_tree))
    p = (np.exp(r * dt_tree) - d) / (u - d)
    discount_tree = np.exp(-r * dt_tree)
    
    # Determine S0
    S0 = np.median(training_paths[:, 0])
    
    log_S0 = np.log(S0)
    log_u = np.log(u)
    log_d = np.log(d)
    
    # Exercise dates in tree steps
    exercise_tree_steps = [int(round(n_tree_steps * i / n_exercise)) for i in range(1, n_exercise + 1)]
    
    # Initialize the value at maturity
    j_max = n_tree_steps
    j_vals = np.arange(j_max + 1)
    log_S_final = log_S0 + n_tree_steps * log_d + j_vals * (log_u - log_d)
    S_final = np.exp(log_S_final)
    
    # Value at maturity: put payoff
    V = np.maximum(K - S_final, 0.0)
    
    # Boundaries for each exercise date
    boundaries = [None] * n_exercise
    
    # Map from tree step to exercise index
    step_to_ex_idx = {}
    for k, ts in enumerate(exercise_tree_steps):
        step_to_ex_idx[ts] = k
    
    # Backward induction
    for i in range(n_tree_steps - 1, -1, -1):
        # Continuation value
        # V has length i+2
        V_cont = discount_tree * (p * V[1:] + (1.0 - p) * V[:-1])
        
        # Stock prices at current step
        j_curr = np.arange(i + 1)
        log_S_curr = log_S0 + i * log_d + j_curr * (log_u - log_d)
        S_curr = np.exp(log_S_curr)
        
        # Payoff at current step
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Exercise if payoff > continuation
        exercise_mask = payoff > V_cont
        
        # Check if this is an exercise date
        if i in step_to_ex_idx:
            ex_idx = step_to_ex_idx[i]
            if np.any(exercise_mask):
                # Find the boundary
                # We want the largest S where exercise is optimal.
                # Find the last index where exercise_mask is True
                ex_indices = np.where(exercise_mask)[0]
                if len(ex_indices) > 0:
                    j_ex = ex_indices[-1]
                    
                    # If j_ex is the last node, boundary is S_curr[j_ex]
                    # Otherwise, interpolate between j_ex and j_ex+1
                    if j_ex < i:
                        S1 = S_curr[j_ex]
                        S2 = S_curr[j_ex + 1]
                        
                        # Values at these points
                        # At S1: Exercise is better. V_ex1 = payoff[S1], V_cont1 = V_cont[j_ex]
                        # At S2: Continue is better. V_ex2 = payoff[S2], V_cont2 = V_cont[j_ex+1]
                        
                        V_ex1 = payoff[j_ex]
                        V_cont1 = V_cont[j_ex]
                        
                        V_ex2 = payoff[j_ex + 1]
                        V_cont2 = V_cont[j_ex + 1]
                        
                        # We want S* where V_cont(S*) = V_ex(S*) = K - S*
                        # Linear interpolation for V_cont:
                        # V_cont(S) = V_cont1 + (V_cont2 - V_cont1) * (S - S1) / (S2 - S1)
                        # Payoff(S) = K - S
                        
                        # V_cont1 + slope_cont * (S - S1) = K - S
                        # S * (1 + slope_cont) = K - V_cont1 + slope_cont * S1
                        # S = (K - V_cont1 + slope_cont * S1) / (1 + slope_cont)
                        
                        slope_cont = (V_cont2 - V_cont1) / (S2 - S1)
                        
                        # Check for numerical stability
                        if abs(1 + slope_cont) < 1e-12:
                            boundary_val = S1
                        else:
                            boundary_val = (K - V_cont1 + slope_cont * S1) / (1 + slope_cont)
                            
                        # Clamp to the interval [S1, S2] to be safe
                        if boundary_val < S1:
                            boundary_val = S1
                        elif boundary_val > S2:
                            boundary_val = S2
                            
                        boundaries[ex_idx] = float(boundary_val)
                    else:
                        boundaries[ex_idx] = float(S_curr[j_ex])
            else:
                boundaries[ex_idx] = 0.0
        
        # Update V
        V = np.maximum(payoff, V_cont)
        
        # Deadline check
        if time.time() > deadline:
            break
    
    # Fill any remaining None boundaries
    for k in range(n_exercise):
        if boundaries[k] is None:
            boundaries[k] = K * 0.95
    
    # Ensure monotonicity (non-decreasing for puts)
    for k in range(n_exercise - 1):
        if boundaries[k] > boundaries[k + 1]:
            boundaries[k] = boundaries[k + 1]
            
    # Cap at K
    for k in range(n_exercise):
        if boundaries[k] > K:
            boundaries[k] = K

    class ExercisePolicy:
        def __init__(self, K, boundaries, n_exercise, T):
            self.K = K
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.T = T
            
            # Pre-compute exercise times for faster lookup
            self.ex_times = np.array([i / n_exercise for i in range(1, n_exercise + 1)])
            
        def exercise(self, current_state, time):
            n_paths = len(current_state)
            result = np.zeros(n_paths, dtype=bool)
            
            # Find closest exercise date
            # time is a scalar
            dists = np.abs(self.ex_times - time)
            best_ex_idx = np.argmin(dists)
            
            threshold = self.boundaries[best_ex_idx]
            
            # Exercise if S <= threshold
            result = (current_state <= threshold)
            
            return result
    
    policy = ExercisePolicy(K, boundaries, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END
