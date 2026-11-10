import numpy as np
import math
import time

class ExercisePolicy:
    def __init__(self, boundaries, n_exercises, T):
        """
        boundaries: list of critical prices for each exercise date (1-indexed internally, but stored 0-11 for k=1..12)
        n_exercises: number of exercise dates
        T: total maturity
        """
        self.boundaries = boundaries
        self.n_exercises = n_exercises
        self.T = T
        self.dt = T / n_exercises

    def exercise(self, current_state, time):
        """
        current_state: array of stock prices
        time: scalar or array of times
        returns: boolean vector
        """
        current_state = np.asarray(current_state, dtype=float)
        time = np.asarray(time, dtype=float)
        
        if time.ndim == 0:
            time = np.full(current_state.shape, time)
        
        n = len(current_state)
        result = np.zeros(n, dtype=bool)
        
        for i in range(n):
            t = time[i]
            s = current_state[i]
            
            # Determine which exercise date this is
            # k is in 1..n_exercises
            # t_k = k * dt
            # k = t / dt
            k_float = t / self.dt
            k = int(round(k_float))
            
            # Clamp k to valid range [1, n_exercises]
            if k < 1:
                k = 1
            elif k > self.n_exercises:
                k = self.n_exercises
            
            # Check if we are actually at an exercise date
            # If the time is not close to an exercise date, we shouldn't exercise.
            # However, the task implies the policy is called at exercise dates.
            # To be safe, if |t - k*dt| is large, return False.
            if abs(t - k * self.dt) > 1e-6:
                result[i] = False
                continue

            # Get the boundary for this date
            # boundaries is 0-indexed, so index is k-1
            s_crit = self.boundaries[k-1]
            
            # For a put, exercise if S <= S_crit
            if s <= s_crit:
                result[i] = True
            else:
                result[i] = False
                
        return result

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option.
    Uses a 1200-step binomial tree to determine optimal exercise boundaries.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # High resolution tree for accurate boundaries
    N_steps = 1200
    dt = T / N_steps
    
    # CRR parameters
    u = math.exp(sigma * math.sqrt(dt))
    d = math.exp(-sigma * math.sqrt(dt))
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    
    # We need to find the critical price S_crit for each exercise date k=1..12.
    # Exercise dates are at steps j_k = k * (N_steps / n_exercises) = k * 100.
    
    # We perform backward induction.
    # To save memory, we only keep the value array for the next time step.
    # V_next has length j+2 when computing step j.
    
    # Initialize at T (j=N_steps)
    # i ranges from 0 to N_steps
    # S(i, j) = S_0 * u^(2*i - j)
    # We choose S_0 = 100.0 for the tree construction.
    S_0_tree = 100.0
    log_S_0 = math.log(S_0_tree)
    log_u = math.log(u)
    
    # Terminal values
    N = N_steps
    i_terminal = np.arange(N + 1)
    exponents = 2 * i_terminal - N
    log_S_terminal = log_S_0 + exponents * log_u
    S_terminal = np.exp(log_S_terminal)
    V_next = np.maximum(K - S_terminal, 0.0)
    
    # Store critical indices for each exercise date
    # i_crits[k] will store the node index i at step j=k*100
    # where the boundary is located.
    i_crits = {}
    
    # Exercise dates in terms of tree steps
    # k=1 -> j=100, ..., k=12 -> j=1200 (maturity, handled in init, but we might check j=1200 in loop? No, loop goes to 0)
    # The loop j goes from N-1 down to 0.
    # Exercise dates are j=100, 200, ..., 1100.
    # j=1200 is maturity.
    
    for j in range(N - 1, -1, -1):
        if time.time() > deadline:
            # Fallback to a simpler policy if we run out of time (unlikely)
            # Just return a simple threshold policy
            break
            
        # Current step j has i = 0 .. j
        i_curr = np.arange(j + 1)
        exponents_curr = 2 * i_curr - j
        log_S_curr = log_S_0 + exponents_curr * log_u
        S_curr = np.exp(log_S_curr)
        
        # Intrinsic Value
        V_int = np.maximum(K - S_curr, 0.0)
        
        # Continuation Value
        # V_cont(i) = disc * (p * V_next(i+1) + (1-p) * V_next(i))
        # V_next has length j+2
        V_cont = disc * (p * V_next[1:] + (1 - p) * V_next[:-1])
        
        # Check if current step j is an exercise date
        # Exercise dates are at j = k * 100 for k=1..11 (k=12 is maturity)
        # Note: The task says 12 exercise dates. Usually maturity is one of them.
        # "12 monthly exercise dates". If T=1, months are 1/12, ..., 12/12.
        # So k=1..12.
        # j_k = k * 100.
        # j=1200 is k=12.
        # j=1100 is k=11.
        # ...
        # j=100 is k=1.
        
        if j % 100 == 0 and j > 0:
            k = j // 100
            # Find the boundary where V_int >= V_cont
            # For a put, as S increases (i increases), V_int decreases, V_cont increases.
            # So V_int >= V_cont is true for small i, false for large i.
            # We want the largest i where exercise is optimal.
            
            diff = V_int - V_cont
            exercise_mask = diff >= 0
            
            if np.any(exercise_mask):
                # Find the last True index
                last_true_idx = np.where(exercise_mask)[0][-1]
                i_crits[k] = last_true_idx
            else:
                # Never exercise at this step (boundary below lowest node)
                i_crits[k] = -1
        
        # Update V for this step
        V_curr = np.where(V_int > V_cont, V_int, V_cont)
        
        # Prepare for next iteration (j-1)
        V_next = V_curr
        
    # Convert i_crits to S_crits
    s_crit_values = [0.0] * n_exercises
    
    for k in range(1, n_exercises + 1):
        j = k * 100
        if k == 12:
            # Maturity: exercise if ITM
            # S_crit at maturity is K (exercise if S < K)
            # Actually, at maturity, you exercise if K - S > 0.
            # So S_crit = K.
            s_crit_values[k-1] = K
        else:
            i = i_crits.get(k, -1)
            if i < 0:
                # Boundary is below the lowest node of the tree.
                # The lowest node is S_0 * u^(-j) = S_0 * d^j.
                # If we never exercise, it means even at the lowest price, continuation is better?
                # Unlikely for a put with K=100, S_0=100.
                # But if it happens, set boundary to 0.
                s_crit_values[k-1] = 0.0
            else:
                # Calculate S at node (i, j)
                exponents = 2 * i - j
                log_S = log_S_0 + exponents * log_u
                s_crit_values[k-1] = math.exp(log_S)
                
    # Optimization: If we have time, we could refine the boundary by interpolation?
    # The discrete boundary is usually sufficient.
    
    return ExercisePolicy(s_crit_values, n_exercises, T)
