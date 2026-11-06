import numpy as np
import time

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = np.asarray(critical_prices)
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        # Determine which exercise step this time corresponds to
        # Times are 1/12, 2/12, ..., 12/12
        # time is a scalar or array of scalars? 
        # The contract says "exercise(current_state, time)" returning a boolean vector.
        # Usually time is a scalar representing the current time step.
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return current_state <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    # Structural Change: Use a high-resolution binomial tree to compute the
    # optimal exercise boundary directly, rather than fitting to MC paths.
    # This avoids Monte Carlo noise in the optimization phase and relies on
    # the convergence of the tree to the true solution.
    
    # We need a tree fine enough to be accurate but fast enough to compute.
    # 2400 steps (200 per month) was used before. Let's try 4800 (400 per month)
    # if time permits, but we must vectorize to be fast.
    
    steps_per_month = 400
    N = n_ex * steps_per_month 
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    S0_ref = 100.0 
    
    # We only need the boundary at the 12 exercise dates.
    # We can run the tree backwards and record the boundary at the specific steps.
    
    # To save memory and time, we can use the fact that we only need the critical price
    # (the max S where exercise is optimal).
    
    # Initialize value array at maturity
    # Size N+1 is ~4801, which is small.
    V = np.zeros(N + 1)
    i_vals = np.arange(N + 1)
    
    # Log-space for stability
    log_d = np.log(d)
    log_u = np.log(u)
    log_S_N = np.log(S0_ref) + N * log_d + i_vals * (log_u - log_d)
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(12)
    S_curr = S_N.copy()
    
    # We need to check time frequently to avoid timeout
    # The loop runs N times. 4800 iterations is very fast in Python if vectorized.
    
    for j in range(N, 0, -1):
        # V_cont[i] = discount * (p * V[i] + (1-p) * V[i+1])
        V_cont = discount_factor * (p * V[0:j] + (1 - p) * V[1:j+1])
        
        # S_prev[i] = S_curr[i] / d
        S_prev = S_curr[0:j] / d
        
        V_ex = np.maximum(K - S_prev, 0.0)
        V_new = np.maximum(V_cont, V_ex)
        
        step_num = j - 1
        
        # Check if we are at an exercise date
        if step_num % steps_per_month == 0 and step_num > 0:
            k_idx = step_num // steps_per_month
            # Find the highest S where Exercise >= Continue
            exercise_mask = V_ex >= V_cont
            if np.any(exercise_mask):
                # np.where returns indices. We want the last one (highest S).
                last_true_idx = np.where(exercise_mask)[0][-1]
                critical_prices[k_idx - 1] = S_prev[last_true_idx]
            else:
                critical_prices[k_idx - 1] = 0.0
            
        V = V_new
        S_curr = S_prev
        
        # Time check
        if time.time() > deadline - 10.0:
            break

    # Ensure monotonicity for puts: c_1 <= c_2 <= ... <= c_12
    # The tree should produce this naturally, but numerical noise might break it.
    for k in range(1, 12):
        if critical_prices[k] < critical_prices[k-1]:
            critical_prices[k] = critical_prices[k-1]
            
    # At maturity, if S <= K, exercise. So c_12 should be K.
    critical_prices[11] = K
    
    # Optional: If we have time, we can refine these critical prices using the training paths
    # via a very fast grid search or local optimization.
    # Given the directive to make a structural change, the tree-based boundary is the primary change.
    # However, adding a light refinement step can capture the specific MC noise if any.
    
    # Let's check if we have time for a simple refinement.
    if time.time() < deadline - 20.0:
        paths = np.asarray(training_paths)
        if paths.ndim == 1:
            paths = paths.reshape(-1, 12)
        M = paths.shape[0]
        
        t_steps = np.arange(1, 13) / 12.0
        discount_factors = np.exp(-r * t_steps)
        
        def evaluate_policy(c_prices):
            W = paths <= c_prices[np.newaxis, :]
            exercised = np.zeros(M, dtype=bool)
            exercise_step = np.full(M, -1)
            
            for k in range(12):
                not_exercised = ~exercised
                cond = W[:, k] & not_exercised
                exercised[cond] = True
                exercise_step[cond] = k
                if np.all(exercised):
                    break
                    
            payoffs = np.zeros(M)
            if np.any(exercised):
                idx = np.arange(M)
                ex_idx = idx[exercised]
                ex_steps = exercise_step[exercised]
                S_ex = paths[ex_idx, ex_steps]
                disc_ex = discount_factors[ex_steps]
                payoffs[ex_idx] = disc_ex * (K - S_ex)
            return np.mean(payoffs)
        
        # Simple local refinement: check +/- small delta around the tree critical prices
        # This is a structural addition: Hybrid Tree + Local MC Refinement
        delta = 0.5
        for k in range(11): # Don't touch k=11 (K)
            base_val = evaluate_policy(critical_prices)
            best_val = base_val
            best_c = critical_prices[k]
            
            # Try a few offsets
            for dc in [-delta, -delta/2, delta/2, delta]:
                c_test = critical_prices.copy()
                c_test[k] = critical_prices[k] + dc
                # Enforce monotonicity roughly for the test
                if k > 0 and c_test[k] < c_test[k-1]: continue
                if k < 10 and c_test[k] > c_test[k+1]: continue
                
                val = evaluate_policy(c_test)
                if val > best_val:
                    best_val = val
                    best_c = c_test[k]
            
            critical_prices[k] = best_c
            
        # Final monotonicity pass
        for k in range(1, 12):
            if critical_prices[k] < critical_prices[k-1]:
                critical_prices[k] = critical_prices[k-1]
        critical_prices[11] = K

    return ExercisePolicy(critical_prices)
