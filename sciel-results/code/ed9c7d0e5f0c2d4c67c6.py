"""Bermudan exercise policy using Coordinate Descent on Critical Prices."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12
    
    # Extract states at exercise dates from training paths
    # training_paths shape: (n_paths, n_steps_total) ? 
    # The contract says "12 monthly exercise dates". 
    # Usually training_paths are full GBM paths. We need to identify the columns corresponding to exercise dates.
    # Assuming training_paths are sampled at equal intervals including exercise dates.
    # If n_paths is 4096, and we have 12 exercise dates, we need to know the time grid.
    # Standard assumption: The paths are provided at the exercise dates or we can interpolate.
    # Let's assume training_paths contains the spot prices at the 12 exercise dates directly or we can derive them.
    # Given the ambiguity, let's assume training_paths is (n_paths, 12) or (n_paths, T_steps).
    # If it's a full path, we need to map indices. 
    # Let's look at the shape. If it's 2D, first dim is paths.
    
    if len(training_paths.shape) == 2:
        # Assume columns are time steps. We need to pick the 12 exercise dates.
        # If the grid is uniform over T, and there are 12 exercise dates at T/12, 2T/12...
        # We don't know the total number of time steps in training_paths.
        # However, the contract says "12 monthly exercise dates".
        # Let's assume the input `training_paths` is already aligned to the exercise dates, 
        # OR we can simulate the states if the input is just seeds? No, it says "training_paths".
        # Let's assume training_paths has shape (n_paths, n_time_steps).
        # If n_time_steps is large, we need to downsample.
        # If n_time_steps is 12, we use them directly.
        
        n_paths, n_steps = training_paths.shape
        if n_steps >= 12:
            # Assume uniform spacing. Exercise dates are at indices proportional to 1..12.
            # If n_steps is exactly 12, indices are 0..11.
            # If n_steps > 12, we pick indices: int(i * n_steps / 12) for i in 1..12?
            # Or maybe the paths are already at the exercise dates.
            # Let's be safe: if n_steps == 12, use all. Else, interpolate or pick nearest.
            if n_steps == 12:
                states = training_paths
            else:
                # Linear interpolation to get states at the 12 exercise dates
                # Time grid of paths: 0, 1/(n_steps-1), ..., 1
                # Exercise times: 1/12, 2/12, ..., 12/12
                # This is risky if n_steps is not compatible.
                # Let's assume the evaluator provides paths at the exercise dates.
                # If not, we can't reliably extract them without knowing the time grid.
                # Let's assume the standard case: training_paths is (n_paths, 12).
                # If it's not, we fall back to a tree method.
                # For now, assume it's (n_paths, 12) or we can treat the last 12 as exercise dates?
                # Let's check if we can just use the last 12 columns if n_steps > 12.
                # Actually, a robust fallback is to use the Binomial Tree if the path structure is ambiguous.
                
                # Let's try to infer. If n_steps is 12, we are good.
                # If n_steps is e.g. 252, we need to map.
                # Let's assume the input is specifically prepared for the 12 exercise dates.
                # If n_steps != 12, we will use a Binomial Tree fallback to be safe, 
                # but try to optimize the tree speed.
                
                pass # Fallback logic below if needed
        else:
            states = None
    else:
        states = None

    # If we couldn't reliably extract states, or if we want to be robust,
    # we can use the Binomial Tree method but optimized.
    # However, the "Structural Change" directive suggests moving away from the tree.
    # Let's implement the Coordinate Descent assuming `states` is (n_paths, 12).
    
    if states is None or states.shape[1] != 12:
        # Fallback to a fast Binomial Tree (N=200)
        N = 200
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(r * dt) - d) / (u - d)
        discount = np.exp(-r * dt)
        
        exercise_steps = np.arange(1, n_exercise + 1) * (N // n_exercise)
        S0_tree = 100.0
        log_S0 = np.log(S0_tree)
        log_u = np.log(u)
        
        k_mat = np.arange(N + 1)
        log_S_mat = log_S0 + (2 * k_mat - N) * log_u
        S_mat = np.exp(log_S_mat)
        values_N = np.maximum(K - S_mat, 0.0)
        
        current_values = values_N.copy()
        critical_prices = {}
        
        for n in range(N - 1, -1, -1):
            if time.time() > DEADLINE:
                break
            up_val = current_values[1:n+2]
            down_val = current_values[0:n+1]
            cont_val = discount * (p * up_val + (1 - p) * down_val)
            
            if n in exercise_steps:
                k_nodes = np.arange(n + 1)
                log_S_nodes = log_S0 + (2 * k_nodes - n) * log_u
                S_nodes = np.exp(log_S_nodes)
                ex_val = np.maximum(K - S_nodes, 0.0)
                
                exercise_mask = ex_val > cont_val
                exercising_indices = np.where(exercise_mask)[0]
                if len(exercising_indices) > 0:
                    max_exercise_k = exercising_indices[-1]
                    if max_exercise_k < n:
                        S_ex = S_nodes[max_exercise_k]
                        S_no_ex = S_nodes[max_exercise_k + 1]
                        diff_k = ex_val[max_exercise_k] - cont_val[max_exercise_k]
                        diff_k1 = ex_val[max_exercise_k + 1] - cont_val[max_exercise_k + 1]
                        if diff_k > 0 and diff_k1 < 0:
                            frac = diff_k / (diff_k - diff_k1)
                            critical_price = S_ex + frac * (S_no_ex - S_ex)
                        else:
                            critical_price = S_no_ex if diff_k > 0 else S_ex
                    else:
                        critical_price = S_nodes[max_exercise_k]
                    critical_prices[n] = critical_price
                else:
                    critical_prices[n] = 0.0
                new_values = np.maximum(ex_val, cont_val)
            else:
                new_values = cont_val
            
            current_values = new_values
        
        class ExercisePolicy:
            def __init__(self):
                self.critical_prices = critical_prices
                self.exercise_steps_set = set(exercise_steps)
                self.N = N
                self.dt = dt
                self.K = K
                self.r = r
                self.log_S0 = log_S0
                self.log_u = log_u

            def exercise(self, current_state, time):
                n = int(round(time / self.dt))
                n = max(0, min(n, self.N))
                if n not in self.critical_prices:
                    return np.zeros(len(current_state), dtype=bool)
                critical_price = self.critical_prices[n]
                return current_state < critical_price
        
        return ExercisePolicy()

    # If we have states (n_paths, 12), proceed with Coordinate Descent
    n_paths, n_dates = states.shape
    
    # Precompute discount factors for each exercise date
    # Exercise dates are at t_i = i * T / 12 for i=1..12
    # Discount factor for payoff at t_i is exp(-r * t_i)
    t_ex = np.linspace(T/12, T, 12)
    disc_factors = np.exp(-r * t_ex)
    
    # Initialize critical prices
    # Heuristic: C_t = K * exp(-r * t) is a good start for puts? 
    # Or use a small tree. Let's use a small tree (N=50) for initialization to be safe.
    N_init = 50
    dt_init = T / N_init
    u_init = np.exp(sigma * np.sqrt(dt_init))
    d_init = 1.0 / u_init
    p_init = (np.exp(r * dt_init) - d_init) / (u_init - d_init)
    disc_init = np.exp(-r * dt_init)
    ex_steps_init = np.arange(1, 13) * (N_init // 12)
    
    S0_init = 100.0
    log_S0_init = np.log(S0_init)
    log_u_init = np.log(u_init)
    
    k_mat_init = np.arange(N_init + 1)
    log_S_mat_init = log_S0_init + (2 * k_mat_init - N_init) * log_u_init
    S_mat_init = np.exp(log_S_mat_init)
    val_init = np.maximum(K - S_mat_init, 0.0)
    
    curr_val = val_init.copy()
    init_crit = [0.0] * 12
    
    for n in range(N_init - 1, -1, -1):
        up_val = curr_val[1:n+2]
        down_val = curr_val[0:n+1]
        cont_val = disc_init * (p_init * up_val + (1 - p_init) * down_val)
        if n in ex_steps_init:
            k_nodes = np.arange(n + 1)
            log_S_nodes = log_S0_init + (2 * k_nodes - n) * log_u_init
            S_nodes = np.exp(log_S_nodes)
            ex_val = np.maximum(K - S_nodes, 0.0)
            mask = ex_val > cont_val
            idx = np.where(mask)[0]
            if len(idx) > 0:
                max_k = idx[-1]
                if max_k < n:
                    S1 = S_nodes[max_k]
                    S2 = S_nodes[max_k+1]
                    d1 = ex_val[max_k] - cont_val[max_k]
                    d2 = ex_val[max_k+1] - cont_val[max_k+1]
                    if d1 > 0 and d2 < 0:
                        frac = d1 / (d1 - d2)
                        init_crit[n // (N_init//12) - 1] = S1 + frac * (S2 - S1)
                    else:
                        init_crit[n // (N_init//12) - 1] = S2 if d1 > 0 else S1
                else:
                    init_crit[n // (N_init//12) - 1] = S_nodes[max_k]
            else:
                init_crit[n // (N_init//12) - 1] = 0.0
            curr_val = np.maximum(ex_val, cont_val)
        else:
            curr_val = cont_val
            
    critical_prices = np.array(init_crit)
    
    # Coordinate Descent
    # For each date, sweep a range of critical prices to maximize expected payoff
    # Range: [0, K]
    
    def calculate_payoff(crits):
        # crits: array of 12 critical prices
        # Returns total discounted payoff for all paths
        # We need to simulate the decision sequentially
        # For each path, we check dates from last to first? No, decisions are made forward, 
        # but the value depends on future decisions.
        # Actually, the "payoff" of a policy is the sum of discounted exercise values for paths where we exercise,
        # assuming we stop exercising after the first exercise.
        # Since it's a Bermudan option, we exercise at the FIRST date where S < C_t.
        
        payoffs = np.zeros(n_paths)
        active = np.ones(n_paths, dtype=bool)
        
        for t_idx in range(12):
            if not np.any(active):
                break
            S_t = states[:, t_idx]
            C_t = crits[t_idx]
            
            # Exercise if S < C_t and still active
            ex_mask = (S_t < C_t) & active
            
            if np.any(ex_mask):
                # Add discounted payoff
                payoffs[ex_mask] += disc_factors[t_idx] * (K - S_t[ex_mask])
                # Deactivate these paths
                active[ex_mask] = False
        
        return np.mean(payoffs)
    
    # Optimize each critical price
    # We do a few passes
    n_sweeps = 3
    for sweep in range(n_sweeps):
        for t_idx in range(12):
            if time.time() > DEADLINE:
                break
            
            # Define a grid of critical prices to test
            # Focus around the current estimate
            current_c = critical_prices[t_idx]
            # Range: [max(0, current_c - 10), min(K, current_c + 10)]
            # Or just [0, K] with 100 points
            c_grid = np.linspace(0, K, 100)
            
            best_c = current_c
            best_val = calculate_payoff(critical_prices)
            
            for c in c_grid:
                critical_prices[t_idx] = c
                val = calculate_payoff(critical_prices)
                if val > best_val:
                    best_val = val
                    best_c = c
            
            critical_prices[t_idx] = best_c
            
        if time.time() > DEADLINE:
            break

    # Final Policy
    class ExercisePolicy:
        def __init__(self):
            self.critical_prices = critical_prices
            self.n_dates = 12
            self.T = T
            self.K = K

        def exercise(self, current_state, time):
            # Determine which date this is
            # time is in [0, T]
            # Dates are at T/12, 2T/12...
            # t_idx = round(time / (T/12)) - 1
            t_idx = int(round(time / (T / 12))) - 1
            
            # Clamp to valid range
            if t_idx < 0 or t_idx >= 12:
                return np.zeros(len(current_state), dtype=bool)
            
            c_t = self.critical_prices[t_idx]
            return current_state < c_t

    return ExercisePolicy()

# EVOLVE-BLOCK-END
