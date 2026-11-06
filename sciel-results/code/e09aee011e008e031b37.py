import numpy as np
import time

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return current_state <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    paths_raw = np.asarray(training_paths)
    
    # Determine shape of paths
    if paths_raw.ndim == 1:
        paths_raw = paths_raw.reshape(1, -1)
        
    if paths_raw.shape[1] == 13:
        paths = paths_raw[:, 1:13]
    elif paths_raw.shape[1] == 12:
        paths = paths_raw[:, 0:12]
    else:
        paths = paths_raw[:, -12:]
        
    M = paths.shape[0]
    
    # Discount factors for the 12 exercise dates
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    # 1. Initial guess: Fine Binomial Tree
    # Using a very fine tree (e.g., 2400 steps) for a high-quality starting point.
    # 2400 steps * 12 months is manageable in numpy if vectorized, but we only need boundaries at 12 points.
    # Actually, a simpler approach: use the training paths themselves to get a "sample" boundary,
    # or just use a moderate tree. Let's stick to a tree for robustness but make it efficient.
    
    # Let's use 1200 steps as in the original, which is a good balance.
    steps_per_month = 100
    N_tree = n_ex * steps_per_month
    dt_tree = T / N_tree
    u_tree = np.exp(sigma * np.sqrt(dt_tree))
    d_tree = 1.0 / u_tree
    p_tree = (np.exp(r * dt_tree) - d_tree) / (u_tree - d_tree)
    disc_tree = np.exp(-r * dt_tree)
    
    S0_ref = 100.0
    i_vals = np.arange(N_tree + 1)
    log_S_N = np.log(S0_ref) + N_tree * np.log(d_tree) + i_vals * (np.log(u_tree) - np.log(d_tree))
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(12)
    S_curr = S_N.copy()
    
    # Backward induction
    for j in range(N_tree, 0, -1):
        V_cont = disc_tree * (p_tree * V[0:j] + (1.0 - p_tree) * V[1:j+1])
        S_prev = S_curr[0:j] / d_tree
        V_ex = np.maximum(K - S_prev, 0.0)
        V_new = np.maximum(V_cont, V_ex)
        
        step_num = j - 1
        if step_num % steps_per_month == 0 and step_num > 0:
            k_idx = step_num // steps_per_month # 1 to 12
            exercise_mask = V_ex >= V_cont
            if np.any(exercise_mask):
                # The boundary is the max S where exercise is optimal
                # Since S_prev is increasing with index, we find the last index where mask is True
                last_true_idx = np.where(exercise_mask)[0][-1]
                critical_prices[k_idx - 1] = S_prev[last_true_idx]
            else:
                critical_prices[k_idx - 1] = 0.0
        
        V = V_new
        S_curr = S_curr[0:j] / d_tree
        
        if time.time() > deadline - 60.0:
            break
    
    critical_prices = np.clip(critical_prices, 0, K)
    
    # 2. Structural Change: Stochastic Approximation / Online Gradient Ascent
    # Instead of coordinate descent on the full sample (which is O(M) per eval and requires many evals),
    # we use a stochastic approach. We treat the objective as the expected payoff.
    # We can approximate the gradient of the expected payoff with respect to the critical prices.
    # However, the "exercise at first opportunity" makes the gradient tricky (kinks).
    # 
    # Alternative Structural Change: **Pairwise Swap Optimization (Hill Climbing)**
    # The space of 12 critical prices is small. 
    # The previous method used coordinate descent with bisection. Bisection assumes unimodality which might fail or be slow.
    # Let's try a **Random Restart + Local Search** or **Simulated Annealing** on the 12-dimensional space.
    # 
    # Actually, a better structural change for this specific "threshold" problem:
    # The optimal policy is defined by the boundaries. 
    # We can optimize the boundaries directly using the training data but with a **faster evaluation**.
    # 
    # Let's stick to the coordinate descent but improve the search:
    # The previous bisection was 8 iterations. That's quite coarse (2^-8 resolution).
    # Let's use a **Nelder-Mead** simplex method or **Powell's method** from scipy.optimize?
    # The prompt says SciPy is available.
    # 
    # Let's use `scipy.optimize.minimize` with Nelder-Mead on the negative payoff.
    # This is a different optimization family (direct search vs coordinate descent).
    
    from scipy.optimize import minimize
    
    def evaluate_policy(c_prices):
        """Evaluate the mean discounted payoff for a given policy."""
        # Vectorized check for exercise
        W = paths <= c_prices[np.newaxis, :]
        
        exercised = np.zeros(M, dtype=bool)
        exercise_step = np.full(M, -1, dtype=int)
        
        for k in range(12):
            not_exercised = ~exercised
            if not np.any(not_exercised):
                break
            cond = W[:, k] & not_exercised
            exercised[cond] = True
            exercise_step[cond] = k
        
        payoffs = np.zeros(M)
        if np.any(exercised):
            idx = np.arange(M)
            ex_idx = idx[exercised]
            ex_steps = exercise_step[exercised]
            S_ex = paths[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            payoffs[ex_idx] = disc_ex * (K - S_ex)
        
        return np.mean(payoffs)
    
    def negative_payoff(c_prices):
        # Clip to valid range [0, K] to help the optimizer
        c_clipped = np.clip(c_prices, 0, K)
        return -evaluate_policy(c_clipped)
    
    # Initial point from tree
    x0 = critical_prices.copy()
    
    # Use Nelder-Mead
    # Max iterations to keep within time limit
    # Each evaluation is O(M*12) which is fast (4096*12 ~ 50k ops). 
    # Nelder-Mead might need 100-500 evaluations.
    # 500 * 50k = 25M ops. Very fast in numpy.
    
    try:
        if time.time() < deadline - 10.0:
            res = minimize(negative_payoff, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            if res.success or res.fun < -evaluate_policy(x0):
                critical_prices = np.clip(res.x, 0, K)
            
            # If time permits, do a second pass with tighter tolerance or different start
            if time.time() < deadline - 5.0:
                # Perturb and re-optimize to escape local minima?
                # Nelder-Mead is decent. Let's just refine.
                res2 = minimize(negative_payoff, critical_prices, method='Nelder-Mead', 
                                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7})
                if res2.fun < -evaluate_policy(critical_prices):
                    critical_prices = np.clip(res2.x, 0, K)
                    
    except Exception:
        pass # Fallback to tree solution

    return ExercisePolicy(critical_prices)
