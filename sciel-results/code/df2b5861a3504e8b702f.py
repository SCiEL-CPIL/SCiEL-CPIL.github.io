from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary
    with a neural network approach for the threshold function.
    
    Uses backward induction on training paths for initialization,
    then optimizes a small neural network to fit the optimal boundary.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s limit
    
    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))
    
    if n_dates is None:
        n_dates = 12
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = paths.shape[0]
    
    # Extract spot prices at each exercise date
    # Exercise dates correspond to steps 1 through n_dates (no time-0 exercise)
    if paths.shape[1] >= n_dates + 1:
        exercise_spots = paths[:, 1:n_dates+1]
    else:
        exercise_spots = paths[:, 1:]
        n_dates = exercise_spots.shape[1]
    
    dt = T / n_dates
    disc = math.exp(-r * dt)
    
    # European put price function
    def european_put_price(S, K_val, r_val, sigma_val, tau):
        """Compute European put price."""
        if tau <= 0:
            return np.maximum(K_val - S, 0)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K_val) + (r_val + 0.5 * sigma_val**2) * tau) / (sigma_val * sqrt_tau)
        d2 = d1 - sigma_val * sqrt_tau
        N = norm.cdf
        return K_val * np.exp(-r_val * tau) * N(-d2) - S * N(-d1)
    
    # Backward induction on training paths to get initial thresholds
    def compute_backward_induction_thresholds():
        """
        Use backward induction on training paths to estimate optimal thresholds.
        """
        thresholds = np.zeros(n_dates)
        
        # Start from the last exercise date
        # At the last date, exercise if S < K (immediate payoff >= 0)
        thresholds[-1] = K
        
        # Work backwards
        for t in range(n_dates - 2, -1, -1):
            tau_t = (n_dates - t) * dt
            
            # For each path, compute continuation value using European put
            # as an approximation (this is a heuristic initialization)
            S_t = exercise_spots[:, t]
            cont_value = european_put_price(S_t, K, r, sigma, tau_t)
            exercise_payoff = np.maximum(K - S_t, 0)
            
            # Find the threshold where exercise is optimal
            # Use quantile-based approach: find S where payoff > cont_value
            diff = exercise_payoff - cont_value
            positive_mask = diff > 0
            
            if not np.any(positive_mask):
                thresholds[t] = 0.1
            elif np.all(positive_mask):
                thresholds[t] = K * 1.5
            else:
                # Find the median S where exercise is optimal
                S_exercise = S_t[positive_mask]
                S_continue = S_t[~positive_mask]
                
                if len(S_exercise) > 0 and len(S_continue) > 0:
                    thresholds[t] = np.median(np.concatenate([S_exercise, S_continue]))
                elif len(S_exercise) > 0:
                    thresholds[t] = np.max(S_exercise)
                else:
                    thresholds[t] = np.min(S_continue)
        
        return thresholds
    
    # Get initial thresholds from backward induction
    initial_thresholds = compute_backward_induction_thresholds()
    
    # Neural network parametric family:
    # threshold_t = K * sigmoid(W2^T * tanh(W1 * [u_t, 1] + b1) + b2)
    # where u_t = (n_dates - t) / n_dates
    
    hidden_size = 8
    n_input = 1  # u_t
    n_params_nn = n_input * hidden_size + hidden_size + hidden_size + 1  # W1, b1, W2, b2
    
    u_values = np.array([(n_dates - t) / n_dates for t in range(n_dates)])
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(x):
        return np.tanh(np.clip(x, -500, 500))
    
    def nn_thresholds(params):
        """Compute thresholds from neural network parameters."""
        # Unpack parameters
        W1 = params[:n_input * hidden_size].reshape(n_input, hidden_size)
        b1 = params[n_input * hidden_size : n_input * hidden_size + hidden_size]
        W2 = params[n_input * hidden_size + hidden_size : n_input * hidden_size + hidden_size + hidden_size]
        b2 = params[-1]
        
        # Forward pass
        X = u_values.reshape(-1, 1)  # (n_dates, 1)
        h = tanh(X @ W1 + b1)  # (n_dates, hidden_size)
        z = h @ W2 + b2  # (n_dates,)
        return K * sigmoid(z)
    
    # Initialize parameters from initial thresholds
    def params_from_thresholds(thresholds):
        """Initialize NN parameters to approximately match given thresholds."""
        thresholds_clipped = np.clip(thresholds, 1e-6, K - 1e-6)
        target_z = np.log(thresholds_clipped / (K - thresholds_clipped))
        
        # Simple initialization: set W2 to fit target_z from u_values
        # Use least squares: target_z ≈ W2^T * tanh(W1 * u + b1) + b2
        # For simplicity, start with W1 = 0, b1 = 0, so tanh(0) = 0
        # Then target_z ≈ b2, so b2 = mean(target_z)
        # And W2 can be initialized to spread the values
        
        rng = np.random.RandomState(42)
        W1 = rng.normal(0, 0.1, size=(n_input, hidden_size))
        b1 = rng.normal(0, 0.1, size=hidden_size)
        W2 = rng.normal(0, 1.0, size=hidden_size)
        b2 = np.mean(target_z)
        
        params = np.concatenate([W1.flatten(), b1, W2, [b2]])
        return params
    
    a_init = params_from_thresholds(initial_thresholds)
    
    def compute_payoff_under_policy(paths_exercise, thresholds, K_val, r_val, n_d, T_val):
        """
        Simulate the Bermudan put payoff under a given threshold policy.
        
        paths_exercise: (n_paths, n_dates) spot prices at each exercise date
        thresholds: (n_dates,) array of exercise thresholds
        Returns: (n_paths,) array of discounted payoffs
        """
        n_p = paths_exercise.shape[0]
        payoffs = np.zeros(n_p)
        exercised = np.zeros(n_p, dtype=bool)
        
        dt_val = T_val / n_d
        disc_val = math.exp(-r_val * dt_val)
        
        for t in range(n_d):
            mask = ~exercised
            if not np.any(mask):
                break
            s_t = paths_exercise[:, t]
            should_exercise = s_t < thresholds[t]
            new_exercise = mask & should_exercise
            
            if np.any(new_exercise):
                disc_factor = disc_val ** t
                payoffs[new_exercise] = (K_val - s_t[new_exercise]) * disc_factor
                exercised[new_exercise] = True
        
        return payoffs
    
    def negative_mean_payoff(params):
        thresholds = nn_thresholds(params)
        payoffs = compute_payoff_under_policy(exercise_spots, thresholds, K, r, n_dates, T)
        return -np.mean(payoffs)
    
    # Primary optimization using L-BFGS-B
    try:
        result = minimize(
            negative_mean_payoff,
            a_init,
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'ftol': 1e-15,
                'gtol': 1e-13,
                'maxfun': 20000
            }
        )
        a_opt = result.x
    except Exception:
        a_opt = a_init
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20.0:
        # Not enough time for refinement, use current result
        thresholds = nn_thresholds(a_opt)
    else:
        # Local refinement with multi-start
        best_params = a_opt.copy()
        best_val = -negative_mean_payoff(a_opt)
        
        remaining_time = DEADLINE - (time.time() - start_time) - 15.0
        if remaining_time > 5.0:
            # Local perturbations to escape potential local minima
            n_restarts = min(30, int(remaining_time / 1.0))
            rng = np.random.RandomState(42)
            for restart in range(n_restarts):
                # Try different perturbation scales
                scale = 0.05 * (restart % 3 + 1)
                params_try = a_opt + rng.normal(0, scale, size=n_params_nn)
                try:
                    val = -negative_mean_payoff(params_try)
                    if val > best_val:
                        best_val = val
                        best_params = params_try.copy()
                except Exception:
                    pass
                
                # Check time
                if time.time() - start_time > DEADLINE - 15.0:
                    break
            
            # If improvement found, do a local optimization from the best point
            if not np.allclose(best_params, a_opt):
                try:
                    result2 = minimize(
                        negative_mean_payoff,
                        best_params,
                        method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
                    )
                    if -result2.fun > best_val:
                        best_params = result2.x
                        best_val = -result2.fun
                except Exception:
                    pass
        
        thresholds = nn_thresholds(best_params)
    
    # Clamp thresholds to [0, K]
    thresholds = np.clip(thresholds, 0.0, K)
    
    class ExercisePolicy:
        def __init__(self, thresholds, n_dates):
            self.thresholds = thresholds
            self.n_dates = n_dates
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            - current_state: array of spot prices
            - time: the exercise date index (1-indexed, so time=1 is the first exercise date)
            
            Returns:
            - boolean array indicating whether to exercise for each path
            """
            t_idx = int(time) - 1
            if t_idx < 0 or t_idx >= self.n_dates:
                cs = np.asarray(current_state)
                return np.zeros(cs.shape, dtype=bool)
            
            threshold = float(self.thresholds[t_idx])
            cs = np.asarray(current_state, dtype=np.float64)
            return cs < threshold
    
    policy = ExercisePolicy(thresholds, n_dates)
    
    return policy
# EVOLVE-BLOCK-END
