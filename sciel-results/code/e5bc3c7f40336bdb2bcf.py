# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize, brentq
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary
    fitted via grid search on a coarse 1D grid, with bisection refinement.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_time_steps + 1) representing spot prices
    - contract: dict with contract parameters
    
    Returns:
    - An object with an exercise(current_state, time) method returning a boolean vector
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
    
    # The training_paths has shape (n_paths, n_time_steps + 1)
    # Exercise dates are at indices 1 through n_dates in the path array
    # So we need n_dates columns for the exercise dates
    # If the array has more columns, we use the last n_dates columns as exercise dates
    # Actually, let's be careful: the contract says 12 monthly exercise dates, no time-0 exercise
    # So the exercise dates correspond to steps 1, 2, ..., 12
    # training_paths[:, 1:n_dates+1] would give us the spot at each exercise date
    
    # Let's extract the spot prices at each exercise date
    # The shape is (n_paths, n_time_steps + 1) where n_time_steps should be n_dates
    # So columns 1 through n_dates are the exercise dates
    
    if paths.shape[1] >= n_dates + 1:
        # Use columns 1 through n_dates as exercise dates
        exercise_spots = paths[:, 1:n_dates+1]
    else:
        # Fallback: use all columns except the first
        exercise_spots = paths[:, 1:]
        n_dates = exercise_spots.shape[1]
    
    dt = T / n_dates
    disc = math.exp(-r * dt)
    
    # Parametric smooth boundary: represent the exercise boundary as a smooth function
    # of log-spot. We'll use a polynomial in log(S/K) for the boundary.
    # The boundary at each date t is: S_threshold(t) = K * exp(sum_{k=0}^{p} c_k(t) * (log(S/K))^k)
    # But this is circular. Instead, let's use a simpler parametric form.
    
    # Better approach: use a monotonic polynomial in a normalized time variable.
    # Let u = (n_dates - t) / n_dates be the remaining fraction of time (1 at start, 0 at end)
    # The threshold is: S_threshold(u) = K * (1 - exp(-alpha * u^beta))
    # This ensures the threshold is in (0, K) and increases toward K as u -> 0 (maturity)
    
    # Actually, let's use a more flexible parametric form:
    # S_threshold(u) = K * sigmoid(sum_{k=0}^{p} a_k * u^k)
    # where u is the remaining time fraction
    
    # Let's use a polynomial in u with degree p=4
    p = 4
    n_params = p + 1
    
    # Initial guess for parameters
    # We want the threshold to be low when there's lots of time remaining (u close to 1)
    # and high (close to K) when there's little time remaining (u close to 0)
    
    # Let's use the European put as a guide for the initial guess
    def european_put_price(S, K, r, sigma, tau):
        """Compute European put price."""
        if tau <= 0:
            return np.maximum(K - S, 0)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        N = norm.cdf
        return K * np.exp(-r * tau) * N(-d2) - S * N(-d1)
    
    # Compute initial thresholds using European put approximation
    def compute_european_thresholds():
        thresholds = np.zeros(n_dates)
        for t in range(n_dates):
            tau_t = (n_dates - t) * dt
            if tau_t <= 0:
                thresholds[t] = K
            else:
                # Find the threshold where immediate payoff equals European put price
                S_grid = np.linspace(0.1, K * 1.5, 1000)
                cont_value = european_put_price(S_grid, K, r, sigma, tau_t)
                exercise_payoff = np.maximum(K - S_grid, 0)
                diff = exercise_payoff - cont_value
                
                positive_mask = diff > 0
                if not np.any(positive_mask):
                    thresholds[t] = 0.1
                elif np.all(positive_mask):
                    thresholds[t] = K * 1.5
                else:
                    idx = np.where(positive_mask[:-1] & ~positive_mask[1:])[0]
                    if len(idx) > 0:
                        i = idx[0]
                        S1, S2 = S_grid[i], S_grid[i+1]
                        d1_val, d2_val = diff[i], diff[i+1]
                        if d1_val != d2_val:
                            thresholds[t] = S1 - d1_val * (S2 - S1) / (d2_val - d1_val)
                        else:
                            thresholds[t] = S1
                    else:
                        thresholds[t] = K
        return thresholds
    
    # Get initial thresholds from European put
    initial_thresholds = compute_european_thresholds()
    
    # Now fit a parametric curve to these initial thresholds
    # u_t = (n_dates - t) / n_dates for t = 0, ..., n_dates-1
    u_values = np.array([(n_dates - t) / n_dates for t in range(n_dates)])
    
    # Fit: threshold/K = sigmoid(sum_{k=0}^{p} a_k * u^k)
    # So: log(threshold/K) - log(1 - threshold/K) = sum_{k=0}^{p} a_k * u^k
    # Let y_t = log(threshold_t / (K - threshold_t)) for threshold_t in (0, K)
    
    # Clip thresholds to avoid log of 0 or 1
    thresholds_clipped = np.clip(initial_thresholds, 1e-6, K - 1e-6)
    y_values = np.log(thresholds_clipped / (K - thresholds_clipped))
    
    # Design matrix for polynomial fit
    X = np.zeros((n_dates, n_params))
    for k in range(n_params):
        X[:, k] = u_values ** k
    
    # Least squares fit
    try:
        a_init, residuals, rank, sv = np.linalg.lstsq(X, y_values, rcond=None)
    except:
        a_init = np.zeros(n_params)
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def thresholds_from_params(a):
        """Compute thresholds from polynomial parameters."""
        # u values for each date
        u_t = np.array([(n_dates - t) / n_dates for t in range(n_dates)])
        # Compute polynomial value
        poly_val = np.zeros(n_dates)
        for k in range(n_params):
            poly_val += a[k] * u_t ** k
        # Apply sigmoid and scale by K
        return K * sigmoid(poly_val)
    
    def compute_payoff_under_policy(paths_exercise, thresholds, K, r, n_dates, T):
        """
        Simulate the Bermudan put payoff under a given threshold policy.
        
        paths_exercise: (n_paths, n_dates) spot prices at each exercise date
        thresholds: (n_dates,) array of exercise thresholds
        Returns: (n_paths,) array of discounted payoffs
        """
        n_paths = paths_exercise.shape[0]
        payoffs = np.zeros(n_paths)
        
        # Track which paths have been exercised
        exercised = np.zeros(n_paths, dtype=bool)
        
        dt = T / n_dates
        disc = math.exp(-r * dt)
        
        for t in range(n_dates):
            mask = ~exercised
            if not np.any(mask):
                break
            # At date t, check if S < threshold[t]
            s_t = paths_exercise[:, t]
            should_exercise = s_t < thresholds[t]
            new_exercise = mask & should_exercise
            
            if np.any(new_exercise):
                # Payoff is K - S_t, discounted back to time 0
                disc_factor = disc ** t  # discount from date t to time 0
                payoffs[new_exercise] = (K - s_t[new_exercise]) * disc_factor
                exercised[new_exercise] = True
        
        return payoffs
    
    def negative_mean_payoff(a):
        thresholds = thresholds_from_params(a)
        payoffs = compute_payoff_under_policy(exercise_spots, thresholds, K, r, n_dates, T)
        return -np.mean(payoffs)
    
    # Optimize using L-BFGS-B
    try:
        result = minimize(
            negative_mean_payoff,
            a_init,
            method='L-BFGS-B',
            options={
                'maxiter': 300,
                'ftol': 1e-12,
                'gtol': 1e-10,
                'maxfun': 2000
            }
        )
        a_opt = result.x
    except Exception:
        a_opt = a_init
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15.0:
        # Not enough time for refinement, use current result
        thresholds = thresholds_from_params(a_opt)
    else:
        # Try local refinement with multi-start
        best_a = a_opt.copy()
        best_val = -negative_mean_payoff(a_opt)
        
        remaining_time = DEADLINE - (time.time() - start_time) - 10.0
        if remaining_time > 5.0:
            # A few local perturbations to escape potential local minima
            n_restarts = min(8, int(remaining_time / 2.0))
            rng = np.random.RandomState(42)
            for restart in range(n_restarts):
                a_try = a_opt + rng.normal(0, 0.05, size=n_params)
                try:
                    val = -negative_mean_payoff(a_try)
                    if val > best_val:
                        best_val = val
                        best_a = a_try
                except Exception:
                    pass
                
                # Check time
                if time.time() - start_time > DEADLINE - 10.0:
                    break
            
            # If improvement found, do a local optimization from the best point
            if not np.allclose(best_a, a_opt):
                try:
                    result2 = minimize(
                        negative_mean_payoff,
                        best_a,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if -result2.fun > best_val:
                        best_a = result2.x
                        best_val = -result2.fun
                except Exception:
                    pass
        
        thresholds = thresholds_from_params(best_a)
    
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
            # time is 1-indexed, so the index into thresholds is time - 1
            t_idx = int(time) - 1
            if t_idx < 0 or t_idx >= self.n_dates:
                # Invalid time, don't exercise
                cs = np.asarray(current_state)
                return np.zeros(cs.shape, dtype=bool)
            
            threshold = float(self.thresholds[t_idx])
            cs = np.asarray(current_state, dtype=np.float64)
            return cs < threshold
    
    policy = ExercisePolicy(thresholds, n_dates)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Shouldn't happen, but just in case
        pass
    
    return policy
# EVOLVE-BLOCK-END
