from scipy.optimize import minimize, minimize_scalar
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary
    (sigmoid of polynomial in remaining time fraction) with vectorized payoff,
    coordinate descent refinement, and local polynomial refinement.
    """
    start_time = time.time()
    DEADLINE = 145.0

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

    if paths.shape[1] >= n_dates + 1:
        exercise_spots = paths[:, 1:n_dates+1]
    else:
        exercise_spots = paths[:, 1:]
        n_dates = exercise_spots.shape[1]

    dt = T / n_dates
    disc = math.exp(-r * dt)
    disc_powers = np.array([disc ** t for t in range(n_dates)])

    def european_put_price(S, K_val, r_val, sigma_val, tau):
        if tau <= 0:
            return np.maximum(K_val - S, 0)
        S = np.maximum(S, 1e-12)
        sqrt_tau = np.sqrt(tau)
        d1 = (np.log(S / K_val) + (r_val + 0.5 * sigma_val**2) * tau) / (sigma_val * sqrt_tau)
        d2 = d1 - sigma_val * sqrt_tau
        N = norm.cdf
        return K_val * np.exp(-r_val * tau) * N(-d2) - S * N(-d1)

    def compute_european_thresholds():
        thresholds = np.zeros(n_dates)
        for t in range(n_dates):
            tau_t = (n_dates - t) * dt
            if tau_t <= 0:
                thresholds[t] = K
            else:
                S_grid = np.linspace(0.1, K * 1.5, 3000)
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
                        if abs(d1_val - d2_val) > 1e-15:
                            thresholds[t] = S1 - d1_val * (S2 - S1) / (d2_val - d1_val)
                        else:
                            thresholds[t] = S1
                    else:
                        thresholds[t] = K
        return thresholds

    initial_thresholds = compute_european_thresholds()

    # Parametric smooth boundary: sigmoid of polynomial in u (remaining time fraction)
    p = 6
    n_params = p + 1

    u_values = np.array([(n_dates - t) / n_dates for t in range(n_dates)])

    thresholds_clipped = np.clip(initial_thresholds, 1e-6, K - 1e-6)
    y_values = np.log(thresholds_clipped / (K - thresholds_clipped))

    X = np.zeros((n_dates, n_params))
    for k in range(n_params):
        X[:, k] = u_values ** k

    try:
        a_init, residuals, rank, sv = np.linalg.lstsq(X, y_values, rcond=None)
    except:
        a_init = np.zeros(n_params)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def thresholds_from_params(a):
        u_t = u_values
        poly_val = np.zeros(n_dates)
        for k in range(n_params):
            poly_val += a[k] * u_t ** k
        return K * sigmoid(poly_val)

    def compute_mean_payoff(thresholds):
        """
        Vectorized computation of mean discounted payoff under threshold policy.
        """
        n_p = exercise_spots.shape[0]
        n_d = n_dates
        
        payoffs = np.zeros(n_p)
        exercised = np.zeros(n_p, dtype=bool)
        
        for t in range(n_d):
            mask = ~exercised
            if not np.any(mask):
                break
            s_t = exercise_spots[:, t]
            should_exercise = s_t < thresholds[t]
            new_exercise = mask & should_exercise
            
            if np.any(new_exercise):
                disc_factor = disc_powers[t]
                payoffs[new_exercise] = (K - s_t[new_exercise]) * disc_factor
                exercised[new_exercise] = True
        
        return np.mean(payoffs)

    def negative_mean_payoff(a):
        thresholds = thresholds_from_params(a)
        return -compute_mean_payoff(thresholds)

    # Primary optimization using L-BFGS-B
    try:
        result = minimize(
            negative_mean_payoff,
            a_init,
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 2000
            }
        )
        a_opt = result.x
    except Exception:
        a_opt = a_init

    # Enhanced Coordinate Descent on individual thresholds
    # Use minimize_scalar for high precision
    thresholds = thresholds_from_params(a_opt)
    best_thresholds = thresholds.copy()
    best_val = compute_mean_payoff(thresholds)
    
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10.0:
        n_cd_iters = 30  # More iterations for better convergence
        
        for cd_iter in range(n_cd_iters):
            if time.time() - start_time > DEADLINE - 5.0:
                break
                
            improved = False
            for t in range(n_dates):
                current_thr = best_thresholds[t]
                
                # Bounds for the threshold
                lo = 0.0
                hi = K
                
                def obj_single(th, t_idx=t, ths=best_thresholds):
                    ths[t_idx] = th
                    return -compute_mean_payoff(ths)
                
                try:
                    res_cd = minimize_scalar(
                        obj_single,
                        bounds=(lo, hi),
                        method='bounded',
                        options={'xatol': 1e-6}
                    )
                    if res_cd.success:
                        new_thr = res_cd.x
                        if new_thr != current_thr:
                            best_thresholds[t] = new_thr
                            improved = True
                        new_val = compute_mean_payoff(best_thresholds)
                        if new_val > best_val:
                            best_val = new_val
                        else:
                            # Revert if it got worse (shouldn't happen with minimization of -payoff)
                            best_thresholds[t] = current_thr
                except Exception:
                    pass
            
            if not improved:
                break
    
    # Final polish
    if time.time() - start_time < DEADLINE - 2.0:
        # One more pass with tighter tolerance
        for t in range(n_dates):
            if time.time() - start_time > DEADLINE - 2.0:
                break
            current_thr = best_thresholds[t]
            lo = max(0.0, current_thr - K/100.0)
            hi = min(K, current_thr + K/100.0)
            
            def obj_single(th, t_idx=t, ths=best_thresholds):
                ths[t_idx] = th
                return -compute_mean_payoff(ths)
            
            try:
                res_cd = minimize_scalar(
                    obj_single,
                    bounds=(lo, hi),
                    method='bounded',
                    options={'xatol': 1e-8}
                )
                if res_cd.success:
                    best_thresholds[t] = res_cd.x
            except Exception:
                pass
        
        final_val = compute_mean_payoff(best_thresholds)
        if final_val > best_val:
            best_val = final_val

    # Clamp thresholds to [0, K]
    thresholds = np.clip(best_thresholds, 0.0, K)
    
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
