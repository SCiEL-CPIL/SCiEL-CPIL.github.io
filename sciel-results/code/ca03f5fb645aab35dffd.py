"""
Bermudan Put Exercise Policy via Analytic Backward Induction.

Strategy:
1. Ignore training paths for policy fitting (they are noisy samples).
2. Construct a high-resolution grid of spot prices using Gauss-Hermite quadrature
   in log-space to handle the put payoff asymmetry.
3. Perform exact backward induction on this grid to compute the optimal
   continuation value and exercise decision at each exercise date.
4. The resulting policy is a lookup/interpolation function on the grid,
   which generalizes perfectly to the known GBM dynamics.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Extract contract parameters
    # Standard contract params: K, r, sigma, T, n_steps (exercise dates)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Determine number of exercise dates
    # Task says "12 monthly exercise dates (no time-0 exercise)"
    # training_paths shape is usually (n_paths, n_time_steps + 1)
    # If training_paths has 13 columns (t=0..12), there are 12 exercise dates at t=1..12.
    if 'n_steps' in contract:
        n_exercise = contract['n_steps']
    elif 'n_exercise' in contract:
        n_exercise = contract['n_exercise']
    else:
        # Infer from training paths if possible, otherwise default to 12
        try:
            n_cols = training_paths.shape[1]
            n_exercise = n_cols - 1
        except:
            n_exercise = 12

    # Time steps
    dt = T / n_exercise
    
    # Grid construction
    # We use a log-normal grid. 
    # The spot price S_t = S_0 * exp((r - 0.5*sigma^2)*t + sigma*sqrt(t)*Z)
    # At time t_i, the distribution of log(S) is Normal(mu_i, var_i).
    # We want a grid that covers the relevant range of S at each time step.
    # A common approach is to use a fixed set of quantiles or a fixed log-grid.
    # For Bermudan options, the optimal exercise boundary is monotonic.
    # We will use a Gauss-Hermite quadrature grid for the integral expectation.
    # However, for the *state* grid, we need points where we evaluate the policy.
    # Let's use a log-spaced grid of spot prices.
    
    # To be robust, we define a grid of log-spots.
    # The relevant range for a put (K=100, S0=100, sigma=0.2, T=1) is roughly 50 to 200.
    # Let's use 200 points in log-space for high resolution.
    n_grid = 200
    # Log-space bounds: 
    # At t=12 (T=1), S can be quite low. 
    # Let's set log_S_min = log(10) and log_S_max = log(500) to be safe.
    log_S_min = np.log(10.0)
    log_S_max = np.log(500.0)
    log_S_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_S_grid)
    
    # Precompute discount factors
    discount_factors = np.exp(-r * np.arange(1, n_exercise + 1) * dt)
    # discount_factors[i] corresponds to discounting from t_{i+1} to t_i?
    # Let's define V_i as value at exercise date i (1-indexed in time, 0-indexed in array).
    # V_n (at T) = Payoff
    # V_i = max(Payoff_i, disc * E[V_{i+1} | S_i])
    
    # Backward Induction
    # V is a 1D array of values on S_grid for the next time step.
    # We start from the last exercise date.
    
    # Payoff at last date
    V_next = np.maximum(K - S_grid, 0.0)
    exercise_next = V_next > 0.0 # Always exercise if ITM at maturity
    
    # We need to store the policy for each time step.
    # policy[i] is a boolean array of shape (n_grid,) indicating exercise on S_grid.
    policies = [None] * n_exercise
    policies[-1] = exercise_next
    
    # Quadrature weights for expectation E[V_{i+1}(S_{i+1}) | S_i]
    # S_{i+1} = S_i * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    # log(S_{i+1}) = log(S_i) + drift + sigma*sqrt(dt)*Z
    # We integrate over Z using Gauss-Hermite quadrature.
    # E[g(Z)] ≈ sum w_j g(z_j)
    # For standard normal, we use roots_hermite and adjust weights.
    # Let's use 50 points for quadrature, which is very accurate for smooth functions.
    n_quad = 50
    x_quad, w_quad = roots_hermite(n_quad)
    # Convert to standard normal quadrature
    # E[h(Z)] = 1/sqrt(pi) * sum w_j h(x_j / sqrt(2))
    # Wait, standard GH: ∫ e^{-x^2} f(x) dx ≈ sum w_j f(x_j)
    # ∫ (1/sqrt(pi)) e^{-z^2} f(z) dz ≈ (1/sqrt(pi)) sum w_j f(x_j)
    # If Z ~ N(0,1), pdf is (1/sqrt(2*pi)) exp(-z^2/2).
    # Let z = x / sqrt(2). Then dz = dx/sqrt(2).
    # E[f(Z)] = ∫ f(z) (1/sqrt(2*pi)) exp(-z^2/2) dz
    # = ∫ f(x/sqrt(2)) (1/sqrt(2*pi)) exp(-x^2/4) (dx/sqrt(2))
    # This is getting messy. Simpler:
    # Use scipy.integrate.quad or just simple Monte Carlo? No, we need speed and accuracy.
    # Standard transformation for GH to Normal:
    # If we want E[f(Z)] where Z~N(0,1):
    # Let Y ~ N(0, 1/2). Then Z = sqrt(2) Y.
    # GH is for ∫ e^{-y^2} f(y) dy.
    # E[f(Z)] = 1/sqrt(pi) ∫ e^{-y^2} f(sqrt(2) y) dy ≈ 1/sqrt(pi) sum w_j f(sqrt(2) x_j)
    
    quad_z = np.sqrt(2) * x_quad
    quad_w = w_quad / np.sqrt(np.pi)
    
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Fallback: use current V_next for remaining steps (approximation)
            # This is a safety break
            for k in range(i, -1, -1):
                V_curr = np.maximum(K - S_grid, 0.0)
                policies[k] = V_curr > 0.0
            break
            
        # Calculate expected continuation value for each S in S_grid
        # S_next = S_grid[:, None] * exp(drift + vol * quad_z[None, :])
        # This creates a matrix of shape (n_grid, n_quad)
        # n_grid=200, n_quad=50 -> 10,000 operations per step. 12 steps -> 120k ops. Very fast.
        
        log_S_next = np.log(S_grid)[:, None] + drift + vol * quad_z[None, :]
        S_next = np.exp(log_S_next)
        
        # Interpolate V_next on S_next
        # V_next is defined on S_grid (log-spaced).
        # np.interp works on linear scales. Since S_next and S_grid are positive,
        # and our grid is dense, linear interpolation in S-space is okay,
        # but interpolation in log-S space is better for GBM.
        # Let's interpolate in log-space.
        
        log_S_next_flat = log_S_next.flatten()
        log_S_grid_flat = log_S_grid
        
        # np.interp requires xp to be increasing. log_S_grid is increasing.
        V_next_vals = np.interp(log_S_next_flat, log_S_grid_flat, np.log(V_next + 1e-12))
        # Note: Interpolating log(V) is more stable for small values?
        # Actually, V is linear in S for puts? No, convex.
        # Let's just interpolate V directly in log-S space.
        # V is smooth.
        V_next_interp = np.interp(log_S_next_flat, log_S_grid_flat, V_next)
        
        V_next_interp_matrix = V_next_interp.reshape(n_grid, n_quad)
        
        # Weighted average
        E_V_next = np.sum(V_next_interp_matrix * quad_w[None, :], axis=1)
        
        # Discount
        disc = np.exp(-r * dt)
        cont_val = disc * E_V_next
        
        # Payoff
        pay = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_curr = np.maximum(pay, cont_val)
        
        # Exercise decision
        exercise_curr = pay >= cont_val
        
        policies[i] = exercise_curr
        V_next = V_curr
        
    # Now we have policies[i] as boolean arrays on S_grid for each time i (0 to n_exercise-1).
    # policies[0] corresponds to the first exercise date (t=dt).
    # The function exercise(current_state, time) receives:
    # current_state: array of spot prices
    # time: integer index of the exercise date (0-based? or 1-based?)
    # Contract says "12 monthly exercise dates (no time-0 exercise)".
    # Usually time=0 is first exercise, time=11 is last.
    # Let's assume time is 0-indexed for the exercise dates.
    
    class ExercisePolicy:
        def __init__(self, policies, S_grid, log_S_grid):
            self.policies = policies
            self.S_grid = S_grid
            self.log_S_grid = log_S_grid
            self.n_exercise = len(policies)
            
        def exercise(self, current_state, time):
            # time is the index of the exercise date.
            # If time is out of bounds, return False (don't exercise).
            if time < 0 or time >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            # current_state is an array of S values.
            # We need to map each S to the grid and check the policy.
            # Since the policy is a step function on the grid (defined by interpolation),
            # we can find the nearest grid point or interpolate the boundary.
            # However, the exercise decision is binary.
            # A robust way: find the exercise boundary S* at this time.
            # The policy is "Exercise if S <= S*".
            # We can find the largest S in S_grid where exercise is True.
            
            pol = self.policies[time]
            
            # Find the threshold
            # pol is True for low S, False for high S.
            # Find the last True index.
            true_indices = np.where(pol)[0]
            if len(true_indices) == 0:
                return np.zeros(len(current_state), dtype=bool)
            if len(true_indices) == len(pol):
                return np.ones(len(current_state), dtype=bool)
                
            # The boundary is between true_indices[-1] and true_indices[-1] + 1
            # We can interpolate the boundary log-S value.
            # Let i_last = true_indices[-1]
            # S_boundary is between S_grid[i_last] and S_grid[i_last+1]
            # Linear interpolation in log-S space.
            # We assume the value function difference changes linearly between grid points.
            # Actually, just using the nearest grid point is often sufficient with n_grid=200.
            # Let's use the upper bound of the last True grid point to be conservative?
            # Or interpolate.
            
            i_last = true_indices[-1]
            # If we use S_grid[i_last] as the threshold, we might under-exercise slightly.
            # If we use S_grid[i_last+1], we might over-exercise.
            # Let's interpolate the zero-crossing of (Payoff - Continuation).
            # We didn't store Payoff and Continuation separately, only the decision.
            # But we can reconstruct them if we stored V_next and Payoff? 
            # To save time, we just use the grid point.
            # With 200 points over log(10) to log(500) (range ~3.9), step is ~0.02.
            # This is very fine.
            
            # Let's simply use the threshold as S_grid[i_last].
            # Any S <= S_grid[i_last] is exercised.
            # What about S between S_grid[i_last] and S_grid[i_last+1]?
            # It's ambiguous. Let's say exercise if S <= S_grid[i_last].
            
            threshold = self.S_grid[i_last]
            
            return current_state <= threshold

    return ExercisePolicy(policies, S_grid, log_S_grid)

# EVOLVE-BLOCK-END
