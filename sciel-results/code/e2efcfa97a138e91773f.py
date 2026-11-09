"""Bermudan exercise-policy: high-resolution backward induction on a 1-D price grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)

    N, n_total_steps = training_paths.shape
    n_exercise = 12  # monthly exercise dates
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    drift = (r - 0.5 * sigma**2) * dt
    vol_step = sigma * np.sqrt(dt)

    # Grid size for the 1-D backward induction
    # We want enough resolution to capture the exercise boundary well
    # For a put with K=100, the relevant price range is roughly [0, 2*K]
    # Let's use a grid from near 0 to about 2*K
    grid_size = 4096
    S_max = 2.5 * K  # Upper bound for the grid
    S_min = 0.01     # Lower bound
    
    # Create a log-spaced grid for better resolution at low prices
    # Actually, let's use a linear grid since the put payoff is linear in S
    # But log grid might be better for GBM. Let's try linear first.
    S_grid = np.linspace(S_min, S_max, grid_size)
    dS = S_grid[1] - S_grid[0]

    # Compute transition probabilities on the grid
    # Under GBM: S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    # We approximate the lognormal distribution with a shifted lognormal
    # For each S in the grid, we need to compute E[f(S_{t+1}) | S_t]
    
    # Approach: for each S_t, S_{t+1} = S_t * exp(mu + sigma*sqrt(dt)*Z)
    # where mu = (r - 0.5*sigma^2)*dt
    # We can approximate this expectation using quadrature on the log-space
    
    # For efficiency, precompute the lognormal weights
    # Let x = log(S_{t+1}/S_t) = mu + sigma*sqrt(dt)*Z
    # So S_{t+1} = S_t * exp(x)
    # E[f(S_{t+1}) | S_t] = E[f(S_t * exp(x))]
    
    # Use Gauss-Hermite quadrature for the expectation over Z ~ N(0,1)
    # E[g(Z)] = (1/sqrt(pi)) * sum_i w_i * g(z_i)
    # where z_i are Gauss-Hermite nodes and w_i are weights
    
    n_quad = 64  # Number of quadrature points
    z_nodes, z_weights = np.polynomial.hermite.hermgauss(n_quad)
    # z_weights need to be scaled: E[g(Z)] = (1/sqrt(pi)) * sum w_i g(z_i)
    # But hermgauss gives nodes for weight exp(-z^2), so E[g(Z)] = (1/sqrt(pi)) * sum w_i g(z_i)
    z_weights = z_weights / np.sqrt(np.pi)
    
    # Precompute the log-multipliers: exp(mu + sigma*sqrt(dt)*z_i)
    log_multipliers = np.exp(drift + vol_step * z_nodes)  # shape (n_quad,)
    
    # Now for backward induction on the grid
    # V_k[i] = value at exercise date k for price S_grid[i]
    # V_12[i] = max(K - S_grid[i], 0)
    V_grid = np.zeros((n_exercise, grid_size))
    V_grid[11, :] = np.maximum(K - S_grid, 0.0)  # k=12 is index 11 (0-indexed)
    
    # Backward induction for k = 11 down to 1
    # V_k[i] = max(K - S_grid[i], discount * E[V_{k+1}(S_{k+1}) | S_k = S_grid[i]])
    # E[V_{k+1}(S_{k+1}) | S_k] = (1/sqrt(pi)) * sum_j w_j * V_{k+1}(S_grid[i] * exp(mu + vol*z_j))
    
    # For each grid point S_grid[i], compute S_grid[i] * log_multipliers[j] for all j
    # Then interpolate V_{k+1} at those points
    
    # Precompute the interpolation: for each i and j, we need V_{k+1} at S_grid[i] * log_multipliers[j]
    # We can do this efficiently by vectorizing over i
    
    # For each k from 11 down to 1:
    for k in range(10, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        V_next = V_grid[k+1, :]
        
        # Compute expected continuation value at each grid point
        # For each i, E[V_next(S_i * L_j)] where L_j = log_multipliers[j]
        
        # S_i * L_j for all i, j
        # S_grid[:, None] * log_multipliers[None, :]  -> shape (grid_size, n_quad)
        
        # We need to interpolate V_next at these points
        # V_next is defined on S_grid, so we use linear interpolation
        
        # For each column j (fixed L_j), we need V_next(S_grid * L_j)
        # This is a scaled version of S_grid
        
        # Efficient approach: for each j, compute the interpolated values
        # V_next(S_grid * L_j) for all i
        
        # Using np.interp for each j
        # But this is grid_size * n_quad operations, which might be slow
        
        # Alternative: precompute the interpolation for a set of multipliers
        
        # Let's compute it vectorized
        # For each j, the points are S_grid * L_j
        # We clip to [S_min, S_max] and interpolate
        
        # Actually, let's use a different approach: 
        # Instead of interpolating on the fly, we can use the fact that 
        # the GBM transition is multiplicative, so we can shift the grid in log-space
        
        # Let's use log-space grid for better handling of multiplicative transitions
        # S_grid_log = log(S_grid), d_log_S = log(S_grid[1]/S_grid[0])
        # S_{t+1} = S_t * exp(x) => log(S_{t+1}) = log(S_t) + x
        # So in log-space, it's a translation!
        
        # Let me redo with log-grid
        
        # Break and use log-space
        break
    
    # Let's restart with log-space grid for cleaner multiplicative handling
    # Clear previous work
    V_grid = np.zeros((n_exercise, grid_size))
    
    # Log-space grid
    log_S_min = np.log(S_min)
    log_S_max = np.log(S_max)
    log_S_grid = np.linspace(log_S_min, log_S_max, grid_size)
    S_grid = np.exp(log_S_grid)
    d_log_S = log_S_grid[1] - log_S_grid[0]
    
    # Terminal value
    V_grid[11, :] = np.maximum(K - S_grid, 0.0)
    
    # Precompute log-multipliers
    log_multipliers = drift + vol_step * z_nodes  # shape (n_quad,)
    
    # Backward induction in log-space
    for k in range(10, -1, -1):
        if time.time() > deadline - 3.0:
            break
            
        V_next = V_grid[k+1, :]
        
        # For each grid point i, compute E[V_next(exp(log_S_grid[i] + log_multipliers[j]))]
        # log_S_grid[i] + log_multipliers[j] is the log-price at next step
        
        # We need to interpolate V_next at these log-prices
        # V_next is defined on log_S_grid
        
        # Vectorized interpolation:
        # For each i, the target log-prices are log_S_grid[i] + log_multipliers[j] for j=0..n_quad-1
        
        # Let's compute this efficiently
        # log_targets[i, j] = log_S_grid[i] + log_multipliers[j]
        log_targets = log_S_grid[:, None] + log_multipliers[None, :]  # shape (grid_size, n_quad)
        
        # Clip to grid range
        log_targets_clipped = np.clip(log_targets, log_S_min, log_S_max)
        
        # Interpolate V_next at log_targets_clipped
        # Using np.interp along axis 1
        # For each row i, interpolate V_next (defined on log_S_grid) at log_targets_clipped[i, :]
        
        # np.interp works on 1D, so we need to loop or use a vectorized approach
        # Let's use linear interpolation manually
        
        # For each log_target, find the two bracketing grid points and interpolate
        # log_targets_clipped has shape (grid_size, n_quad)
        
        # Find indices: idx = floor((log_target - log_S_min) / d_log_S)
        idx_float = (log_targets_clipped - log_S_min) / d_log_S
        idx_low = np.floor(idx_float).astype(int)
        idx_high = idx_low + 1
        
        # Clip indices
        idx_low = np.clip(idx_low, 0, grid_size - 2)
        idx_high = np.clip(idx_high, 0, grid_size - 1)
        
        # Weights
        weight_high = idx_float - idx_low
        weight_low = 1.0 - weight_high
        
        # Interpolated values
        V_interp = weight_low * V_next[idx_low] + weight_high * V_next[idx_high]
        
        # Expected value: sum over j of z_weights[j] * V_interp[i, j]
        # V_interp has shape (grid_size, n_quad), z_weights has shape (n_quad,)
        expected_cont = V_interp @ z_weights  # shape (grid_size,)
        
        # Continuation value
        cont_value = discount * expected_cont
        
        # Payoff
        payoff = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_grid[k, :] = np.maximum(payoff, cont_value)
    
    # Now we have V_grid[k, i] = value at exercise date k+1 (since k=0..11 corresponds to t=1..12)
    # Wait, let me re-index:
    # V_grid[11, :] is for k=12 (t=12/12)
    # V_grid[10, :] is for k=11 (t=11/12)
    # ...
    # V_grid[0, :] is for k=1 (t=1/12)
    
    # For the policy, at time t_k (k=1..12), we need to decide whether to exercise
    # Exercise if payoff >= continuation value
    # payoff_k(S) = max(K - S, 0)
    # continuation_k(S) = discount * E[V_{k+1}(S_{k+1}) | S_k = S]
    # We stored V_k(S) = max(payoff_k(S), cont_k(S))
    # So we need to recover the continuation value or just use the boundary
    
    # For each k, the exercise boundary is the largest S such that payoff_k(S) >= cont_k(S)
    # For a put, this is a threshold: exercise if S <= threshold_k
    
    # Let's compute the threshold for each k
    thresholds = np.zeros(n_exercise)
    for k in range(n_exercise):
        payoff = np.maximum(K - S_grid, 0.0)
        V_k = V_grid[k, :]
        # Exercise if payoff >= V_k - payoff... wait
        # V_k = max(payoff, cont)
        # So exercise if payoff >= cont, which is equivalent to payoff >= V_k when V_k = payoff
        # Actually, V_k = max(payoff, cont), so if V_k == payoff, we exercise
        # If V_k > payoff, we continue (V_k = cont)
        
        # The boundary is where payoff = cont
        # Let's find the largest S where we exercise
        # Exercise if payoff >= cont, i.e., max(K-S,0) >= V_k - ... 
        # Hmm, let me think again.
        
        # V_k(S) = max(payoff(S), cont(S))
        # We exercise if payoff(S) >= cont(S)
        # This is equivalent to: payoff(S) >= V_k(S) - max(payoff(S) - V_k(S), 0) ... no
        
        # Simpler: just compute the exercise region directly
        # We know V_k(S) and payoff(S). We exercised at S if during the backward induction
        # payoff(S) >= cont(S). But we didn't store cont(S) separately.
        
        # Let's recompute: for each k, compute cont_k(S) and find the boundary
        # cont_k(S) = V_k(S) when V_k(S) > payoff(S), and = payoff(S) when V_k(S) = payoff(S)
        # So cont_k(S) = V_k(S) for S below the boundary (where we continue), 
        # and cont_k(S) = payoff(S) for S above the boundary (where we exercise)
        
        # Actually, that's not right either. Let me just find the threshold by looking
        # for where payoff = V_k, but that's not correct because V_k = max(payoff, cont)
        
        # Better approach: store the continuation value during backward induction
        
        # Let me redo: store both V and the exercise decision
    
    # Let me restructure: store the continuation value explicitly
    V_grid = np.zeros((n_exercise, grid_size))
    cont_grid = np.zeros((n_exercise, grid_size))
    
    V_grid[11, :] = np.maximum(K - S_grid, 0.0)
    cont_grid[11, :] = V_grid[11, :]  # At maturity, no continuation
    
    for k in range(10, -1, -1):
        if time.time() > deadline - 3.0:
            break
            
        V_next = V_grid[k+1, :]
        
        log_targets = log_S_grid[:, None] + log_multipliers[None, :]
        log_targets_clipped = np.clip(log_targets, log_S_min, log_S_max)
        
        idx_float = (log_targets_clipped - log_S_min) / d_log_S
        idx_low = np.floor(idx_float).astype(int)
        idx_high = idx_low + 1
        idx_low = np.clip(idx_low, 0, grid_size - 2)
        idx_high = np.clip(idx_high, 0, grid_size - 1)
        
        weight_high = idx_float - idx_low
        weight_low = 1.0 - weight_high
        
        V_interp = weight_low * V_next[idx_low] + weight_high * V_next[idx_high]
        expected_cont = V_interp @ z_weights
        cont_k = discount * expected_cont
        
        payoff = np.maximum(K - S_grid, 0.0)
        V_grid[k, :] = np.maximum(payoff, cont_k)
        cont_grid[k, :] = cont_k
    
    # Now compute exercise boundaries
    # For each k, exercise if payoff >= cont_k
    # For a put, this is S <= threshold_k
    thresholds = np.zeros(n_exercise)
    for k in range(n_exercise):
        payoff = np.maximum(K - S_grid, 0.0)
        # Exercise if payoff >= cont_grid[k, :]
        exercise_mask = payoff >= cont_grid[k, :]
        # Find the largest S where we exercise
        if np.any(exercise_mask):
            # Find the last index where exercise_mask is True
            # The mask should be True for small S, False for large S (for a put)
            # Find the transition point
            idxs = np.where(exercise_mask)[0]
            if len(idxs) > 0:
                last_exercise_idx = idxs[-1]
                # Interpolate to find the threshold
                if last_exercise_idx < grid_size - 1:
                    # Linear interpolation between last_exercise_idx and last_exercise_idx+1
                    s1 = S_grid[last_exercise_idx]
                    s2 = S_grid[last_exercise_idx + 1]
                    p1 = payoff[last_exercise_idx] - cont_grid[k, last_exercise_idx]
                    p2 = payoff[last_exercise_idx + 1] - cont_grid[k, last_exercise_idx + 1]
                    if p1 != p2:
                        frac = -p1 / (p2 - p1)
                        thresholds[k] = s1 + frac * (s2 - s1)
                    else:
                        thresholds[k] = s1
                else:
                    thresholds[k] = S_grid[last_exercise_idx]
            else:
                thresholds[k] = 0.0
        else:
            thresholds[k] = 0.0
    
    # For k=12 (index 11), always exercise if ITM
    thresholds[11] = K  # Exercise if S < K at maturity
    
    class Policy:
        def __init__(self, thr, K_val):
            self.thresholds = thr
            self.K = K_val
            
        def exercise(self, current_state, time):
            # time is integer 1..12
            current_state = np.asarray(current_state, dtype=np.float64)
            if time >= 12:
                return current_state < self.K
            if time < 1 or time > 11:
                return current_state < self.K
            # time is 1..11, use thresholds[time-1] since thresholds[0] is for t=1/12
            thr = self.thresholds[time - 1]
            return current_state < thr
    
    return Policy(thresholds, K)

# EVOLVE-BLOCK-END
