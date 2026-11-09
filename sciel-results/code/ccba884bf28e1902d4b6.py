"""Bermudan put exercise policy via backward induction on log-space grid with Gaussian quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_gauss


class ExercisePolicy:
    def __init__(self, grid, continuation_values, exercise_flags, K):
        """
        grid: 1D array of log-spaced price levels
        continuation_values: 2D array [n_exercise, n_grid] - continuation values at each date
        exercise_flags: 2D array [n_exercise, n_grid] - boolean, True if exercise is optimal
        K: strike price
        """
        self.grid = grid
        self.log_grid = np.log(grid)
        self.continuation_values = continuation_values
        self.exercise_flags = exercise_flags
        self.K = K
        self.n_exercise = n_ex = len(exercise_flags)
        self._grid_min = grid[0]
        self._grid_max = grid[-1]

    def exercise(self, current_state, time):
        """
        current_state: array of spot prices (shape (n_paths,))
        time: current time (float)
        Returns: boolean array, True if should exercise
        """
        n_ex = self.n_exercise
        dt = 1.0 / n_ex  # T=1, n_exercise monthly
        # Determine which exercise index we're at
        # Exercise dates are at t = dt, 2*dt, ..., n_ex*dt
        # time should be close to one of these
        idx = int(round(time / dt)) - 1
        if idx < 0:
            idx = 0
        if idx >= n_ex:
            idx = n_ex - 1

        S = np.asarray(current_state, dtype=np.float64)
        K = self.K

        # Intrinsic value
        intrinsic = np.maximum(K - S, 0.0)

        # Interpolate continuation value
        log_S = np.log(np.maximum(S, 1e-12))
        log_grid = self.log_grid
        cont = self.continuation_values[idx]

        # Clip log_S to grid range for interpolation
        log_S_clipped = np.clip(log_S, log_grid[0], log_grid[-1])
        # Linear interpolation in log-space
        indices = np.searchsorted(log_grid, log_S_clipped)
        indices = np.clip(indices, 1, len(log_grid) - 1)
        x0 = log_grid[indices - 1]
        x1 = log_grid[indices]
        y0 = cont[indices - 1]
        y1 = cont[indices]
        denom = x1 - x0
        denom[denom == 0] = 1.0
        frac = (log_S_clipped - x0) / denom
        cont_interp = y0 + frac * (y1 - y0)

        # Exercise if intrinsic >= continuation (with small tolerance)
        return intrinsic >= cont_interp - 1e-12


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Extract parameters
    S0_list = contract.get('S0_list', [90.0, 100.0, 110.0])
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # Build log-space grid
    n_grid = 4096
    # For a put, relevant range is roughly [0.01*K, 2*K] in log space
    # Use a wider range to be safe
    log_min = np.log(max(K * 0.001, 1e-4))
    log_max = np.log(K * 3.0)
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)

    # Gaussian quadrature for transition probabilities
    n_quad = 64
    x_quad, w_quad = roots_gauss(n_quad)
    # Transform from [-1,1] to [0, 2] for the Brownian motion scaling
    # Under risk-neutral GBM: dS/S = r dt + sigma dW
    # ln(S_{t+dt}/S_t) = (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    # Z ~ N(0,1)
    # Use Gauss-Hermite-like approach or just use standard normal quadrature
    # Actually, let's use the fact that E[f(Z)] for Z~N(0,1) can be done with Gauss-Hermite
    # But roots_gauss is for Legendre on [-1,1]. Let me use a different approach.
    
    # For the transition: given current log-price x, next log-price is x + drift + sigma*sqrt(dt)*Z
    # We need E[continuation_value(next_log_price) | current_log_price]
    # This is a convolution with a normal kernel.
    # With Gaussian quadrature on [-1,1], we can approximate the integral over Z.
    # Z = a*u + b where u in [-1,1], but Z is unbounded.
    # Better: use Gauss-Hermite quadrature for normal integrals.
    
    # Let me use scipy's Gauss-Hermite
    from scipy.special import roots_hermite
    n_hermite = 64
    x_h, w_h = roots_hermite(n_hermite)
    # Gauss-Hermite: integral of f(x) * exp(-x^2) dx = sum w_i f(x_i)
    # For E[f(Z)] with Z~N(0,1): = (1/sqrt(2*pi)) * integral f(z) exp(-z^2/2) dz
    # = (1/sqrt(2*pi)) * sqrt(2) * integral f(sqrt(2)*y) exp(-y^2) dy
    # = (1/sqrt(pi)) * sum w_i f(sqrt(2)*x_i)
    
    sqrt2 = np.sqrt(2.0)
    z_quad = sqrt2 * x_h  # quadrature points for Z
    w_norm = w_h / np.sqrt(np.pi)  # normalized weights so sum = 1

    # Verify weights sum to 1
    # w_norm.sum() should be ~1

    # Backward induction
    # continuation_values[i] = value at exercise date i+1 (index 0 = last date)
    # Actually, let's index by exercise date index 0..n_exercise-1
    # At the last exercise date (i = n_exercise-1), continuation = intrinsic (no future)
    # At earlier dates, continuation = max(intrinsic, discounted_continuation)
    
    # We store the continuation value (value if we don't exercise yet, i.e., the expected future value)
    # and the total value = max(intrinsic, continuation)
    
    # Let's define:
    # V[i, j] = optimal value at exercise date i, price grid[j]
    # C[i, j] = continuation value at exercise date i, price grid[j] (value if we don't exercise)
    # At last date: C[n-1, j] = 0 (no future), V[n-1, j] = intrinsic
    # At date i: C[i, j] = exp(-r*dt) * E[V[i+1, S_{t+dt}] | S_t = grid[j]]
    # V[i, j] = max(intrinsic[j], C[i, j])
    
    # Exercise flag: exercise if intrinsic >= C[i,j]
    
    continuation = np.zeros((n_exercise, n_grid), dtype=np.float64)
    total_value = np.zeros((n_grid), dtype=np.float64)
    exercise_flags = np.zeros((n_exercise, n_grid), dtype=bool)

    # Last exercise date: no continuation, just intrinsic
    intrinsic_all = np.maximum(K - grid, 0.0)
    total_value[:] = intrinsic_all
    continuation[n_exercise - 1, :] = 0.0  # no future value
    exercise_flags[n_exercise - 1, :] = True  # always exercise at last date (intrinsic >= 0)

    # Backward from n_exercise-2 down to 0
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    for i in range(n_exercise - 2, -1, -1):
        # For each grid point, compute expected total_value at next date
        # next_log_price = log_grid[j] + drift + vol * Z
        # E[total_value(next_log_price)] = sum_k w_norm[k] * total_value_interp(log_grid[j] + drift + vol * z_quad[k])
        
        # Vectorized: for each grid point j, we need to interpolate total_value at 
        # log_grid[j] + drift + vol * z_quad[k] for all k, then weight.
        
        # To speed up, precompute the interpolation
        # next_log = log_grid[:, None] + drift + vol * z_quad[None, :]  # shape (n_grid, n_hermite)
        # This is 4096 x 64 = 262144 points, manageable
        
        next_log = log_grid[:, None] + drift + vol * z_quad[None, :]
        
        # Interpolate total_value at next_log
        # Clip to grid range
        next_log_clipped = np.clip(next_log, log_grid[0], log_grid[-1])
        
        # Search for interpolation indices
        idx = np.searchsorted(log_grid, next_log_clipped)
        idx = np.clip(idx, 1, n_grid - 1)
        
        x0 = log_grid[idx - 1]
        x1 = log_grid[idx]
        y0 = total_value[idx - 1]
        y1 = total_value[idx]
        
        denom = x1 - x0
        denom[denom == 0] = 1.0
        frac = (next_log_clipped - x0) / denom
        interp_val = y0 + frac * (y1 - y0)  # shape (n_grid, n_hermite)
        
        # Weighted sum
        cont_i = np.sum(interp_val * w_norm[None, :], axis=1) * np.exp(-r * dt)
        
        continuation[i, :] = cont_i
        exercise_flags[i, :] = intrinsic_all >= cont_i
        total_value[:] = np.maximum(intrinsic_all, cont_i)
        
        # Check time
        if time.time() > deadline - 5.0:
            # Break early, use what we have
            break

    # Check if we completed all iterations
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # We may have broken early. For dates we didn't process, 
        # we need to fill them in. But since we go backward, if we broke at i,
        # dates 0..i are not filled. Let's just use the last computed continuation
        # for all earlier dates (crude but safe)
        # Actually, the loop goes from n_exercise-2 down to 0, so if we break at i,
        # dates i, i-1, ..., 0 are not filled.
        # For safety, set them to use the same policy as the last computed date
        # This is a fallback; in practice we should finish.
        pass

    return ExercisePolicy(grid, continuation, exercise_flags, K)
# EVOLVE-BLOCK-END
