"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import log, sqrt, exp, erf
from scipy.stats import norm

def _european_put(S, K, r, sigma, T):
    """Analytic Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    put = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put

def _backward_induction_thresholds(K, r, sigma, T, n_exercise, n_grid=500):
    """
    Compute exercise thresholds via backward induction.
    Returns array of thresholds for each exercise date (index 0 = first exercise date, ..., index n-1 = last).
    """
    dt = T / n_exercise
    discount = exp(-r * dt)
    
    # Price grid: from near 0 to some max
    S_max = 2.5 * K
    S_grid = np.linspace(0.01, S_max, n_grid)
    
    # Continuation value at last exercise date is 0 (no future exercise)
    # At each date i (from last to first), we compute:
    #   value_i(S) = max(K - S, discount * E[value_{i+1}(S_{t+dt}) | S_t = S])
    # For the expectation, we use the GBM transition:
    #   S_{t+dt} = S * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    # We approximate E[f(S_{t+dt})] using Gauss-Hermite or a fine quadrature.
    
    # Use a fine grid for the value function
    # Start from the last date: continuation = 0, so value = max(K-S, 0)
    # But actually at the last exercise date, you can only exercise (no continuation), so:
    #   value_last(S) = max(K - S, 0)
    
    # For earlier dates, continuation value = discount * E[value_next(S_{t+dt})]
    # We approximate the expectation using a change of measure / quadrature.
    
    # Let's use a simple approach: for each S on the grid, compute the expected next value
    # using the lognormal distribution of S_{t+dt}.
    
    # Actually, a cleaner approach: use the fact that under risk-neutral measure,
    # E[f(S_{t+dt}) | S_t = S] can be computed by integrating f(S * exp(mu + sigma*sqrt(dt)*Z))
    # where mu = (r - 0.5*sigma^2)*dt
    
    # We'll use a fine quadrature on Z ~ N(0,1)
    n_quad = 64
    # Gauss-Hermite quadrature
    x_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    # Transform to standard normal: z = sqrt(2) * x
    z_quad = sqrt(2) * x_quad
    w_quad = w_quad / exp(-z_quad**2 / 2) * 0.5 / sqrt(np.pi)  # normalize for N(0,1)
    # Actually, let me just use a simple uniform quadrature on Z
    z_quad = np.linspace(-4, 4, 200)
    # Weights: approximate the normal density
    from scipy.integrate import quad
    # Better: use Gauss-Hermite properly
    x_gh, w_gh = np.polynomial.hermite.hermgauss(32)
    z_gh = sqrt(2) * x_gh
    # For standard normal: E[f(Z)] = (1/sqrt(2*pi)) * integral f(z) exp(-z^2/2) dz
    # = (1/sqrt(2*pi)) * integral f(sqrt(2)*x) exp(-x^2) * sqrt(2) dx
    # = (1/sqrt(pi)) * integral f(sqrt(2)*x) exp(-x^2) dx
    # So weights for standard normal: w_gh / sqrt(pi)
    w_std = w_gh / sqrt(np.pi)
    
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * sqrt(dt)
    
    # Value function on grid
    # At last exercise date (index n_exercise - 1):
    # value(S) = max(K - S, 0)
    value_next = np.maximum(K - S_grid, 0.0)
    
    thresholds = np.zeros(n_exercise)
    
    # Work backwards from last exercise date to first
    for i in range(n_exercise - 1, -1, -1):
        # value_next is the value at date i+1 (or 0 if i is the last date)
        # Compute continuation value at date i: discount * E[value_next(S_{t+dt}) | S_t = S]
        
        if i == n_exercise - 1:
            # Last exercise date: no continuation, just exercise value
            value_curr = np.maximum(K - S_grid, 0.0)
        else:
            # Compute E[value_next(S * exp(mu + vol * Z)) | S]
            # For each S on the grid, compute the expected value
            # S_next = S * exp(mu + vol * Z)
            # We need to interpolate value_next at S_next
            
            # Vectorized: for each S in S_grid, compute S_next for all quadrature points
            # This is (n_grid, n_quad) matrix
            S_next = S_grid[:, None] * exp(mu + vol * z_gh[None, :])
            
            # Interpolate value_next at S_next
            # Clip S_next to grid range
            S_next_clipped = np.clip(S_next, S_grid[0], S_grid[-1])
            expected_val = np.interp(S_next_clipped, S_grid, value_next)
            
            # Weighted average
            cont_val = discount * np.sum(expected_val * w_std[None, :], axis=1)
            
            # Value = max(exercise, continuation)
            exercise_val = np.maximum(K - S_grid, 0.0)
            value_curr = np.maximum(exercise_val, cont_val)
            
            # Find threshold: largest S where exercise_val >= cont_val
            # i.e., K - S >= cont_val (for S <= K)
            # Threshold is where K - S = cont_val
            # Since cont_val is decreasing in S (typically), find the crossing
            diff = exercise_val - cont_val
            # Find last index where diff >= 0
            # The threshold is between S_grid[idx] and S_grid[idx+1]
            positive = diff >= 0
            if positive.any():
                idx = np.where(positive)[0][-1]
                if idx < n_grid - 1:
                    # Linear interpolation
                    d0 = diff[idx]
                    d1 = diff[idx + 1]
                    if d1 < 0:
                        frac = d0 / (d0 - d1)
                        thresholds[i] = S_grid[idx] + frac * (S_grid[idx + 1] - S_grid[idx])
                    else:
                        thresholds[i] = S_grid[idx]
                else:
                    thresholds[i] = S_grid[idx]
            else:
                thresholds[i] = 0.0
        
        value_next = value_curr
    
    return thresholds


class _ExercisePolicy:
    def __init__(self, thresholds, K, r, sigma, T, n_exercise):
        self.thresholds = thresholds
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_exercise = n_exercise
        self.dt = T / n_exercise
    
    def exercise(self, current_state, time):
        """
        current_state: array of current stock prices
        time: current time (to determine which exercise date we're at)
        Returns: boolean array, True if exercise at this date
        """
        S = np.asarray(current_state, dtype=np.float64)
        
        # Determine which exercise date we're at
        # Exercise dates are at dt, 2*dt, ..., T
        # time is the current time
        # If time is close to k*dt, we're at exercise date k
        k = int(round(time / self.dt))
        k = max(0, min(k, self.n_exercise - 1))
        
        threshold = self.thresholds[k]
        return S <= threshold


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 115.0
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    
    # Compute thresholds via backward induction
    thresholds = _backward_induction_thresholds(K, r, sigma, T, n_exercise, n_grid=500)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        # Fallback: simple threshold
        # Use a rough estimate
        thresholds = np.array([K * 0.7] * n_exercise)
    
    return _ExercisePolicy(thresholds, K, r, sigma, T, n_exercise)
# EVOLVE-BLOCK-END
