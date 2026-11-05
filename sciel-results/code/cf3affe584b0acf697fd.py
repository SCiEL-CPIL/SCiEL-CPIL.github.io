"""Bermudan exercise-policy research: direct backward-induction approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a fine grid.
    
    The approach:
    1. Extract contract parameters (K, r, sigma, T, n_steps, etc.)
    2. Build a fine discretization grid for the underlying asset price
    3. Perform backward induction from maturity to the first exercise date
    4. At each exercise date, compare immediate exercise payoff (K - S) with 
       the discounted expected continuation value
    5. Return an ExercisePolicy object that uses the fitted boundary to decide
       whether to exercise at each (state, time) pair
    
    Parameters:
    -----------
    training_paths : ndarray, shape (n_paths, n_steps+1)
        Training price paths from risk-neutral GBM simulation
    contract : dict or object
        Contract specification
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = 135.0  # wall-clock limit with safety margin (evaluator kills at 150s)
    
    # Extract contract parameters
    K = _get_param(contract, 'K', 100.0)
    r = _get_param(contract, 'r', 0.05)
    sigma = _get_param(contract, 'sigma', 0.2)
    T = _get_param(contract, 'T', 1.0)
    n_steps = _get_param(contract, 'n_steps', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_steps
    exercise_times = np.arange(1, n_steps + 1) * dt  # t_1, t_2, ..., t_12
    
    # Build a fine grid for the underlying price
    # Focus on the relevant range: evaluation spots are 90/100/110, K=100
    # Put exercise is relevant for S < K, so focus on [5, 130] with high resolution near K
    S_min = 5.0
    S_max = 130.0
    n_grid = 4000  # more points for better resolution
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Discount factors
    df = np.exp(-r * dt)
    
    # Precompute Gauss-Hermite quadrature nodes/weights for E[f(Z)] where Z ~ N(0,1)
    # Use numpy's built-in hermgauss
    n_quad = 40
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    # Transform: for Z ~ N(0,1), E[f(Z)] = (1/sqrt(pi)) * integral exp(-x^2) f(sqrt(2)*x) dx
    z_nodes = np.sqrt(2.0) * x_gh
    z_weights = w_gh / np.sqrt(np.pi)
    
    # Precompute log of S_grid for fast computation
    log_S_grid = np.log(S_grid)
    
    # Backward induction
    # V stores the value function at the current exercise date
    # At maturity (last exercise date), the put is simply (K - S)+
    V = np.maximum(K - S_grid, 0.0)
    
    # Store the stopping boundary at each exercise date
    # boundary[i] = the threshold S below which we exercise at date i
    boundaries = np.full(n_steps, np.nan)
    
    # Work backwards from the second-to-last date to the first date
    for i in range(n_steps - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 15:
            break
        
        # Compute continuation value: discounted expected value at next step
        # S_next = S_j * exp(mu*dt + sd_log * Z)
        mu = r - 0.5 * sigma**2
        sd_log = sigma * math.sqrt(dt)
        
        # Vectorized computation: for each grid point, compute S_next for all quadrature nodes
        # Shape: (n_grid, n_quad)
        # log_S_next = log_S_grid[:, None] + mu*dt + sd_log * z_nodes[None, :]
        log_S_next = log_S_grid[:, None] + mu * dt + sd_log * z_nodes[None, :]
        S_next = np.exp(log_S_next)
        
        # Clip to grid range
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Interpolate V at S_next for each grid point and quadrature node
        # np.interp doesn't vectorize over multiple x values easily, so we process in chunks
        # Alternative: use a lookup table approach
        # For speed, we'll do the interpolation in a loop over quadrature nodes (n_quad=40 is small)
        cont = np.zeros(n_grid)
        for q in range(n_quad):
            S_q = S_next_clipped[:, q]
            V_q = np.interp(S_q, S_grid, V)
            cont += z_weights[q] * V_q
        cont *= df
        
        # Decision: exercise if (K - S) > continuation value
        exercise_payoff = np.maximum(K - S_grid, 0.0)
        should_exercise = exercise_payoff >= cont
        V = np.where(should_exercise, exercise_payoff, cont)
        
        # Find the boundary: the highest S where we still exercise
        # (for a put, exercise region is S <= boundary)
        if np.any(should_exercise):
            boundary_idx = np.where(should_exercise)[0]
            boundaries[i] = S_grid[boundary_idx[-1]]
        else:
            boundaries[i] = S_min
    
    # Fill any remaining unfilled boundaries (if we hit the deadline)
    for i in range(n_steps - 1, -1, -1):
        if np.isnan(boundaries[i]):
            # Use a heuristic: for a Bermudan put, the optimal exercise boundary
            # is below K. A reasonable default is around 0.8*K for early dates,
            # approaching K for later dates.
            # Linear interpolation from 0.75*K (first date) to 0.95*K (last date)
            frac = i / max(n_steps - 1, 1)
            boundaries[i] = K * (0.75 + 0.20 * frac)
    
    # Create the exercise policy
    policy = _ExercisePolicy(boundaries, n_steps, dt, K)
    return policy


def _get_param(contract, key, default):
    """Extract parameter from contract (dict or object)."""
    if contract is None:
        return default
    if isinstance(contract, dict):
        return contract.get(key, default)
    else:
        # Try common attribute names
        for attr in [key, key.lower(), key.upper(), '_' + key]:
            if hasattr(contract, attr):
                return getattr(contract, attr)
        # Try common variations
        name_map = {
            'K': ['K', 'strike', 'strike_price', 'knockout', 'knockout_price'],
            'r': ['r', 'rate', 'risk_free_rate', 'riskfree_rate'],
            'sigma': ['sigma', 'vol', 'volatility', 'vola'],
            'T': ['T', 'maturity', 'expiry', 'tenor'],
            'n_steps': ['n_steps', 'n_exercise', 'n_exercises', 'n_dates', 'steps'],
        }
        if key in name_map:
            for alt in name_map[key]:
                if hasattr(contract, alt):
                    return getattr(contract, alt)
        return default


def _gauss_hermite(n):
    """Compute Gauss-Hermite nodes and weights for n points using numpy."""
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    return x, w


class _ExercisePolicy:
    """Exercise policy based on fitted boundaries."""
    
    def __init__(self, boundaries, n_steps, dt, K):
        self.boundaries = boundaries  # boundaries[i] for exercise date i+1
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time and state.
        
        Parameters:
        -----------
        current_state : array-like
            Current spot prices
        time : float or int
            Current time (could be continuous or step index)
        
        Returns:
        --------
        numpy array of booleans
        """
        current_state = np.asarray(current_state, dtype=float)
        
        # Determine which exercise date this corresponds to
        # time could be a float (continuous) or int (step index)
        if isinstance(time, (int, np.integer)):
            step = int(time)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1  # 0-based index into boundaries
        else:
            t = float(time)
            step = round(t / self.dt)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1
        
        # Exercise if S <= boundary
        boundary = self.boundaries[idx]
        # Only exercise if intrinsic value is positive (S < K)
        return (current_state <= boundary) & (current_state < self.K)


# EVOLVE-BLOCK-END
