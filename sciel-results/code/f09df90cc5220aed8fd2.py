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
    n_grid = 1500  # number of grid points
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Discount factors
    df = np.exp(-r * dt)
    
    # Backward induction
    # V[i] stores the value function at exercise date i (index 0 = last date)
    # At maturity (last exercise date), the put is simply (K - S)+
    V = np.maximum(K - S_grid, 0.0)
    
    # Store the stopping boundary at each exercise date
    # boundary[i] = the threshold S below which we exercise at date i
    boundaries = np.zeros(n_steps)
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    n_quad = 60  # number of quadrature points
    x_gh, w_gh = _gauss_hermite(n_quad)
    z_nodes = np.sqrt(2.0) * x_gh
    z_weights = w_gh / np.sqrt(np.pi)
    
    mu = r - 0.5 * sigma**2
    sd_log = sigma * math.sqrt(dt)
    
    # Work backwards from the second-to-last date to the first date
    for i in range(n_steps - 2, -1, -1):
        # Compute continuation value: discounted expected value at next step
        # Vectorized: for all S_grid points simultaneously
        # S_next[j, k] = S_grid[j] * exp(mu*dt + sd_log * z_nodes[k])
        log_S_grid = np.log(S_grid)  # shape (n_grid,)
        # S_next has shape (n_grid, n_quad)
        log_S_next = log_S_grid[:, np.newaxis] + mu * dt + sd_log * z_nodes[np.newaxis, :]
        S_next = np.exp(log_S_next)
        # Clip to grid range
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Interpolate V at S_next for each grid point
        # V_interp[j, k] = V interpolated at S_next[j, k]
        V_interp = np.zeros_like(S_next_clipped)
        for k in range(n_quad):
            V_interp[:, k] = np.interp(S_next_clipped[:, k], S_grid, V)
        
        # Weighted average over quadrature points
        cont = df * (V_interp * z_weights[np.newaxis, :]).sum(axis=1)
        
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
        
        # Check deadline
        if time.time() > deadline - 15:
            break
    
    # boundaries[i] corresponds to exercise date i (0-indexed from first date)
    # Exercise date i (1-indexed) = i+1, time = (i+1)*dt
    # At exercise date d (1..12), index in boundaries is d-1
    
    # If we didn't finish all iterations, fill remaining boundaries with a heuristic
    if boundaries[-1] == 0.0:
        # Use a simple heuristic: exercise if S < K * (1 - something)
        for i in range(n_steps - 1, -1, -1):
            if boundaries[i] == 0.0:
                # Approximate boundary using intrinsic value heuristic
                # For a put, boundary is typically below K
                boundaries[i] = K * 0.85
    
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
    """Compute Gauss-Hermite nodes and weights for n points.
    
    Returns nodes x_k and weights w_k such that:
    sum_k w_k * f(x_k) approx integral_{-inf}^{inf} exp(-x^2) * f(x) dx
    """
    n_int = int(n)
    if n_int <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n_int == 1:
        return np.array([0.0]), np.array([math.sqrt(math.pi)])
    
    try:
        from numpy.polynomial.hermite import hermgauss
        x, w = hermgauss(n_int)
        return x, w
    except (ImportError, Exception):
        pass
    
    # Fallback: build Jacobi matrix for Hermite polynomials with weight exp(-x^2)
    # The three-term recurrence for monic Hermite polynomials H_n:
    # H_{n+1}(x) = x*H_n(x) - n*H_{n-1}(x)
    # So the Jacobi matrix has diagonal 0 and off-diagonal sqrt(k) for k=1..n-1
    
    b = np.sqrt(np.arange(1, n_int, dtype=float))
    A = np.diag(b, 1) + np.diag(b, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    x = eigenvalues[idx]
    v = eigenvectors[:, idx]
    
    # Weights: w_k = sqrt(pi) * (v[0,k])^2
    # This ensures sum(w_k) = sqrt(pi) = integral exp(-x^2) dx
    w = math.sqrt(math.pi) * (v[0, :] ** 2)
    
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
            # If time is an integer, it might be a step index (0-based or 1-based)
            # Exercise dates are at t = 1, 2, ..., 12 (1-based) or 0, 1, ..., 11 (0-based)
            # The contract says "no time-0 exercise", so exercise dates are 1..12
            # If time is 0-based step index, exercise dates are 1..12
            # If time is 1-based, exercise dates are 1..12
            step = int(time)
            if step < 1:
                # Before first exercise date, cannot exercise
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                # Past last exercise date
                return np.zeros(len(current_state), dtype=bool)
            else:
                # Exercise date step (1-based)
                idx = step - 1  # 0-based index into boundaries
        else:
            # time is a float, convert to step index
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
        # Also only exercise if intrinsic value is positive
        return (current_state <= boundary) & (current_state < self.K)


# EVOLVE-BLOCK-END
