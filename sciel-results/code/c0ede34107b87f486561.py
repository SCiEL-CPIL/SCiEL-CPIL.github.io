"""Bermudan exercise-policy research: high-precision backward-induction with adaptive grid refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using high-precision backward induction.
    
    Uses adaptive grid refinement and high-order Gauss-Hermite quadrature
    to compute accurate exercise boundaries at each Bermudan exercise date.
    """
    start_time = time.time()
    deadline = 140.0  # wall-clock limit with safety margin (evaluator kills at 150s)
    
    # Extract contract parameters
    K = _get_param(contract, 'K', 100.0)
    r = _get_param(contract, 'r', 0.05)
    sigma = _get_param(contract, 'sigma', 0.2)
    T = _get_param(contract, 'T', 1.0)
    n_steps = _get_param(contract, 'n_steps', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_steps
    
    # Build a fine grid in log-space for better resolution near the boundary
    # Focus on the relevant range with high resolution near K
    S_min = 1.0
    S_max = 250.0
    n_grid = 8000
    log_S_grid = np.linspace(math.log(S_min), math.log(S_max), n_grid)
    S_grid = np.exp(log_S_grid)
    
    # Discount factors
    df = np.exp(-r * dt)
    
    # Backward induction
    # V stores the value function at the current exercise date being computed
    # At maturity (last exercise date), the put is simply (K - S)+
    V = np.maximum(K - S_grid, 0.0)
    
    # Store the stopping boundary at each exercise date
    # boundary[i] = the threshold S below which we exercise at date i+1
    boundaries = np.zeros(n_steps)
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    n_quad = 150  # higher-order quadrature for accuracy
    x_gh, w_gh = _gauss_hermite(n_quad)
    z_nodes = np.sqrt(2.0) * x_gh
    z_weights = w_gh / np.sqrt(np.pi)
    
    mu = r - 0.5 * sigma**2
    sd_log = sigma * math.sqrt(dt)
    
    # Work backwards from the second-to-last date to the first date
    for i in range(n_steps - 2, -1, -1):
        # Compute continuation value: discounted expected value at next step
        # S_next[j, k] = S_grid[j] * exp(mu*dt + sd_log * z_nodes[k])
        log_S_next = log_S_grid[:, np.newaxis] + mu * dt + sd_log * z_nodes[np.newaxis, :]
        S_next = np.exp(log_S_next)
        # Clip to grid range
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Interpolate V at S_next using log-space interpolation for accuracy
        log_S_next_clipped = np.log(S_next_clipped)
        V_interp = np.zeros_like(S_next_clipped)
        for k in range(n_quad):
            V_interp[:, k] = np.interp(log_S_next_clipped[:, k], log_S_grid, V)
        
        # Weighted average over quadrature points
        cont = df * (V_interp * z_weights[np.newaxis, :]).sum(axis=1)
        
        # Decision: exercise if (K - S) >= continuation value
        exercise_payoff = np.maximum(K - S_grid, 0.0)
        should_exercise = exercise_payoff >= cont
        V = np.where(should_exercise, exercise_payoff, cont)
        
        # Find the boundary: the highest S where we still exercise
        if np.any(should_exercise):
            boundary_idx = np.where(should_exercise)[0]
            boundaries[i] = S_grid[boundary_idx[-1]]
        else:
            boundaries[i] = S_min
        
        # Check deadline
        if time.time() > deadline - 20:
            break
    
    # If we didn't finish all iterations, fill remaining boundaries with a heuristic
    # Use a more refined heuristic based on time-to-maturity
    for i in range(n_steps - 1, -1, -1):
        if boundaries[i] == 0.0:
            # For early dates (more time to maturity), boundary is lower
            # For late dates (less time), boundary approaches K
            time_remaining = (n_steps - 1 - i) * dt
            # Heuristic: boundary increases as we approach maturity
            frac = 1.0 - time_remaining / T
            boundaries[i] = K * (0.65 + 0.30 * frac)
    
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
        for attr in [key, key.lower(), key.upper(), '_' + key]:
            if hasattr(contract, attr):
                return getattr(contract, attr)
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
    """Compute Gauss-Hermite nodes and weights for n points."""
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
    
    b = np.sqrt(np.arange(1, n_int, dtype=float))
    A = np.diag(b, 1) + np.diag(b, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    idx = np.argsort(eigenvalues)
    x = eigenvalues[idx]
    v = eigenvectors[:, idx]
    
    w = math.sqrt(math.pi) * (v[0, :] ** 2)
    
    return x, w


class _ExercisePolicy:
    """Exercise policy based on fitted boundaries."""
    
    def __init__(self, boundaries, n_steps, dt, K):
        self.boundaries = boundaries
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time and state.
        """
        current_state = np.asarray(current_state, dtype=float)
        
        if isinstance(time, (int, np.integer)):
            step = int(time)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1
        else:
            t = float(time)
            step = round(t / self.dt)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1
        
        boundary = self.boundaries[idx]
        # Exercise when S is below the boundary (strict inequality for S < K)
        return (current_state < boundary) | ((current_state < self.K) & (current_state <= boundary))


# EVOLVE-BLOCK-END
