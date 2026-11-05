"""Bermudan put exercise-policy research with LSMC-based continuation value estimation."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time
from scipy.stats import norm

_DEADLINE = 142.0  # seconds; safety margin under 150 s wall-clock limit


def _bs_put_price(S, K, r, sigma, T):
    """European put price via Black-Scholes."""
    S = np.asarray(S, dtype=float)
    if T <= 1e-12:
        return np.maximum(K - S, 0.0)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    N = norm.cdf
    price = K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    return price


def _fit_regression(y, S, degree=3):
    """Fit polynomial regression of y on S (and powers). Returns coefficients."""
    n = len(S)
    # Features: 1, S, S^2, S^3
    X = np.column_stack([np.ones(n), S, S**2, S**3])
    # Ridge regularization for stability
    lam = 1e-4
    A = X.T @ X + lam * np.eye(4)
    b = X.T @ y
    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    return coeffs


def _predict_regression(S, coeffs):
    """Predict from polynomial regression coefficients."""
    S = np.asarray(S, dtype=float)
    return coeffs[0] + coeffs[1]*S + coeffs[2]*S**2 + coeffs[3]*S**3


class ExercisePolicy:
    def __init__(self, thresholds, r, sigma, K, T_total, n_steps):
        """
        thresholds: array of length n_steps, thresholds[i] is the exercise threshold
                    at step i (exercise if S <= thresholds[i]).
        """
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.r = r
        self.sigma = sigma
        self.K = K
        self.T_total = T_total
        self.n_steps = n_steps

    def exercise(self, current_state, time):
        """
        current_state: array of spot prices (shape (n,))
        time: array of times (shape (n,)) or scalar
        Returns: boolean array of shape (n,)
        """
        current_state = np.asarray(current_state, dtype=float)
        time = np.asarray(time, dtype=float)
        if time.ndim == 0:
            time = np.full(current_state.shape, float(time))
        
        # Map time to step index
        dt = self.T_total / self.n_steps
        step_idx = np.clip((time / dt).astype(int), 0, self.n_steps - 1)
        
        # Get thresholds for each path's time step
        thresh = self.thresholds[step_idx]
        
        return current_state <= thresh


def fit_exercise_policy(training_paths, contract):
    """
    Fit exercise policy using LSMC with polynomial regression for continuation values.
    
    training_paths: array of shape (n_paths, n_time_points) of GBM paths
    contract: dict with keys like 'S0', 'K', 'r', 'sigma', 'T', 'n_steps', etc.
    """
    start_time = _time.time()
    
    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # training_paths shape: (n_paths, n_time_points)
    # Time points are at t_1, t_2, ..., t_n_steps (no t_0)
    paths = np.asarray(training_paths, dtype=float)
    n_paths = paths.shape[0]
    n_tp = paths.shape[1]
    
    if n_tp != n_steps:
        # Try to infer
        n_steps = n_tp
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Terminal payoff (put)
    S_T = paths[:, -1]
    payoff = np.maximum(K - S_T, 0.0)
    
    # Backward induction with LSMC
    # continuation_values[i] = estimated continuation value at step i
    # We store the regression coefficients at each step
    coeffs_list = [None] * n_steps
    
    # At the last step, continuation value = payoff
    # For step n_steps-1 (0-indexed), continuation = payoff
    
    # Work backwards from step n_steps-1 down to 0
    # At step i, we have S_i = paths[:, i]
    # The "target" for regression is the discounted expected continuation from next step
    # But we don't know the optimal policy yet, so we use the standard LSMC approach:
    # 1. First pass: assume exercise only at terminal, get "raw" continuation
    # 2. Or use the standard two-pass approach
    
    # Standard LSMC for Bermudan:
    # Pass 1: Compute continuation values assuming no early exercise (just discounted terminal payoff)
    # Pass 2: Use those to determine exercise decisions, then re-estimate
    
    # Actually, the standard approach is:
    # - Start from terminal
    # - At each step, regress the discounted continuation value from the next step on current state
    # - Exercise if intrinsic > continuation
    
    # Let's do a single backward pass with regression at each step
    # continuation[i] = max(intrinsic, E[discount * continuation[i+1] | S_i])
    # We estimate E[discount * continuation[i+1] | S_i] via regression
    
    # Initialize: at terminal step, "continuation" is just the payoff
    # For step n_steps-1 (last), continuation = payoff
    # For step n_steps-2, we regress discount * payoff on S at step n_steps-2
    # etc.
    
    # Store continuation values at each step
    cont_values = np.zeros((n_paths, n_steps))
    
    # At the last time point, continuation = payoff
    cont_values[:, -1] = payoff
    
    # Work backwards
    for i in range(n_steps - 2, -1, -1):
        S_i = paths[:, i]
        # Target: discount * cont_values[:, i+1]
        target = discount * cont_values[:, i + 1]
        
        # Fit regression of target on S_i
        coeffs = _fit_regression(target, S_i, degree=3)
        coeffs_list[i] = coeffs
        
        # Predicted continuation value
        cont_pred = _predict_regression(S_i, coeffs)
        
        # Intrinsic value
        intrinsic = np.maximum(K - S_i, 0.0)
        
        # Exercise if intrinsic > continuation
        exercise_mask = intrinsic > cont_pred
        
        # Continuation value at this step: if exercising, get intrinsic; else get predicted cont
        cont_values[:, i] = np.where(exercise_mask, intrinsic, cont_pred)
    
    # Now compute thresholds at each step
    # At step i, exercise if S <= threshold_i
    # threshold_i is where intrinsic = continuation
    # intrinsic = K - S (for S < K), continuation = _predict_regression(S, coeffs[i])
    # Solve K - S = c0 + c1*S + c2*S^2 + c3*S^3
    # => c3*S^3 + c2*S^2 + (c1+1)*S + (c0 - K) = 0
    
    thresholds = np.zeros(n_steps)
    for i in range(n_steps):
        if coeffs_list[i] is None:
            thresholds[i] = K  # default
            continue
        
        c = coeffs_list[i]
        # Solve c3*S^3 + c2*S^2 + (c1+1)*S + (c0 - K) = 0
        # for S in (0, K)
        a3 = c[3]
        a2 = c[2]
        a1 = c[1] + 1.0
        a0 = c[0] - K
        
        # Find roots
        roots = np.roots([a3, a2, a1, a0])
        # Filter real roots in (0, K)
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        valid_roots = real_roots[(real_roots > 0) & (real_roots < K)]
        
        if len(valid_roots) > 0:
            # Choose the largest valid root (threshold is where we start exercising)
            thresholds[i] = np.max(valid_roots)
        else:
            # No root in (0,K), check if we always exercise or never
            # At S=0: intrinsic = K, continuation = c0
            # At S=K: intrinsic = 0, continuation = c0 + c1*K + c2*K^2 + c3*K^3
            cont_at_K = _predict_regression(np.array([K]), c)[0]
            if cont_at_K <= 0:
                # Continuation is non-positive at K, so we always exercise for S < K
                thresholds[i] = K
            else:
                # We never exercise (or exercise only at very low S)
                thresholds[i] = 0.0
    
    # Check deadline
    elapsed = _time.time() - start_time
    if elapsed > _DEADLINE:
        # Return what we have
        policy = ExercisePolicy(thresholds, r, sigma, K, T, n_steps)
        return policy
    
    # Optional: refine with more polynomial terms or local search
    # For now, return the policy
    policy = ExercisePolicy(thresholds, r, sigma, K, T, n_steps)
    return policy

# EVOLVE-BLOCK-END
