"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def _lorenz96_step(x, F, dt=0.01):
    """Single RK4 step for Lorenz-96 with periodic boundary conditions."""
    p = len(x)
    def f(state):
        dx = np.empty_like(state)
        for i in range(p):
            dx[i] = (state[(i+1) % p] - state[(i-2) % p]) * state[(i-1) % p] - state[i] + F
        return dx
    k1 = f(x)
    k2 = f(x + 0.5*dt*k1)
    k3 = f(x + 0.5*dt*k2)
    k4 = f(x + dt*k3)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def _rk4_lorenz96(X, F, steps=5, dt=0.01):
    """Propagate ensemble forward using RK4 for given number of steps."""
    n, p = X.shape
    for _ in range(steps):
        k1 = np.array([_lorenz96_step(X[i], F, 0) for i in range(n)])
        # Actually, let's vectorize this properly
    # Vectorized RK4
    for _ in range(steps):
        def f_vec(state):
            # state is (n, p)
            dx = np.empty_like(state)
            for i in range(p):
                dx[:, i] = (state[:, (i+1) % p] - state[:, (i-2) % p]) * state[:, (i-1) % p] - state[:, i] + F
            return dx
        k1 = f_vec(X)
        k2 = f_vec(X + 0.5*dt*k1)
        k3 = f_vec(X + 0.5*dt*k2)
        k4 = f_vec(X + dt*k3)
        X = X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return X


def _compute_weights(X, y, obs_idx, R_var):
    """Compute importance weights for particle filter."""
    n = len(X)
    k = len(obs_idx)
    # Likelihood: N(y | X[:, obs], R_var * I)
    innov = y[None, :] - X[:, obs_idx]  # (n, k)
    # log likelihood = -0.5 * sum(innov^2 / R_var) - 0.5*k*log(2*pi*R_var)
    log_w = -0.5 * np.sum(innov**2 / R_var, axis=1) - 0.5 * k * np.log(2*np.pi*R_var)
    # Normalize to avoid underflow
    log_w_max = np.max(log_w)
    w = np.exp(log_w - log_w_max)
    w = w / np.sum(w)
    return w


def _systematic_resample(weights, n):
    """Systematic resampling to avoid degeneracy."""
    r = np.random.rand() / n
    u = (np.arange(n) + r) / n
    c = np.cumsum(weights)
    idx = np.searchsorted(c, u, side='right')
    idx = np.clip(idx, 0, n-1)
    return idx


def _lorenz96_jacobian(x, F):
    """Compute Jacobian of Lorenz-96 dynamics for covariance propagation."""
    p = len(x)
    J = np.zeros((p, p))
    for i in range(p):
        # dx_i/dx_j for Lorenz-96
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # d(dx_i)/dx_{i-1} = (x_{i+1} - x_{i-2})
        # d(dx_i)/dx_{i+1} = x_{i-1}
        # d(dx_i)/dx_{i-2} = -x_{i-1}
        # d(dx_i)/dx_i = -1
        J[i, (i-1) % p] = x[(i+1) % p] - x[(i-2) % p]
        J[i, (i+1) % p] = x[(i-1) % p]
        J[i, (i-2) % p] = -x[(i-1) % p]
        J[i, i] = -1.0
    return J


def _propagate_covariance(P, J, dt=0.01, steps=5):
    """Propagate covariance forward: P' = J*P*(J*)^T + P (for stability)"""
    for _ in range(steps):
        # Use first-order approximation: P' = J P J^T
        # Add small regularization to maintain positive definiteness
        P = J @ P @ J.T + 1e-10 * np.eye(P.shape[0])
    return P


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X.copy())

    # --- Particle Filter Approach ---
    # 1. Compute weights based on observation likelihood
    weights = _compute_weights(X, y, obs, R_var)
    
    # 2. Check effective sample size
    ess = 1.0 / np.sum(weights**2)
    
    # 3. Resample if ESS is too low
    if ess < n / 2.0:
        resample_idx = _systematic_resample(weights, n)
        X = X[resample_idx]
        weights = np.ones(n) / n
    
    # 4. Estimate ensemble mean and covariance
    m = X.mean(axis=0)
    A = X - m[None, :]
    P = (A.T @ A) / (n - 1)
    
    # 5. Localize covariance (Gaspari-Cohn)
    idx = np.arange(p)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, p - dist)
    r0, r1 = 4.0, 9.0
    loc = np.ones((p, p))
    mask = dist > r0
    d = dist[mask]
    if np.any(mask):
        x = d / (r1 - r0)
        loc[mask] = 0.25 * x**4 * (5.0 * x - 9.0) + 0.5 * x**2
    loc = np.clip(loc, 0.0, 1.0)
    P = P * loc
    
    # 6. Estimate forcing F from current state (rough estimate)
    # For Lorenz-96, F is constant but we don't know it. 
    # We can estimate it from the ensemble mean dynamics or use a default.
    # Since we don't have access to F, we'll use a heuristic:
    # The typical F values are 6, 8, 10. We'll estimate from the state.
    # Actually, we can't estimate F without knowing the dynamics parameters.
    # Let's use a simple approach: don't propagate, just do a weighted update.
    
    # 7. Weighted update: blend toward observation with localization
    # Use the covariance to compute a Kalman-like gain
    k = len(obs)
    Po = P[np.ix_(obs, obs)]
    Ho = P[obs, :].T  # (p, k)
    
    try:
        K_gain = Ho @ np.linalg.solve(Po + R_var * np.eye(k), np.eye(k))
    except np.linalg.LinAlgError:
        K_gain = Ho @ np.linalg.pinv(Po + R_var * np.eye(k))
    
    # Apply weighted update
    innov = y[None, :] - X[:, obs]  # (n, k)
    X_new = X + innov @ K_gain.T
    
    # 8. Adaptive inflation based on spread
    m2 = X_new.mean(axis=0)
    An = X_new - m2[None, :]
    if k > 1:
        obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
        obs_var_exp = np.diag(Po + R_var * np.eye(k))
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        alpha = np.clip(1.0 + 0.3 * (1.0 - ratio), 0.85, 1.4)
    else:
        alpha = 1.0
    X_new = m2[None, :] + alpha * An
    
    # 9. Small nudge on observed coordinates
    w_nudge = 0.08
    X_new[:, obs] += w_nudge * (y[None, :] - X_new[:, obs])
    
    out = _finite_clip(X_new)
    if (time.time() - _START) > _DEADLINE - 5.0:
        return out
    return out
# EVOLVE-BLOCK-END
