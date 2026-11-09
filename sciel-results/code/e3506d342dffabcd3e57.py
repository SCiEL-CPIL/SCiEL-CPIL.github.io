"""
Lorenz-96 ensemble assimilation with localised ensemble Kalman filter.
Structural change from naive nudging to a proper data-assimilation update
using ensemble covariance, localisation, and adaptive inflation.
"""
import numpy as np
import time

# Precompute the Lorenz-96 Jacobian structure for tangent linear / adjoint
# Actually, we'll use the ensemble-based approach directly.

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i + 1) % n] - x[(i - 2) % n]) * x[(i - 1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# We don't know F inside analyze, but we can infer it from the ensemble mean
# or just use a reasonable default. Actually, F is not passed to analyze.
# The evaluator propagates between updates, so we only need to do the analysis step.
# We don't need to propagate inside analyze.

def _localization_kernel(dist, r):
    """Gaussian localization kernel."""
    if dist <= r:
        return np.exp(-0.5 * (dist / r) ** 2)
    return 0.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman filter analysis step for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed coordinate indices
    - observation_variance: scalar observation noise variance
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    ens = np.array(ensemble, dtype=np.float64)
    n_members, n_vars = ens.shape
    n_obs = len(observations)
    obs_var = observation_variance
    obs_std = np.sqrt(obs_var)
    
    # Estimate F from the ensemble: average of x_i + (x_{i+1}-x_{i-2})*x_{i-1}
    # Actually, F is a constant parameter. We can estimate it from the ensemble mean.
    x_mean = np.mean(ens, axis=0)
    F_estimates = []
    for i in range(n_vars):
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # At quasi-steady state, dx_i/dt ≈ 0, so F ≈ x_i - (x_{i+1} - x_{i-2})*x_{i-1}
        F_estimates.append(x_mean[i] - (x_mean[(i + 1) % n_vars] - x_mean[(i - 2) % n_vars]) * x_mean[(i - 1) % n_vars])
    F = float(np.mean(F_estimates))
    
    # Ensemble mean and perturbations
    x_bar = np.mean(ens, axis=0)
    Xp = ens - x_bar  # (n_members, n_vars)
    
    # Ensemble covariance: P = (1/(n-1)) * Xp^T Xp
    P = (Xp.T @ Xp) / (n_members - 1.0)
    
    # Observation operator H: selects observed_indices
    # H is (n_obs, n_vars) with 1s at observed positions
    
    # Innovation: y - H x_bar
    y_innov = observations - x_bar[observed_indices]
    
    # Observation covariance R = obs_var * I
    R = obs_var * np.eye(n_obs)
    
    # H P H^T
    HP = P[np.ix_(observed_indices, observed_indices)]
    S = HP + R  # (n_obs, n_obs)
    
    # Kalman gain: K = P H^T S^{-1}
    # K is (n_vars, n_obs)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = (P[:, observed_indices] @ S_inv.T)  # (n_vars, n_obs)
    
    # Apply localization to the gain
    # Use a Gaussian localization with range ~ 3-4 grid points
    loc_range = 3.5
    for i in range(n_vars):
        for j_idx, obs_idx in enumerate(observed_indices):
            dist = abs(i - obs_idx)
            # Handle periodic boundary
            dist = min(dist, n_vars - dist)
            loc_val = _localization_kernel(dist, loc_range)
            K[i, j_idx] *= loc_val
    
    # Update ensemble mean
    x_bar_new = x_bar + K @ y_innov
    
    # Update ensemble perturbations using the ensemble Kalman filter formula
    # Xp_new = Xp - K H Xp  (for each member)
    # H Xp for each member: Xp[:, observed_indices]
    HXp = Xp[:, observed_indices]  # (n_members, n_obs)
    # K @ HXp: (n_vars, n_obs) @ (n_members, n_obs) -> need to transpose
    # Xp_new = Xp - (K @ HXp.T).T
    Xp_new = Xp - (K @ HXp.T).T
    
    # Adaptive inflation to prevent filter collapse
    # Compute innovation variance and compare to expected
    innovation_var = np.var(y_innov)
    expected_innovation_var = np.trace(HP + R) / n_obs
    # Inflate if innovation variance is too low
    inflation_factor = 1.0
    if expected_innovation_var > 0:
        ratio = innovation_var / expected_innovation_var
        if ratio < 0.8:
            inflation_factor = 1.0 + 0.1 * (0.8 - ratio)
    
    # Apply inflation to perturbations
    Xp_new *= inflation_factor
    
    # Reconstruct ensemble
    ens_new = x_bar_new[None, :] + Xp_new
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return np.clip(ens_new, -100, 100)
    
    # Ensure finite values
    ens_new = np.where(np.isfinite(ens_new), ens_new, x_bar)
    
    # Clip to valid range
    ens_new = np.clip(ens_new, -100, 100)
    
    return ens_new
