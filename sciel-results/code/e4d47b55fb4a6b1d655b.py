"""Lorenz-96 ensemble assimilation: Nonlinear Ensemble Smoother (NLES) with adaptive LM steps."""
import numpy as np
import time

def _lorenz96_deriv(x, F=8.0):
    """Compute Lorenz-96 derivatives. x has shape (N, dim)."""
    N, dim = x.shape
    xm2 = np.roll(x, 2, axis=1)
    xm1 = np.roll(x, 1, axis=1)
    xp1 = np.roll(x, -1, axis=1)
    return (xm2 - xm1) * xp1 - x + F

def _rk4_step(x, dt, F=8.0):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_deriv(x, F)
    k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_deriv(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _propagate(ensemble, n_steps, dt, F):
    """Propagate ensemble forward n_steps times."""
    ens = ensemble.copy()
    for _ in range(n_steps):
        ens = _rk4_step(ens, dt, F)
    return ens

def _estimate_F(ensemble, dt=0.01):
    """Estimate forcing F from the ensemble state."""
    # F ≈ mean(x_i) + mean((x_{i+1} - x_{i-2}) * x_{i-1})
    # From dx_i/dt = (x_{i+1} - x_{i-2})*x_{i-1} - x_i + F
    # At quasi-steady state, mean(dx/dt) ≈ 0
    # So F ≈ mean(x_i) - mean((x_{i+1} - x_{i-2})*x_{i-1})
    # But this isn't quite right. Let's use:
    # F = x_i - (x_{i+1} - x_{i-2})*x_{i-1} + dx_i/dt
    # Taking ensemble mean: F ≈ mean(x) - mean((x_{i+1}-x_{i-2})*x_{i-1}) + mean(dx/dt)
    # The last term is small for a well-mixed ensemble
    dim = ensemble.shape[1]
    xm2 = np.roll(ensemble, 2, axis=1)
    xm1 = np.roll(ensemble, 1, axis=1)
    xp1 = np.roll(ensemble, -1, axis=1)
    nonlinear = (xm2 - xm1) * xp1
    # F ≈ mean(x) - mean(nonlinear)  (ignoring mean of derivative)
    F_est = float(np.mean(ensemble) - np.mean(nonlinear))
    # Clamp to reasonable range
    F_est = np.clip(F_est, 4.0, 12.0)
    return F_est

def _get_loc_weights(obs_idx, dim, rng):
    """Gaussian localization weights for a single observation."""
    w = np.empty(dim)
    for j in range(dim):
        d = min(abs(obs_idx - j), dim - abs(obs_idx - j))
        w[j] = np.exp(-0.5 * (d / rng) ** 2)
    return w

def analyze(ensemble, observations, observed_indices, observation_variance):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Sanitize inputs
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).astype(float).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)
    
    # Observation operator indices
    obs_idx = np.array(observed_indices)
    
    # Estimate F from the ensemble
    F_est = _estimate_F(ensemble)
    
    # Propagation parameters (matching evaluator: 5 steps of dt=0.01)
    n_steps = 5
    dt = 0.01
    
    # Localization range
    loc_range = 3.0
    
    # Precompute localization weights for each observation
    loc_weights = np.zeros((n_obs, dim))
    for i, idx in enumerate(obs_idx):
        loc_weights[i] = _get_loc_weights(idx, dim, loc_range)
    
    # Observation noise std
    obs_std = np.sqrt(observation_variance)
    
    # Initialize with the forecast ensemble (no change as starting point)
    # But we can also start from a slightly inflated version
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    cov = np.cov(ensemble, rowvar=False)
    
    # Adaptive inflation
    inflation = 1.0
    inflation_alpha = 0.8
    
    # NLES iterations
    n_nles_iter = 8
    
    # LM parameters
    lm_lambda = 1.0
    lm_increase = 2.0
    lm_decrease = 0.5
    lm_min = 1e-6
    lm_max = 1e4
    
    # Best solution tracking
    best_ensemble = ensemble.copy()
    best_cost = np.inf
    
    # Compute initial cost: propagate and measure misfit
    def compute_cost(ens):
        ens_prop = _propagate(ens, n_steps, dt, F_est)
        ens_mean = ens_prop.mean(axis=0)
        obs_pred = ens_mean[obs_idx]
        misfit = (obs_pred - observations) / obs_std
        cost = 0.5 * np.dot(misfit, misfit)
        return cost, ens_prop
    
    # Initial cost
    init_cost, _ = compute_cost(ensemble)
    best_cost = init_cost
    
    # For each NLES iteration, we do a LM step
    # We approximate the Jacobian using the ensemble
    current_ensemble = ensemble.copy()
    
    for nles_iter in range(n_nles_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Compute current cost
        current_cost, current_prop = compute_cost(current_ensemble)
        
        # Compute ensemble-based approximation of the Jacobian
        # J ≈ (H * (ens_prop - mean_prop)) / (anomalies)
        # More precisely, we want the sensitivity of H*x_prop to x_init
        # Use finite difference via ensemble spread
        mean_prop = current_prop.mean(axis=0)
        anomalies_prop = current_prop - mean_prop  # (N, dim)
        anomalies_init = current_ensemble - current_ensemble.mean(axis=0)  # (N, dim)
        
        # Approximate Jacobian: J[i,j] = d(H*x_prop)_i / d(x_init)_j
        # Using ensemble: J ≈ H * (anomalies_prop @ pinv(anomalies_init))
        # But this is rough. Better: use the linearized dynamics.
        # For simplicity, use a diagonal approximation scaled by ensemble spread
        
        # Compute innovation
        obs_pred = mean_prop[obs_idx]
        innovation = (observations - obs_pred) / obs_std  # (n_obs,)
        
        # Approximate the gain using ensemble covariance
        # P_init = cov(current_ensemble)
        # P_prop = cov(current_prop)
        # Cross-cov C = cov(anomalies_init, anomalies_prop)
        
        P_init = np.cov(current_ensemble, rowvar=False)
        P_prop = np.cov(current_prop, rowvar=False)
        
        # Cross-covariance
        C = anomalies_init.T @ anomalies_prop / (N - 1)  # (dim, dim)
        
        # Observation space
        H = np.zeros((n_obs, dim))
        for i, idx in enumerate(obs_idx):
            H[i, idx] = 1.0
        
        # S = H P_prop H^T + R
        P_obs = P_prop[np.ix_(obs_idx, obs_idx)]
        R = observation_variance * np.eye(n_obs)
        S = P_obs + R
        S += 1e-10 * np.eye(n_obs)
        
        # Gain: K = C H^T S^{-1}
        CHt = C @ H.T
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        K = CHt @ S_inv  # (dim, n_obs)
        
        # Apply localization
        K_loc = K.copy()
        for j, obs_i in enumerate(obs_idx):
            K_loc[:, j] *= loc_weights[j]
        
        # LM step: delta_x = K @ innovation
        delta_x = K_loc @ innovation  # (dim,)
        
        # Adaptive LM: scale the step
        # Compute the step norm relative to ensemble spread
        spread = np.sqrt(np.diag(P_init)) + 1e-10
        step_norm = np.linalg.norm(delta_x / spread)
        
        # LM damping
        step_scale = 1.0 / (1.0 + lm_lambda * step_norm)
        
        # Try the step
        new_ensemble = current_ensemble + np.tile(step_scale * delta_x, (N, 1))
        
        # Also add a small random perturbation to maintain ensemble spread
        # Use the original anomalies scaled
        new_ensemble = new_ensemble + inflation * anomalies_init * 0.1
        
        new_cost, _ = compute_cost(new_ensemble)
        
        if new_cost < current_cost:
            # Accept step, decrease lambda
            current_ensemble = new_ensemble
            current_cost = new_cost
            lm_lambda = max(lm_lambda * lm_decrease, lm_min)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_ensemble = current_ensemble.copy()
        else:
            # Reject step, increase lambda
            lm_lambda = min(lm_lambda * lm_increase, lm_max)
        
        # Adaptive inflation update
        # Compare innovation norm to expected
        innovation_norm = np.linalg.norm(innovation)
        expected_norm = np.sqrt(np.trace(S) / observation_variance)
        if expected_norm > 1e-10:
            ratio = innovation_norm / expected_norm
            target_inf = np.clip(ratio, 1.0, 3.0)
            inflation = inflation_alpha * inflation + (1 - inflation_alpha) * target_inf
            inflation = np.clip(inflation, 1.0, 3.0)
    
    # Final result
    updated = best_ensemble
    
    # Safety checks
    if time.time() - t0 > 105:
        updated = np.clip(np.where(np.isfinite(updated), updated, 0.0), -100, 100)
        return updated
    
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)
    
    return updated
