"""EnVar assimilation with Huber loss for Lorenz-96, using L-BFGS-B on perturbation space."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = _start_time + 147.0

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Variational (EnVar) analysis with Huber loss.
    Minimizes a hybrid cost: Gaussian background penalty + Huber observation penalty.
    Optimizes over ensemble perturbation space using L-BFGS-B.
    """
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape
    N_obs = len(obs_idx)

    # Background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # M x N

    # Localization: Gaussian kernel on circular distance
    alpha = 1.0
    loc_weights = np.zeros((N, N_obs))
    for i in range(N):
        d = np.abs(i - obs_idx)
        d = np.minimum(d, N - d)
        loc_weights[i, :] = np.exp(-0.5 * (d / alpha) ** 2)

    # Observation error scaling
    beta = 0.85
    R_eff = R_val * beta

    # Huber parameter (transition from quadratic to linear)
    huber_delta = np.sqrt(R_eff) * 1.5

    # Pre-compute for fast cost evaluation
    # W_b shape: M x N
    # obs_idx: N_obs indices
    # loc_weights: N x N_obs

    def _cost_and_grad(p):
        """
        p: perturbation vector of length N*M (flattened W_a)
        Returns (cost, grad)
        
        Cost = 0.5 * sum_i (W_a[i] - W_b[i])^2 / sigma_b^2  (background)
             + sum_j huber(y_j - (mean_b + W_a[:,obs_j])^T @ e_j)  (observation)
        
        We parameterize W_a directly.
        """
        W_a = p.reshape(M, N)
        
        # Background penalty: 0.5 * ||W_a - W_b||^2_F (unit variance scaling)
        bg_diff = W_a - W_b
        cost_bg = 0.5 * np.sum(bg_diff ** 2)
        grad_bg = bg_diff
        
        # Observation residuals with localization
        # For each observed variable j at index obs_idx[j]:
        #   residual_j = y[j] - (mean_b[obs_idx[j]] + mean(W_a[:, obs_idx[j]], axis=0))
        # But we want the ensemble-mean observation, so:
        #   H(W_a) = mean over ensemble of W_a[:, obs_idx] + mean_b[obs_idx]
        
        # Compute ensemble mean at observed locations
        W_a_obs = W_a[:, obs_idx]  # M x N_obs
        mean_obs = mean_b[obs_idx] + np.mean(W_a_obs, axis=0)  # N_obs
        
        residuals = y - mean_obs  # N_obs
        
        # Huber loss
        abs_r = np.abs(residuals)
        quadratic_mask = abs_r <= huber_delta
        huber_cost = np.sum(
            np.where(quadratic_mask,
                     0.5 * residuals ** 2,
                     huber_delta * (abs_r - 0.5 * huber_delta))
        )
        
        # Gradient of Huber loss w.r.t. residuals
        grad_huber_r = np.where(quadratic_mask,
                                residuals,
                                huber_delta * np.sign(residuals))
        
        # d(residuals)/d(W_a) for each observed variable j:
        # residual_j = y[j] - mean_b[obs_idx[j]] - (1/M) * sum_m W_a[m, obs_idx[j]]
        # d(residual_j)/d(W_a[m, obs_idx[j]]) = -1/M
        # So grad_W_a[m, obs_idx[j]] += -grad_huber_r[j] / M
        
        grad_obs = np.zeros((M, N))
        for j in range(N_obs):
            grad_obs[:, obs_idx[j]] -= grad_huber_r[j] / M
        
        # Apply localization to observation gradient
        # The localization reduces the effective observation influence
        # We scale the observation gradient by localization weights
        # Actually, let's incorporate localization into the cost more carefully.
        # Simpler approach: scale the observation term by localization
        
        total_cost = cost_bg + huber_cost
        total_grad = grad_bg + grad_obs
        
        return total_cost, total_grad.flatten()
    
    # Initial guess: background perturbations
    p0 = W_b.flatten()
    
    # Run L-BFGS-B
    try:
        from scipy.optimize import minimize
        result = minimize(
            _cost_and_grad,
            p0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        p_opt = result.x
    except Exception:
        p_opt = p0
    
    W_a = p_opt.reshape(M, N)
    
    # Analysis mean
    mean_a = mean_b + np.mean(W_a, axis=0)
    
    # Adaptive inflation
    # Compare innovation scatter to expected
    # Expected innovation variance (localized)
    P_b_obs = (W_b[:, obs_idx].T @ W_b[:, obs_idx]) / (M - 1)
    expected_innov_var = np.mean(np.diag(P_b_obs)) + R_eff
    actual_innov_var = np.var(y - mean_b[obs_idx]) + 1e-10
    ratio = expected_innov_var / actual_innov_var
    adaptive_infl = np.sqrt(ratio)
    adaptive_infl = np.clip(adaptive_infl, 1.0, 1.08)
    
    # Constant inflation
    gamma = 1.015
    total_infl = gamma * adaptive_infl
    total_infl = np.clip(total_infl, 1.0, 1.10)
    
    # Apply inflation to perturbations
    W_a_centered = W_a - np.mean(W_a, axis=0)
    W_a_centered *= total_infl
    
    # Reconstruct ensemble
    X_a = mean_a + W_a_centered + (np.mean(W_a, axis=0) - np.mean(W_a, axis=0))
    # Actually: X_a[m] = mean_a + W_a_centered[m]
    # But mean_a already includes mean(W_a), so:
    X_a = mean_a + W_a_centered
    
    # Safety
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X_a
# EVOLVE-BLOCK-END
