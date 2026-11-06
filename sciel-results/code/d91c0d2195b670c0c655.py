"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE_SECONDS = 145.0
_START_TIME = [time.time()]

def _elapsed():
    """Return elapsed time since first call."""
    return time.time() - _START_TIME[0]

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _gaussian_localization(dist, L):
    """Gaussian localization function."""
    return np.exp(-0.5 * (dist / L) ** 2)

def _compute_localization_matrix(D, observed_indices, L):
    """Compute localization matrix for cross-covariance.
    
    Returns (D, M) matrix of localization weights.
    """
    M = len(observed_indices)
    loc = np.ones((D, M))
    d_idx = np.arange(D)
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc[:, m_idx] = _gaussian_localization(dist, L)
    return loc

def _ensemble_square_root_filter(ensemble, obs_vec, observed_indices, observation_variance,
                                 localization_length=2.0, min_inflation=1.0, max_inflation=1.5):
    """
    Ensemble Square-Root Filter update with Gaussian localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : (N, D) array - ensemble of states
    obs_vec : (M,) array - single observation vector
    observed_indices : array of indices that are observed
    observation_variance : float - variance of observation noise
    localization_length : float - length scale for Gaussian localization
    min_inflation : float - minimum inflation factor
    max_inflation : float - maximum inflation factor
    
    Returns:
    --------
    updated_ensemble : (N, D) array
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Ensure obs_vec is 1D
    obs_vec = np.asarray(obs_vec).flatten()
    
    # Compute ensemble mean
    ens_mean = np.mean(ensemble, axis=0)
    
    # Centered ensemble (anomalies)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observation operator: H selects observed components
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # Compute innovation: y - H(x)
    # y is obs_vec, H(x) is ensemble[:, observed_indices]
    # We update each member individually or collectively?
    # EnSRF usually updates the anomaly matrix directly.
    
    # Innovation vector for the ensemble mean
    mean_obs = ens_mean[observed_indices]
    innov_mean = obs_vec - mean_obs  # (M,)
    
    # Innovation anomalies for each member
    # y_n - H(x_n) = (y - mean_obs) - (H(x_n) - H(mean_x)) = innov_mean - obs_anomalies
    innov_anoms = innov_mean[None, :] - obs_anomalies  # (N, M)
    
    # Compute S = H P H^T + R
    # P = A A^T / (N-1)
    # S = (H A) (H A)^T / (N-1) + R
    # Let Y = obs_anomalies (N x M)
    # Y Y^T is M x M
    Y = obs_anomalies
    S = (Y.T @ Y) / (N - 1) + observation_variance * np.eye(M)
    
    # Cholesky decomposition of S
    # S = L L^T
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Add jitter if Cholesky fails
        S += 1e-6 * np.eye(M)
        L = np.linalg.cholesky(S)
    
    # Solve L^{-1} (innov_anoms)^T
    # We want to update anomalies: A_new = A - A H^T S^{-1} (y - Hx)
    # Using square root form:
    # Let Z = L^{-1} (y - Hx)^T (M x N)
    # Then A_new = A - (A H^T) Z^T ? No.
    
    # Standard EnSRF update:
    # 1. Compute innovation matrix Y_innov = (y - Hx_n) for each n? No, that's N x M.
    # 2. Let's use the formulation from Anderson (2003) or similar.
    # 
    # Update: x_n^a = x_n^b + K (y - H x_n^b)
    # K = P H^T S^{-1}
    # P H^T = (A A^T / (N-1)) H^T = A (A^T H^T / (N-1)) = A (H A)^T / (N-1)
    # Let C = H A (M x N).
    # K = A C^T / (N-1) S^{-1}
    # x_n^a = x_n^b + A C^T / (N-1) S^{-1} (y - H x_n^b)
    #
    # Let's compute S_inv (y - H x_n^b) for all n.
    # Let Innov = (y[None, :] - H(ensemble)) (N x M)
    # We need S^{-1} Innov^T (M x N)
    
    Innov = obs_vec[None, :] - ensemble[:, observed_indices]  # (N, M)
    
    # Solve S X = Innov^T
    # X = S^{-1} Innov^T (M x N)
    # Using Cholesky: L L^T X = Innov^T
    # L (L^T X) = Innov^T
    # L^T X = L^{-1} Innov^T
    # X = L^{-T} (L^{-1} Innov^T)
    
    # Solve L Y = Innov^T (M x N)
    Y_sol = np.linalg.solve(L, Innov.T)  # (M, N)
    # Solve L^T X = Y_sol (M x N)
    X_sol = np.linalg.solve(L.T, Y_sol)  # (M, N)
    
    # X_sol is S^{-1} Innov^T
    # We need K_n = A C^T / (N-1) S^{-1} Innov_n
    # C = H A (M x N)
    C = obs_anomalies.T  # (M, N)
    
    # A is (N x D)
    # C^T is (N x M)
    # A C^T is (N x M)
    ACt = anomalies @ C  # (N, M)
    
    # Update term: ACt / (N-1) @ X_sol.T ?
    # X_sol is (M x N). X_sol.T is (N x M).
    # Wait, X_sol[k, n] corresponds to S^{-1} Innov_n[k].
    # We want sum_k ACt[n, k] * (1/(N-1)) * X_sol[k, n] ?
    # No, K = A C^T / (N-1) S^{-1}
    # Update_n = K_n (y - H x_n)
    # K_n = (A C^T / (N-1))[:, n] S^{-1}
    # This is getting confusing with indices.
    
    # Let's use the matrix form:
    # Delta_A = A - A H^T S^{-1} (y - H A) ? No.
    # The update for the anomalies is:
    # A_new = A (I - H^T S^{-1} H A^T / (N-1) ) ? No.
    
    # Let's stick to the explicit gain calculation but vectorized.
    # Gain G = A C^T / (N-1) S^{-1}  (D x M)
    # C^T is N x M. A is N x D.
    # A C^T is N x M.
    # We need G = (A C^T / (N-1)) @ S^{-1}
    # But we have X_sol = S^{-1} Innov^T (M x N).
    # So S^{-1} = X_sol @ (Innov^T)^{-1} ? No, Innov is not invertible.
    
    # Alternative:
    # Update_n = (A C^T / (N-1) S^{-1}) Innov_n
    # = (A C^T / (N-1)) @ (S^{-1} Innov_n)
    # S^{-1} Innov_n is the n-th column of X_sol.
    # So Update_n = (A C^T / (N-1))[:, n] ? No.
    
    # Let's compute the matrix M_update = A C^T / (N-1) (N x M)
    M_update = (anomalies @ C) / (N - 1)  # (N x M)
    
    # Now Update_n = M_update[:, n] @ X_sol[:, n] ? No.
    # Update_n is a vector of size D.
    # M_update is N x M. This doesn't look like D x M.
    
    # Let's re-derive:
    # P H^T = A (A^T H^T) / (N-1) = A (H A)^T / (N-1)
    # H A is M x N. (H A)^T is N x M.
    # A is N x D.
    # P H^T is D x M.
    # Let's compute PHt = A @ (C.T) / (N-1) ?
    # C = H A (M x N). C.T = A.T H.T (N x M).
    # A @ C.T is N x M. This is not D x M.
    
    # Error in derivation:
    # P = A A^T / (N-1). A is N x D. A A^T is D x D.
    # H P H^T = H (A A^T / (N-1)) H^T = (H A) (H A)^T / (N-1).
    # H A is M x N. (H A) (H A)^T is M x M. Correct.
    # P H^T = A A^T / (N-1) H^T = A (A^T H^T) / (N-1).
    # A^T is D x N. H^T is D x M.
    # A^T H^T is D x M.
    # So P H^T = A @ (A^T H^T) / (N-1).
    # A is N x D. (A^T H^T) is D x M.
    # Result is N x M. This is still not D x M.
    
    # Ah, P H^T must be D x M.
    # P is D x D. H^T is D x M.
    # P H^T = (A A^T / (N-1)) H^T = A (A^T H^T) / (N-1).
    # A is N x D. A^T is D x N. H^T is D x M.
    # A^T H^T is D x M.
    # A @ (A^T H^T) is (N x D) @ (D x M) = N x M.
    # This implies P H^T is N x M? No, P is D x D.
    # The formula P = A A^T / (N-1) assumes A is D x N (columns are anomalies).
    # In my code, `anomalies` is N x D (rows are anomalies).
    # So P = anomalies.T @ anomalies / (N-1). (D x D).
    # H P H^T = H (anomalies.T @ anomalies / (N-1)) H^T
    # = (H anomalies.T) (anomalies H^T) / (N-1)
    # H anomalies.T is M x N.
    # anomalies H^T is N x M.
    # (M x N) @ (N x M) = M x M. Correct.
    #
    # P H^T = anomalies.T @ anomalies / (N-1) @ H^T
    # = anomalies.T @ (anomalies @ H^T) / (N-1)
    # anomalies @ H^T is N x M.
    # anomalies.T is D x N.
    # (D x N) @ (N x M) = D x M. Correct.
    
    # So, let's compute:
    # Anom_Ht = anomalies @ np.eye(D)[:, observed_indices] ?
    # H is M x D. H selects observed indices.
    # H @ anomalies.T ? No.
    # Let's just use the observed anomalies directly.
    # obs_anomalies = anomalies[:, observed_indices] (N x M).
    # This is H @ anomalies.T ? No.
    # H is M x D. H @ anomalies.T (M x D @ D x N) = M x N.
    # So obs_anomalies.T is H @ anomalies.T.
    # Therefore, anomalies @ H^T = obs_anomalies. (N x M).
    
    # So P H^T = anomalies.T @ obs_anomalies / (N-1). (D x M).
    PHt = anomalies.T @ obs_anomalies / (N - 1)  # (D, M)
    
    # Now, Gain K = P H^T S^{-1} (D x M @ M x M = D x M).
    # We have X_sol = S^{-1} Innov^T (M x N).
    # We want to compute Update = K @ Innov^T ? No.
    # Update_n = K_n (y - H x_n) = K @ Innov_n (D x 1).
    # Update = K @ Innov (D x N).
    # K = P H^T S^{-1}.
    # K @ Innov = P H^T S^{-1} Innov.
    # We have X_sol = S^{-1} Innov^T.
    # So S^{-1} Innov = X_sol.T (N x M)? No.
    # X_sol is M x N. X_sol[k, n] = (S^{-1} Innov^T)[k, n] = (S^{-1} Innov_n)[k].
    # So S^{-1} Innov is N x M, with (S^{-1} Innov)_{n, k} = X_sol[k, n].
    # So S^{-1} Innov = X_sol.T.
    
    # Update = P H^T @ (X_sol.T) ?
    # P H^T is D x M. X_sol.T is N x M.
    # This doesn't multiply.
    
    # Let's check dimensions again.
    # Update (N x D).
    # Update_n = P H^T S^{-1} Innov_n.
    # P H^T is D x M. S^{-1} is M x M. Innov_n is M x 1.
    # Result D x 1.
    
    # We want to compute the matrix Update (N x D) such that Update[n, :] = P H^T S^{-1} Innov[n, :].
    # Let M1 = P H^T (D x M).
    # Let M2 = S^{-1} Innov (M x N).
    # Then Update^T = M1 @ M2 (D x N).
    # So Update = (M1 @ M2).T (N x D).
    
    # We have X_sol = S^{-1} Innov^T (M x N).
    # So M2 = S^{-1} Innov = X_sol.T (N x M)? No.
    # X_sol[k, n] = (S^{-1} Innov^T)[k, n] = (S^{-1} Innov_n)[k].
    # So S^{-1} Innov_n is the k-th element of X_sol[:, n].
    # So S^{-1} Innov is a matrix where the n-th column is X_sol[:, n].
    # So S^{-1} Innov = X_sol (M x N).
    
    # Wait.
    # Innov is N x M.
    # Innov^T is M x N.
    # S^{-1} is M x M.
    # S^{-1} Innov^T is M x N.
    # Let Y = S^{-1} Innov^T.
    # Y[k, n] = sum_j S^{-1}[k, j] Innov^T[j, n] = sum_j S^{-1}[k, j] Innov[n, j].
    # This is the n-th column of S^{-1} Innov^T.
    # We want S^{-1} Innov (M x N).
    # (S^{-1} Innov)[k, n] = sum_j S^{-1}[k, j] Innov[n, j].
    # This is exactly Y[k, n].
    # So S^{-1} Innov = X_sol (M x N).
    
    # So M2 = X_sol (M x N).
    # M1 = P H^T (D x M).
    # Update^T = M1 @ M2 (D x N).
    # Update = (M1 @ M2).T (N x D).
    
    Update_T = PHt @ X_sol  # (D x N)
    Update = Update_T.T  # (N x D)
    
    updated_ensemble = ensemble + Update
    
    # Apply localization to the cross-covariance PHt before computing gain?
    # Yes, standard practice.
    # Let's re-calculate with localization.
    loc_weights = _compute_localization_matrix(D, observed_indices, localization_length)
    PHt_loc = PHt * loc_weights  # (D, M)
    
    # Recalculate Update with localized PHt
    Update_T_loc = PHt_loc @ X_sol  # (D x N)
    Update_loc = Update_T_loc.T  # (N x D)
    
    updated_ensemble_loc = ensemble + Update_loc
    
    # Adaptive Inflation
    # Check spread at observed locations
    ens_spread_obs = np.std(updated_ensemble_loc[:, observed_indices], axis=0, ddof=1)
    obs_noise_std = np.sqrt(observation_variance)
    
    # If spread is too small, inflate
    # Target spread should be comparable to obs noise or larger
    # Simple heuristic: if mean spread < 0.5 * obs_noise_std, inflate
    mean_spread = np.mean(ens_spread_obs)
    
    if mean_spread < 0.5 * obs_noise_std:
        inflation = 1.1
    elif mean_spread < obs_noise_std:
        inflation = 1.05
    else:
        inflation = 1.0
    
    # Cap inflation
    inflation = np.clip(inflation, min_inflation, max_inflation)
    
    # Apply inflation about the mean
    mean_updated = np.mean(updated_ensemble_loc, axis=0)
    updated_ensemble_loc = mean_updated + inflation * (updated_ensemble_loc - mean_updated)
    
    # Ensure finite values and clip to valid range
    updated_ensemble_loc = np.nan_to_num(updated_ensemble_loc, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble_loc = np.clip(updated_ensemble_loc, -100.0, 100.0)
    
    return updated_ensemble_loc


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr
    
    # Check deadline
    if _elapsed() > _DEADLINE_SECONDS - 5.0:
        # Return ensemble mean as fallback if we're out of time
        ens_mean = np.mean(ensemble, axis=0, keepdims=True)
        return np.tile(ens_mean, (N, 1))
    
    # EnSRF with localization and inflation
    result = _ensemble_square_root_filter(
        ensemble, obs_vec, observed_indices, observation_variance,
        localization_length=2.0,
        min_inflation=1.0,
        max_inflation=1.5
    )
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
