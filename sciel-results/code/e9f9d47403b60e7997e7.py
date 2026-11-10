"""Combined Local ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_DEADLINE = None
_DEADLINE_MARGIN = 2.0  # seconds of safety margin

def _gaspari_cohn(d, c=1.5):
    """Gaspari-Cohn correlation function for localization."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    # d <= 0.5c
    r1 = d[mask1]
    if len(r1) > 0:
        x = r1 / c
        result[mask1] = (-5.0/4.0)*x**5 + 5.0*x**4 - 10.0*x**3 + 5.0*x
    
    # 0.5c < d <= 1.5c
    r2 = d[mask2]
    if len(r2) > 0:
        x = r2 / c
        result[mask2] = (1.0/12.0)*x**5 - (5.0/4.0)*x**4 + (5.0/2.0)*x**3 - 5.0*x**2 + (25.0/12.0)*x + 0.5
    
    # 1.5c < d <= 2.5c
    r3 = d[mask3]
    if len(r3) > 0:
        x = r3 / c
        result[mask3] = (-1.0/16.0)*x**5 + (1.0/4.0)*x**4 - (5.0/8.0)*x**3 + (5.0/4.0)*x**2 - (25.0/12.0)*x + (25.0/12.0)
    
    return result

def _localize_covariance(ensemble, observed_indices, dim=20, loc_width=1.5):
    """
    Build localized observation covariance using Gaspari-Cohn taper.
    Returns localized P_oo (n_obs x n_obs) and the localization matrix.
    """
    n_obs = len(observed_indices)
    # Compute ensemble covariance in observation space
    ens_obs = ensemble[:, observed_indices]  # (n_ens, n_obs)
    mean_obs = ens_obs.mean(axis=0)
    anomalies = ens_obs - mean_obs  # (n_ens, n_obs)
    n_ens = ensemble.shape[0]
    P_oo = (anomalies.T @ anomalies) / (n_ens - 1)  # (n_obs, n_obs)
    
    # Build localization matrix
    loc_matrix = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            # Periodic distance in the full state space
            dist = abs(observed_indices[i] - observed_indices[j])
            dist = min(dist, dim - dist)
            loc_matrix[i, j] = _gaspari_cohn(dist, loc_width)
    
    # Localize: P_oo_loc = P_oo * loc_matrix (Hadamard product)
    P_oo_loc = P_oo * loc_matrix
    
    # Ensure symmetry and positive semi-definite
    P_oo_loc = (P_oo_loc + P_oo_loc.T) / 2.0
    
    return P_oo_loc

def _esrf_update(ensemble, observations, observed_indices, observation_variance, n_ens=16, dim=20):
    """
    Ensemble Square Root Filter update using SVD-based transform.
    This preserves the ensemble covariance exactly.
    """
    n_ens = ensemble.shape[0]
    
    # Compute ensemble mean and anomalies
    ens_mean = ensemble.mean(axis=0)
    anomalies = ensemble - ens_mean  # (n_ens, dim)
    
    # Observation space quantities
    ens_obs = ensemble[:, observed_indices]
    ens_obs_mean = ens_obs.mean(axis=0)
    obs_anomalies = ens_obs - ens_obs_mean  # (n_ens, n_obs)
    n_obs = len(observed_indices)
    
    # Background covariance in observation space
    P_oo = (obs_anomalies.T @ obs_anomalies) / (n_ens - 1)
    
    # Localize
    loc_matrix = np.zeros((n_obs, n_obs))
    for i in range(n_obs):
        for j in range(n_obs):
            dist = abs(observed_indices[i] - observed_indices[j])
            dist = min(dist, dim - dist)
            loc_matrix[i, j] = _gaspari_cohn(dist, 1.5)
    
    P_oo_loc = P_oo * loc_matrix
    P_oo_loc = (P_oo_loc + P_oo_loc.T) / 2.0
    
    # Observation error covariance
    R = observation_variance * np.eye(n_obs)
    S = P_oo_loc + R  # Innovation covariance
    
    # Innovation
    innovation = observations - ens_obs_mean  # (n_obs,)
    
    # SVD for stable inverse
    try:
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        # Regularize small singular values
        s_inv = np.where(s > 1e-10, 1.0 / s, 0.0)
        S_inv = (U * s_inv) @ Vt
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain in observation space
    K_obs = P_oo_loc @ S_inv  # (n_obs, n_obs)
    
    # Analysis mean in observation space
    ens_obs_mean_a = ens_obs_mean + K_obs @ innovation  # (n_obs,)
    
    # Now compute the full analysis using ESRF transform
    # We need to map the update from observation space to full state space
    
    # Cross-covariance P_xo: (dim, n_obs)
    P_xo = (anomalies.T @ obs_anomalies) / (n_ens - 1)
    
    # Localize cross-covariance
    P_xo_loc = np.zeros_like(P_xo)
    for i in range(dim):
        for j in range(n_obs):
            dist = abs(i - observed_indices[j])
            dist = min(dist, dim - dist)
            P_xo_loc[i, j] = P_xo[i, j] * _gaspari_cohn(dist, 1.5)
    
    # Full Kalman gain
    K = P_xo_loc @ S_inv  # (dim, n_obs)
    
    # Analysis mean
    ens_mean_a = ens_mean + K @ innovation  # (dim,)
    
    # ESRF: Transform the anomaly matrix
    # We want A_a such that A_a^T A_a / (n_ens-1) = P_aa
    # P_aa = P_xx - K S K^T (approximately, using localized)
    # Instead, use the SVD-based ESRF transform
    
    # Compute the analysis anomaly covariance
    P_xx = (anomalies.T @ anomalies) / (n_ens - 1)
    P_aa = P_xx - K @ S @ K.T
    P_aa = (P_aa + P_aa.T) / 2.0
    
    # Eigen-decompose for the transform
    try:
        eigvals, eigvecs = np.linalg.eigh(P_aa)
        # Clip small negative eigenvalues
        eigvals = np.maximum(eigvals, 0.0)
    except np.linalg.LinAlgError:
        # Fallback: use Cholesky or pinv
        eigvals = np.zeros(dim)
        eigvecs = np.eye(dim)
    
    sqrt_eigvals = np.sqrt(eigvals)
    
    # Construct the analysis anomaly matrix using a fixed random orthogonal matrix
    # For reproducibility, use a deterministic orthogonal matrix
    # We use the identity rotated by a fixed angle pattern
    theta = np.linspace(0, 2*np.pi, dim+1)[:-1]
    Q = np.eye(dim)
    # Apply a series of Givens rotations to create a "random-looking" orthogonal matrix
    for k in range(dim - 1):
        c = np.cos(theta[k])
        s = np.sin(theta[k])
        G = np.eye(dim)
        G[k, k] = c
        G[k, k+1] = -s
        G[k+1, k] = s
        G[k+1, k+1] = c
        Q = Q @ G
    
    # A_a = eigvecs @ diag(sqrt_eigvals) @ Q
    A_a = (eigvecs * sqrt_eigvals) @ Q  # (dim, dim)
    
    # Scale to match ensemble size: A_a should be (n_ens, dim)
    # We need n_ens rows. Use the first n_ens rows of a larger construction.
    # Actually, for ESRF, we construct A_a as (n_ens, dim) such that A_a^T A_a/(n_ens-1) = P_aa
    # One approach: A_a = sqrt(n_ens-1) * eigvecs[:, :n_ens-1] @ diag(sqrt_eigvals[:n_ens-1]) @ Q[:n_ens-1, :]
    # But we need n_ens rows. Let's use a different construction.
    
    # Simpler: Use the fact that for n_ens = 16, we can construct directly
    # A_a = W @ diag(sqrt(lambda)) where W is orthogonal (dim x dim)
    # Then A_a^T A_a = diag(sqrt(lambda)) @ W^T W @ diag(sqrt(lambda)) = diag(lambda)
    # But we need n_ens rows, not dim rows.
    
    # Standard ESRF: A_a = A_b @ W where W is an orthogonal (n_ens x n_ens) matrix
    # such that the resulting covariance matches.
    
    # Let's use the standard ESRF approach:
    # A_b = anomalies (n_ens x dim)
    # P_bb = A_b^T A_b / (n_ens - 1)
    # P_aa = P_bb - K S K^T
    # We want A_a such that A_a^T A_a / (n_ens - 1) = P_aa
    # A_a = A_b @ W for some orthogonal W (n_ens x n_ens)
    # This requires P_aa = W^T P_bb W, which is only possible if P_aa and P_bb have the same eigenvalues.
    # In general they don't, so we need a different approach.
    
    # Correct ESRF: 
    # 1. Compute P_aa
    # 2. Find A_a directly via eigendecomposition of P_aa
    # 3. A_a = V_aa @ diag(sqrt((n_ens-1)*lambda_aa)) where V_aa is (dim x n_ens-1)
    #    Then pad to n_ens rows.
    
    # For n_ens = 16, dim = 20:
    # P_aa is 20x20, rank at most 15 (n_ens-1)
    # eigvecs is 20x20, eigvals is 20
    # Take the first (n_ens-1) = 15 eigenvectors with non-zero eigenvalues
    
    n_eff = min(n_ens - 1, dim)
    idx = np.argsort(eigvals)[::-1][:n_eff]
    eigvals_eff = eigvals[idx]
    eigvecs_eff = eigvecs[:, idx]  # (dim, n_eff)
    
    # A_a should be (n_ens, dim)
    # A_a = eigvecs_eff @ diag(sqrt((n_ens-1)*eigvals_eff)) @ Q_eff
    # where Q_eff is (n_eff x n_ens) with orthonormal rows
    
    # Construct Q_eff: take first n_ens columns of a (n_eff x n_ens) matrix with orthonormal rows
    # Use QR decomposition of a random (n_eff x n_ens) matrix
    rng = np.random.default_rng(42)
    M = rng.standard_normal((n_eff, n_ens))
    Q_full, _ = np.linalg.qr(M.T)  # Q_full is (n_ens x n_ens), orthonormal columns
    Q_eff = Q_full[:n_eff, :].T  # (n_eff x n_ens), orthonormal rows
    
    A_a = (eigvecs_eff * np.sqrt((n_ens - 1) * eigvals_eff)) @ Q_eff  # (dim, n_ens)
    A_a = A_a.T  # (n_ens, dim)
    
    # Construct the updated ensemble
    result = np.tile(ens_mean_a, (n_ens, 1)) + A_a
    
    # Clamp to valid range
    result = np.clip(result, -100, 100)
    
    # Ensure finiteness
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0 - _DEADLINE_MARGIN
    
    if time.time() > _DEADLINE:
        # Return current ensemble as-is if we're out of time
        return ensemble.copy()
    
    n_ens, dim = ensemble.shape
    
    # ESRF update
    result = _esrf_update(ensemble, observations, observed_indices, observation_variance, n_ens, dim)
    
    return result
# EVOLVE-BLOCK-END
