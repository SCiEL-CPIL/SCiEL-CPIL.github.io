import numpy as np
import time

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    xm2 = np.roll(x, 2)
    xm1 = np.roll(x, 1)
    xp1 = np.roll(x, -1)
    return (xp1 - xm2) * xm1 - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _get_localization_matrix(n, length):
    """Generate Gaussian localization matrix."""
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            d = min(d, n - d)
            dist[i, j] = d
    L = np.exp(-0.5 * (dist / length) ** 2)
    return L

def analyze(ensemble, observations, observed_indices, observation_variance):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n_members, n_vars = ensemble.shape
    result = ensemble.copy()

    # 1. Localization Matrix
    _localization_length = 2.0
    L = _get_localization_matrix(n_vars, _localization_length)

    # 2. Stochastic Ensemble Kalman Filter (SE-Kalman)
    # We use the "perturbed observation" approach with random rotation.
    # This ensures the ensemble mean matches the optimal Kalman update.

    m = np.mean(result, axis=0)
    anomalies = result - m

    # Covariance C = (1/(N-1)) * anomalies.T @ anomalies
    C = (anomalies.T @ anomalies) / (n_members - 1)

    # Apply Localization to C
    C_loc = C * L

    # Compute H C_loc H^T (Observation Covariance)
    C_oo = C_loc[np.ix_(observed_indices, observed_indices)]

    # R (Observation Noise Covariance)
    R = observation_variance * np.eye(len(observed_indices))

    # Innovation
    y = observations
    m_obs = m[observed_indices]
    d = y - m_obs

    # S = C_oo + R
    S = C_oo + R

    # Compute Cholesky of S
    try:
        L_s = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S_jittered = S + 1e-6 * np.eye(len(observed_indices))
        L_s = np.linalg.cholesky(S_jittered)

    # Compute Kalman Gain K = C_loc H^T S^-1
    C_h = C_loc[:, observed_indices] # 20 x 10
    W = np.linalg.inv(L_s)
    K = C_h @ W.T @ W # 20 x 10

    # Generate perturbations for stochastic update
    # We need eps_i such that mean(eps_i) = 0 and Cov(eps_i) = C_oo
    # We use the "random rotation" method:
    # Let Z be a (N x M) matrix of i.i.d N(0,1)
    # Let eps = Z @ A where A A^T = C_oo
    # To ensure mean(eps) = 0, we center Z first.
    
    rng = np.random.default_rng(int(time.time() * 1000) % (2**32))
    Z = rng.standard_normal((n_members, len(observed_indices)))
    Z = Z - np.mean(Z, axis=0, keepdims=True)

    # Get square root of C_oo
    try:
        L_oo = np.linalg.cholesky(C_oo)
        eps = Z @ L_oo.T
    except np.linalg.LinAlgError:
        U, s, Vt = np.linalg.svd(C_oo, full_matrices=False)
        C_oo_sqrt = U * np.sqrt(s)
        eps = Z @ C_oo_sqrt.T

    # Check time
    if time.time() > deadline - 2.0:
        return result

    # Update each member
    x_obs = result[:, observed_indices] # (N, M)
    y_pert = y[None, :] + eps # (N, M)

    # Innovation for each member: d_i = y_pert_i - x_obs_i
    d_all = y_pert - x_obs # (N, M)

    # Correction: K (20 x 10) @ d_all.T (10 x N) -> (20 x N)
    corr_all = K @ d_all.T

    # Update
    result = result + corr_all

    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)

    return result
