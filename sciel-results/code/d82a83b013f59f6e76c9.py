"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def _lorenz96_step(x, F, dt=0.01, n=5):
    """Propagate a single state vector through n RK4 steps of dt."""
    nvars = len(x)
    def f(x):
        # Lorenz-96: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # periodic indices
        x_shifted1 = np.roll(x, -1)  # x_{i+1}
        x_shifted2 = np.roll(x, 2)   # x_{i-2}
        x_shifted3 = np.roll(x, 1)   # x_{i-1}
        return (x_shifted1 - x_shifted2) * x_shifted3 - x + F
    
    for _ in range(n):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def _lorenz96_step_ensemble(ensemble, F, dt=0.01, n=5):
    """Propagate all ensemble members through n RK4 steps of dt."""
    # ensemble shape: (n_members, n_vars)
    n_members, n_vars = ensemble.shape
    for _ in range(n):
        k1 = _lorenz96_rhs(ensemble, F)
        k2 = _lorenz96_rhs(ensemble + 0.5 * dt * k1, F)
        k3 = _lorenz96_rhs(ensemble + 0.5 * dt * k2, F)
        k4 = _lorenz96_rhs(ensemble + dt * k3, F)
        ensemble = ensemble + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return ensemble

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 RHS for an array of states (batch)."""
    n_vars = x.shape[-1]
    x_shifted1 = np.roll(x, -1, axis=-1)  # x_{i+1}
    x_shifted2 = np.roll(x, 2, axis=-1)   # x_{i-2}
    x_shifted3 = np.roll(x, 1, axis=-1)   # x_{i-1}
    return (x_shifted1 - x_shifted2) * x_shifted3 - x + F

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    LETKF-based analysis update.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observations (even-indexed variables)
    - observed_indices: array of indices that are observed
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # We'll use block-localization with radius 4.
    # For each observed variable, we consider a local block of state variables
    # within a window of radius 4 (i.e., 9 variables centered on the observed one).
    # But since we have 20 variables and 10 observations, we need to be careful.
    
    # Standard LETKF approach:
    # 1. For each local block, compute the local ensemble mean and anomalies.
    # 2. Apply the ensemble transform to update the local block.
    # 3. Combine the updates.
    
    # Let's use a simpler but effective approach:
    # We'll do a sequential observation update with localization.
    
    # First, let's compute the ensemble mean and spread
    ens_mean = np.mean(ensemble, axis=0)
    
    # For LETKF, we work with anomalies
    anomalies = ensemble - ens_mean  # (n_members, n_vars)
    
    # We'll apply a localized Kalman filter update.
    # For each observed variable, we update the local block.
    
    # Localization: use a tapering function with radius 4
    def localization_factor(i, j):
        """Compute localization factor for pair (i, j)."""
        # Circular distance on 20-variable ring
        dist = min(abs(i - j), n_vars - abs(i - j))
        radius = 4
        if dist > radius:
            return 0.0
        # Gaspari-Cohn tapering function
        # For simplicity, use a quadratic taper
        if dist == 0:
            return 1.0
        # GC taper: for 0 <= r <= L/2, 1 - 5/4*(r/L)^2 + 5/8*(r/L)^4
        # for L/2 < r <= L, 5/2 - 5/4*(r/L)^2 - 5/2*(r/L)^3 + 1/4*(r/L)^5
        r = dist / radius
        if r <= 0.5:
            return 1.0 - 1.25 * r**2 + 0.625 * r**4
        else:
            return 2.5 - 1.25 * r**2 - 2.5 * r**3 + 0.25 * r**4
    
    # Build the localized observation operator
    # For each observation, we'll do a sequential update
    
    updated_ensemble = ensemble.copy()
    
    # Sequential observation updates with localization
    for obs_k, obs_var_idx in enumerate(obs_idx):
        # Observation value
        obs_val = observations[obs_k]
        obs_var = observation_variance
        
        # Local block: variables within radius 4 of obs_var_idx
        local_indices = []
        for d in range(-4, 5):
            idx = (obs_var_idx + d) % n_vars
            local_indices.append(idx)
        
        # Get local ensemble mean and anomalies
        local_ens = updated_ensemble[:, local_indices]  # (n_members, n_local)
        local_mean = np.mean(local_ens, axis=0)
        local_anom = local_ens - local_mean  # (n_members, n_local)
        
        # Local covariance: H' B H where H is the observation operator
        # For a single observation of variable obs_var_idx, H picks out that variable
        # The local variance of the observed variable:
        # B_obs = variance of the observed variable in the local ensemble
        
        # Find the position of obs_var_idx in local_indices
        obs_pos = local_indices.index(obs_var_idx)
        
        # Local variance of the observed variable
        local_obs_var = np.var(local_anom[:, obs_pos])
        
        # Kalman gain
        k = local_obs_var / (local_obs_var + obs_var)
        
        # Innovation
        innovation = obs_val - local_mean[obs_pos]
        
        # Update the local ensemble
        # For LETKF, we use an ensemble transform
        # Simplified: add the gain times the innovation to the observed variable
        # and adjust the anomalies using the correlation structure
        
        # Compute correlations between all local variables and the observed variable
        # corr[i] = cov(local_anom[:, i], local_anom[:, obs_pos]) / local_obs_var
        
        if local_obs_var > 1e-10:
            cov_with_obs = np.dot(local_anom.T, local_anom[:, obs_pos]) / (n_members - 1)
            corrs = cov_with_obs / local_obs_var
        else:
            corrs = np.zeros(len(local_indices))
            corrs[obs_pos] = 1.0
        
        # Apply localization to correlations
        for i, idx in enumerate(local_indices):
            corrs[i] *= localization_factor(obs_var_idx, idx)
        
        # Update the ensemble: for each member, adjust based on the innovation
        # and the correlation with the observed variable
        for m in range(n_members):
            for i, idx in enumerate(local_indices):
                updated_ensemble[m, idx] += k * innovation * corrs[i]
        
        # Preserve ensemble spread: add a small perturbation to maintain spread
        # This is a simple inflation-like step, but applied locally
        # Actually, LETKF preserves spread by construction, but our simplified
        # version may not. Let's add a small inflation.
        # We'll use a multiplicative inflation of 1.01 on the anomalies
    
    # Final inflation to ensure spread
    final_mean = np.mean(updated_ensemble, axis=0)
    final_anom = updated_ensemble - final_mean
    updated_ensemble = final_mean + 1.01 * final_anom
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Ensure finiteness
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
# EVOLVE-BLOCK-END
