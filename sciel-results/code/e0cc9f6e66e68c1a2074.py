"""Lorenz-96 ensemble assimilation: Particle Flow Filter (PFF) with localization."""
import numpy as np
import time

# Module-level state
_start_time = None
_deadline_margin = 5.0  # seconds before the 150s limit to stop

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96. x is (N,20) or (20,)."""
    if x.ndim == 1:
        xm2 = np.roll(x, 2)
        xm1 = np.roll(x, 1)
        xp1 = np.roll(x, -1)
        return (xp1 - xm2) * xm1 - x + F
    else:
        xm2 = np.roll(x, 2, axis=1)
        xm1 = np.roll(x, 1, axis=1)
        xp1 = np.roll(x, -1, axis=1)
        return (xp1 - xm2) * xm1 - x + F


def _rk4_step(x, F, dt, n_steps=5):
    """Propagate ensemble (N,20) by RK4 for n_steps."""
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _circular_distance(i, j, dim=20):
    """Periodic distance between indices i and j on a ring of size dim."""
    d = abs(i - j)
    return min(d, dim - d)


def _localization_weight(dist, range_param=3.5):
    """Exponential localization weight based on periodic distance."""
    return np.exp(-0.5 * (dist / range_param) ** 2)


def _estimate_F(ensemble):
    """
    Estimate forcing F from the ensemble.
    For Lorenz-96, F is the constant forcing. We can estimate it by
    looking at the equilibrium or using the fact that the mean of
    (xp1 - xm2)*xm1 - x + F should be near zero in the chaotic attractor.
    Actually, a simpler approach: the typical F values are 6, 8, 10.
    We'll estimate F from the ensemble statistics.
    
    For a single member x: dx/dt = (xp1 - xm2)*xm1 - x + F
    At statistical equilibrium, <dx/dt> ≈ 0, so F ≈ <x - (xp1-xm2)*xm1>
    """
    N, dim = ensemble.shape
    # Compute the nonlinear term for each member
    xm2 = np.roll(ensemble, 2, axis=1)
    xm1 = np.roll(ensemble, 1, axis=1)
    xp1 = np.roll(ensemble, -1, axis=1)
    nonlinear = (xp1 - xm2) * xm1  # (N, 20)
    
    # F ≈ mean(x - nonlinear)
    F_est = np.mean(ensemble - nonlinear)
    # Clamp to reasonable range
    F_est = np.clip(F_est, 4.0, 12.0)
    return F_est


def _particle_flow_update(ensemble, observations, observed_indices, observation_variance, F):
    """
    Particle Flow Filter update for Lorenz-96.
    
    The PFF gradually transforms the prior ensemble into a posterior
    that matches the observations by solving a PDE for the flow map.
    We use a simplified version: iteratively nudge the ensemble toward
    observations using a diffusion-like process.
    
    ensemble: (N, 20)
    observations: (n_obs,)
    observed_indices: array of observed state indices
    observation_variance: scalar
    F: forcing parameter
    
    Returns: updated ensemble (N, 20)
    """
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Work with a copy
    particles = ensemble.copy().astype(float)
    
    # Number of flow iterations
    n_flow_steps = 50
    # Learning rate / step size for the flow
    # Scale by observation noise and ensemble spread
    ensemble_spread = np.std(particles, axis=0)
    mean_spread = np.mean(ensemble_spread)
    base_lr = 0.1 * mean_spread / np.sqrt(observation_variance + 1e-10)
    base_lr = np.clip(base_lr, 0.01, 1.0)
    
    # Localization range
    loc_range = 3.0
    
    for step in range(n_flow_steps):
        # Decay learning rate
        lr = base_lr * (1.0 - 0.5 * step / n_flow_steps)
        
        # For each observed index, compute the innovation and apply a
        # localized update to nearby state components
        for obs_i, obs_idx in enumerate(observed_indices):
            y = observations[obs_i]
            
            # Current ensemble mean at this observation point
            y_f_mean = np.mean(particles[:, obs_idx])
            innovation = y - y_f_mean
            
            # Ensemble spread at this point
            y_f_std = np.std(particles[:, obs_idx])
            if y_f_std < 1e-10:
                y_f_std = 1e-10
            
            # Kalman-like gain for this observation
            # S = y_f_std^2 + observation_variance
            S = y_f_std ** 2 + observation_variance
            K = y_f_std ** 2 / S
            
            # Apply update to all particles at the observed index
            # and to nearby indices with localization
            for j in range(dim):
                dist = _circular_distance(obs_idx, j, dim)
                w = _localization_weight(dist, loc_range)
                if w < 0.01:
                    continue
                
                # For the observed index, use full gain
                # For nearby indices, use a reduced gain based on
                # the correlation structure (approximated by localization)
                if j == obs_idx:
                    gain = K
                else:
                    # Estimate correlation via localization weight
                    # The cross-covariance is approximated by w * y_f_std * std_j
                    std_j = np.std(particles[:, j])
                    cross_cov = w * y_f_std * std_j
                    gain = cross_cov / S
                
                # Update each particle
                # innovation_j = y - particles[:, obs_idx] for the observed point
                # For nearby points, use the same innovation scaled
                innov_all = y - particles[:, obs_idx]  # (N,)
                
                # For the observed index, use full innovation
                # For others, the innovation is the same but the gain differs
                particles[:, j] += lr * gain * innov_all * (1.0 if j == obs_idx else w)
        
        # Keep particles finite and bounded
        particles = np.where(np.isfinite(particles), particles, 0.0)
        particles = np.clip(particles, -100, 100)
    
    return particles


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 using Particle Flow Filter.
    
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: array of indices observed (should be [0,2,4,...,18])
    observation_variance: scalar variance of observation noise
    
    Returns: updated ensemble (16, 20)
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > 150.0 - _deadline_margin:
        # Return current ensemble (clipped) as fallback
        result = np.clip(ensemble, -100, 100)
        return np.where(np.isfinite(result), result, 0.0)
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure finite inputs
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0)
    observations = np.where(np.isfinite(observations), observations, 0.0)
    
    # Estimate F
    F = _estimate_F(ensemble)
    
    # Apply Particle Flow Filter update
    updated = _particle_flow_update(ensemble, observations, observed_indices, observation_variance, F)
    
    # Final safety: clip and ensure finite
    updated = np.clip(updated, -100, 100)
    updated = np.where(np.isfinite(updated), updated, 0.0)
    
    return updated
