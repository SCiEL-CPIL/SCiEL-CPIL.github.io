"""Lorenz-96 assimilation using sequential coordinate-wise Ensemble Square Root Filter."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE


def analyze(ensemble, observations, observed_indices, observation_variance):
    n_members, n_state = ensemble.shape
    n_obs = len(observations)

    ens = ensemble.astype(np.float64).copy()

    # Sequential coordinate-wise EnSRF update
    # For each observation, update all unobserved coordinates by regressing onto it
    obs_var = observation_variance
    local_decay = 2.0  # localization decay length

    for j in range(n_obs):
        if _check_deadline():
            break

        obs_idx = observed_indices[j]
        y_j = observations[j]

        # Ensemble values at this observation point
        x_obs = ens[:, obs_idx]  # (n_members,)
        x_obs_mean = np.mean(x_obs)
        x_obs_anom = x_obs - x_obs_mean

        # Innovation for this observation
        innov = y_j - x_obs_mean

        # Variance of observation error + ensemble spread
        # Use ensemble variance of x_obs
        x_obs_var = np.var(x_obs, ddof=1)
        S = x_obs_var + obs_var

        if S < 1e-12:
            continue

        # Kalman gain scalar for this observation: k = x_obs_var / S
        k = x_obs_var / S

        # Update the observed coordinate directly
        ens[:, obs_idx] = ens[:, obs_idx] + k * innov

        # Update all other coordinates by regressing onto x_obs
        for i in range(n_state):
            if i == obs_idx:
                continue

            # Localization weight
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d * d / (2.0 * local_decay * local_decay))

            if w < 1e-6:
                continue

            x_i = ens[:, i]
            x_i_mean = np.mean(x_i)
            x_i_anom = x_i - x_i_mean

            # Cross-covariance between x_i and x_obs (using current ensemble)
            # Use the original x_obs_anom from before updating obs_idx for consistency
            cross_cov = np.mean(x_i_anom * x_obs_anom) / (n_members - 1)

            # Gain for this coordinate
            k_i = w * cross_cov / S

            ens[:, i] = ens[:, i] + k_i * innov

    # Adaptive inflation: match ensemble spread to expected innovation spread
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean

    # Check if ensemble has collapsed on observed coordinates
    obs_spread = np.std(ens[:, observed_indices], axis=0)
    expected_obs_spread = np.sqrt(obs_var + np.var(ens[:, observed_indices], axis=0))

    # Compute inflation factor based on spread ratio
    ratios = []
    for j in range(n_obs):
        idx = observed_indices[j]
        sp = np.std(ens[:, idx])
        if sp > 1e-10:
            # Target: spread should be at least sqrt(obs_var)
            target = np.sqrt(obs_var)
            r = target / sp
            ratios.append(r)

    if ratios:
        geo_mean = np.exp(np.mean(np.log(np.clip(ratios, 0.5, 3.0))))
        inflation = np.clip(geo_mean, 0.95, 1.5)
    else:
        inflation = 1.0

    # Apply inflation around the mean
    ens = ens_mean + inflation * (ens - ens_mean)

    # Clip and ensure finiteness
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)

    return ens

# EVOLVE-BLOCK-END
