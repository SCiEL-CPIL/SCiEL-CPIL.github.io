"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill

    # Input validation and reshaping
    try:
        prizes = np.asarray(prizes, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if prizes.shape != (5, 100):
            prizes = prizes.reshape(5, 100)
        if weights.shape != (5, 100):
            weights = weights.reshape(5, 100)
    except Exception:
        return np.ones(100) * 1e-9

    epsilon = 1e-9

    # 1. Calculate per-constraint efficiency ratios
    # r[k, j] = prizes[k, j] / weights[k, j]
    safe_weights = weights + epsilon
    ratios = prizes / safe_weights

    # 2. Calculate Harmonic Mean of the ratios
    # H[j] = 1 / mean_k(1 / r[k, j])
    inv_ratios = 1.0 / ratios
    harmonic_mean = 1.0 / (np.mean(inv_ratios, axis=0) + epsilon)

    # 3. Normalize the harmonic mean to have a mean of 1
    # This ensures the heuristic is on the same scale as the initial pheromone (1.0)
    mean_h = np.mean(harmonic_mean)
    if mean_h > epsilon:
        harmonic_mean_norm = harmonic_mean / mean_h
    else:
        harmonic_mean_norm = np.ones(100)

    # 4. Apply a power to sharpen the selection
    # Power 2.0 amplifies the difference between good and bad items
    heuristic = np.power(harmonic_mean_norm + epsilon, 2.0)

    # 5. Ensure positivity and finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # Check deadline
    if time.time() > deadline:
        return heuristic

    # Verify shape
    if heuristic.shape != (100,):
        heuristic = heuristic.reshape(100)

    return heuristic
# EVOLVE-BLOCK-END
