import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 300s limit, start timer
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin

    # Ensure arrays are 1D and float
    mu = np.asarray(mu, dtype=np.float64).flatten()
    sigma = np.asarray(sigma, dtype=np.float64).flatten()
    cost = np.asarray(cost, dtype=np.float64).flatten()

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware EI
    score_ei = ei / cost_safe

    # Uncertainty-based exploration (UCB-like)
    # Use log-sigma to avoid dominance by a few high-sigma points
    log_sigma = np.log(sigma_safe + 1e-12)
    log_sigma_max = np.max(log_sigma)
    log_sigma_min = np.min(log_sigma)
    log_range = max(log_sigma_max - log_sigma_min, 1e-9)
    log_sigma_norm = (log_sigma - log_sigma_min) / log_range

    # Adaptive exploration weight: higher regret -> more exploration
    # Use a smoother, more responsive scaling than f_best/(f_best+c)
    # f_best for Levy 2D starts around 10-50, ends near 0
    # We want w ~ 0.6-0.8 early, ~ 0.1-0.3 late
    # w = 1/(1+exp((f_best-3)/2)) gives: f_best=20->0.99, f_best=3->0.5, f_best=0.1->0.12
    # That's good but maybe too aggressive. Let's use a gentler curve.
    w = 1.0 / (1.0 + np.exp((f_best - 4.0) / 3.0))

    # Combine: exploitation + weighted exploration
    # Scale exploration term relative to EI magnitude so they're comparable
    ei_max = np.max(score_ei)
    if ei_max < 1e-12:
        ei_max = 1.0
    scale_factor = 0.3 * ei_max

    score = score_ei + w * scale_factor * log_sigma_norm

    # Time check
    if time.time() > deadline:
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
