"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Ensure arrays are 1-D numpy arrays of the right type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.size
    if n == 0:
        return np.array([], dtype=np.float64)

    # Floor sigma to avoid division by zero / log of zero
    sigma_safe = np.maximum(sigma, 1e-12)

    # Potential improvement (we are minimizing, so improvement = f_best - mu)
    delta = f_best - mu

    # ---------- Expected Improvement (EI) ----------
    z = delta / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = delta * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)

    # ---------- Hypervolume-style Expected Improvement (HEI) ----------
    # HEI ≈ EI * (1 + log(1 + EI / sigma))  – rewards larger improvements
    # Use a mild log bonus that is numerically stable.
    hei = ei * (1.0 + np.log1p(ei / (sigma_safe + 1e-30)))
    hei = np.maximum(hei, 0.0)

    # ---------- Cost-aware scaling ----------
    # cost = 1 + mean(x_scaled) ∈ [1, 2].  We want to prefer cheap points.
    # Instead of a hard division, use a soft power:  cost^-alpha  with alpha in (0,1]
    # This avoids over-penalising slightly more expensive but much better candidates.
    cost_safe = np.maximum(cost, 1e-12)

    # Adaptive alpha: early on (low mean cost) we can afford to be less
    # cost-sensitive and explore; later we tighten cost awareness.
    mean_cost = float(np.mean(cost))
    # cost typically in [1, 2]; normalise to [0, 1]
    norm_cost = np.clip((mean_cost - 1.0) / 1.0, 0.0, 1.0)
    alpha = 0.4 + 0.6 * norm_cost          # 0.4 → 1.0
    cost_weight = cost_safe ** (-alpha)     # cheaper points get higher weight

    # ---------- Exploration term (cost-aware UCB) ----------
    # UCB for minimisation:  f_best - mu + kappa * sigma
    # Make kappa adapt to the "confidence" of the model: when sigma is large
    # relative to the spread of mu, we are in an unexplored region → explore.
    mu_spread = np.std(mu) + 1e-30
    sigma_spread = np.std(sigma_safe)
    # If the GP is uncertain everywhere, increase kappa; if it is confident,
    # decrease it.
    kappa_base = 1.2
    # Scale factor in [0.5, 2.0]
    uncertainty_ratio = sigma_spread / (mu_spread + sigma_spread)
    kappa = kappa_base * (0.5 + 1.5 * uncertainty_ratio)

    ucb = f_best - mu + kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)

    # ---------- Combine EI, HEI and UCB with adaptive weights ----------
    # Normalise each term to [0, 1] roughly by dividing by its max (with guard).
    def _normalise(arr):
        m = np.max(arr)
        if m > 1e-18:
            return arr / m
        return arr

    ei_n = _normalise(ei)
    hei_n = _normalise(hei)
    ucb_n = _normalise(ucb)

    # Weights: as the budget is consumed (norm_cost ↑) shift weight from
    # exploration (UCB) to exploitation (EI/HEI).
    w_ei = 0.35 + 0.20 * norm_cost     # 0.35 → 0.55
    w_hei = 0.25 + 0.15 * norm_cost    # 0.25 → 0.40
    w_ucb = 1.0 - w_ei - w_hei         # 0.40 → 0.05

    scores = w_ei * ei_n + w_hei * hei_n + w_ucb * ucb_n

    # Apply cost weighting
    scores = scores * cost_weight

    # ---------- Final normalisation for numerical stability ----------
    s_mean = np.mean(scores)
    s_std = np.std(scores)
    if s_std > 1e-12:
        scores = (scores - s_mean) / s_std
    else:
        scores = scores - s_mean

    # Guarantee finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
