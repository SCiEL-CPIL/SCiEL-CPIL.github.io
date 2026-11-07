"""Cost-aware acquisition for BO on four benchmark functions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: Ensure we return well before the 330s limit.
    # The evaluator kills at 330s. We aim to finish each call quickly.
    t_start = time.time()
    
    # Ensure arrays are 1-D numpy arrays of the right type
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.size
    if n == 0:
        return np.array([], dtype=np.float64)

    # Floor sigma to avoid division by zero / log of zero
    # 1e-12 is a safe floor for standard deviation
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)

    # Potential improvement (we are minimizing, so improvement = f_best - mu)
    delta = f_best - mu

    # ---------- Expected Improvement (EI) ----------
    # Standard EI for minimization
    z = delta / sigma_safe
    # Clip z to avoid extreme values in cdf/pdf if sigma is tiny
    z_clipped = np.clip(z, -50, 50)
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    ei = delta * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)

    # ---------- Cost-aware scaling ----------
    # cost = 1 + mean(x_scaled) in [1, 2].
    # We prefer cheap points.
    # Adaptive alpha: 
    # Early on (low mean cost implies we haven't explored much or are in cheap regions?),
    # actually mean cost is just a proxy for how much budget we've spent? 
    # No, cost is per-point. 
    # Let's look at the "norm_cost" logic from the inspirations.
    # mean_cost is the average cost of the *candidates*? No, cost array corresponds to the 2000 candidates.
    # Wait, the task says: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # The input `cost` is the cost of each of the 2000 candidates.
    # The "budget" logic in the inspirations used `mean_cost` of the *candidates* to determine alpha.
    # This is a bit heuristic. If candidates are generally expensive, we should be more cost-sensitive?
    # Or does it track the *total* budget? The function signature doesn't give total budget.
    # It only gives `cost` for the current candidates.
    # However, `f_best` and the history are not passed, only `mu, sigma, f_best, cost`.
    # The inspirations used `mean_cost = float(np.mean(cost))`.
    # If the candidates are uniformly sampled, the mean cost is roughly constant (1.5).
    # So `norm_cost` would be roughly constant (0.5).
    # This suggests the "adaptive" part based on `mean(cost)` of candidates is likely not adapting to budget consumption, 
    # but to the *local* cost distribution of the candidates being scored.
    # Let's stick to a simple, effective cost penalty.
    
    # A penalty of cost^(-alpha) rewards low cost.
    # Let's use a fixed alpha or a mildly adaptive one.
    # Inspiration 1 used alpha = 0.3 + 0.5 * norm_cost.
    # If norm_cost ~ 0.5, alpha ~ 0.55.
    # Let's use alpha = 0.5 as a robust middle ground, or slightly lower to encourage exploration.
    # Let's try to be slightly more cost-sensitive later? We don't know "later".
    # We can use the *spread* of costs or just a fixed value.
    # Let's use a fixed alpha = 0.6. This balances exploration (low alpha) and exploitation of cheap points (high alpha).
    
    alpha = 0.6
    cost_weight = cost_safe ** (-alpha)

    # ---------- Exploration term (UCB) ----------
    # UCB for minimization: f_best - mu + kappa * sigma
    # We want to balance exploitation (low mu) and exploration (high sigma).
    
    # Adaptive kappa: 
    # If the model is very uncertain (high sigma relative to mu spread), explore more.
    mu_spread = np.std(mu) + 1e-30
    sigma_spread = np.std(sigma_safe)
    
    # If sigma_spread is high, uncertainty is high.
    # uncertainty_ratio = sigma_spread / (mu_spread + sigma_spread)
    # This ratio is in [0, 1].
    uncertainty_ratio = sigma_spread / (mu_spread + sigma_spread)
    
    # Base kappa of 1.0 is a standard starting point.
    # We scale it between 0.5 and 2.0 based on uncertainty.
    kappa_base = 1.0
    kappa = kappa_base * (0.5 + 1.5 * uncertainty_ratio)

    ucb = f_best - mu + kappa * sigma_safe
    ucb = np.maximum(ucb, 0.0)

    # ---------- Combine EI and UCB ----------
    # Normalize EI and UCB to [0, 1] range to make weights comparable.
    def _normalise(arr):
        m = np.max(arr)
        if m > 1e-18:
            return arr / m
        return arr

    ei_n = _normalise(ei)
    ucb_n = _normalise(ucb)

    # Weights:
    # Inspiration 1: w_ei = 0.50 + 0.25 * norm_cost, w_ucb = 1 - w_ei
    # Since norm_cost is roughly constant for uniform candidates, this is effectively fixed weights.
    # Let's use fixed weights that favor exploitation slightly more as we assume the GP is decent.
    # w_ei = 0.6, w_ucb = 0.4 is a common robust setting.
    # However, for multimodal functions, more UCB might be needed.
    # Let's try w_ei = 0.55, w_ucb = 0.45.
    
    w_ei = 0.55
    w_ucb = 0.45

    scores = w_ei * ei_n + w_ucb * ucb_n

    # Apply cost weighting
    scores = scores * cost_weight

    # ---------- Final normalisation for numerical stability ----------
    # This step ensures the scores are centered and have unit variance,
    # which can help if the evaluator or subsequent steps are sensitive to scale.
    # However, the task says "Return 2000 finite scores (higher = evaluate next)".
    # The relative ordering is what matters. Normalization preserves ordering.
    
    s_mean = np.mean(scores)
    s_std = np.std(scores)
    if s_std > 1e-12:
        scores = (scores - s_mean) / s_std
    else:
        scores = scores - s_mean

    # Guarantee finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Check deadline (should be very fast, but just in case)
    if time.time() - t_start > 2.0:
        # If it took too long, return a simple fallback
        return np.zeros_like(mu)

    return scores
# EVOLVE-BLOCK-END
