"""Step-adaptive blended acquisition: early LCB for exploration, late EI for exploitation,
with a rank-based diversity bonus that spreads candidates without hyperparameter search."""
import time
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    t0 = time.time()
    deadline = t0 + 300.0 - 5.0  # safety margin

    s = sigma + 1e-9

    # --- Infer progress from posterior uncertainty ---
    # As more points are evaluated, overall sigma drops.
    # Use the median sigma as a proxy for "how much we've learned".
    med_sigma = float(np.median(sigma))
    # At start (5 points), med_sigma is relatively large; late (30 points), it's small.
    # Normalize: progress in [0, 1]. We don't know the exact scale, so use a soft mapping.
    # A typical median sigma in [0,1]^2 with length-scale 0.2 starts ~0.3-0.5 and drops to ~0.05-0.15.
    # Map med_sigma from [0.5, 0.05] -> progress [0, 1] with a smooth sigmoid.
    progress = 1.0 / (1.0 + np.exp(-8.0 * (0.25 - med_sigma)))
    # progress ~ 0 when med_sigma > 0.25 (early), ~ 1 when med_sigma < 0.25 (late)
    # Actually we want: early -> more exploration, late -> more exploitation
    # So blend weight for EI = progress, weight for LCB = 1 - progress
    # Let's refine: early (few points) -> med_sigma large -> progress small -> LCB dominant
    # Late (many points) -> med_sigma small -> progress large -> EI dominant

    # Clamp progress to [0.05, 0.95] to keep both components active
    progress = np.clip(progress, 0.05, 0.95)

    # --- Compute EI ---
    z_ei = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z_ei) + s * norm.pdf(z_ei)

    # --- Compute LCB ---
    # Beta: start high (explore) and decrease as we learn.
    # beta = 2.5 early, 1.0 late
    beta = 1.0 + 1.5 * (1.0 - progress)
    lcb = mu - beta * s
    # Normalize LCB to be comparable to EI (both should be "higher = better")
    # EI is in objective units (positive = improvement). LCB is in objective units (lower = better).
    # We want a single score where higher = evaluate next.
    # For LCB, we invert: higher score = lower mu - beta*sigma.
    # But we need to blend with EI which has a different scale.
    # Instead, let's compute a "desirability" for each:
    # EI is already a desirability (higher = better).
    # For LCB, the desirability is how much lower the LCB is relative to f_best.
    # lcb_desirability = f_best - (mu - beta * s) = (f_best - mu) + beta * s
    lcb_desirability = (f_best - mu) + beta * s

    # --- Blend ---
    # Weight EI by progress, LCB by (1 - progress)
    # But they have different scales. Let's normalize each by their max.
    ei_max = float(np.max(ei)) + 1e-12
    lcb_max = float(np.max(lcb_desirability)) + 1e-12

    ei_norm = ei / ei_max
    lcb_norm = lcb_desirability / lcb_max

    blended = progress * ei_norm + (1.0 - progress) * lcb_norm

    # --- Diversity bonus based on sigma rank ---
    # Points with higher sigma (more uncertain) get a small bonus to encourage exploration
    # of less-visited regions. Use rank to be scale-invariant.
    sigma_rank = np.argsort(np.argsort(sigma)).astype(float) / len(sigma)
    # sigma_rank in [0, 1], 1 = highest sigma
    # Bonus: small, decays with progress (less needed when we're exploiting)
    diversity_weight = 0.1 * (1.0 - progress)
    diversity_bonus = diversity_weight * sigma_rank

    # --- Final score ---
    score = blended + diversity_bonus

    # Ensure finite
    if not np.all(np.isfinite(score)):
        score = ei

    return score
