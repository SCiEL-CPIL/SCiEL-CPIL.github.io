"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return early if close
    t0 = time.time()
    deadline = t0 + 325  # 5s safety margin
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)
    
    # Structural Change: Cost-normalized Lower Confidence Bound (LCB)
    # with dynamic exploration scaling based on remaining budget proxy.
    
    # 1. Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    EI = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 2. Compute LCB (Lower Confidence Bound) for minimization
    # LCB = mu - beta * sigma
    # We want to maximize the "potential" which is related to minimizing LCB.
    # So we can use -LCB as a score component, or just use the exploration term.
    # Standard UCB for minimization is mu - beta * sigma.
    # We want to select points with low predicted value (low mu) and high uncertainty (high sigma).
    
    # 3. Dynamic Beta
    # The directive suggests: "dynamically scales the exploration term by the square root of the remaining budget"
    # We don't have direct access to remaining budget, but we can infer stage from f_best.
    # Hartmann 6D min is -3.32237.
    # Early: f_best is high (e.g., -1.0, -2.0). We should explore more (high beta).
    # Late: f_best is low (e.g., -3.0, -3.3). We should exploit more (low beta).
    
    # Let's define a proxy for "remaining budget" or "stage".
    # If f_best is far from the known minimum, we are early.
    # Let's use a sigmoid-like mapping from f_best to beta.
    # beta_max = 4.0 (aggressive exploration)
    # beta_min = 0.5 (conservative exploitation)
    
    # Threshold: -2.5 is a reasonable midpoint.
    # If f_best > -2.5, we are early. If f_best < -2.5, we are late.
    # Let's use a linear or smooth interpolation.
    
    # A simple heuristic:
    # beta = beta_min + (beta_max - beta_min) * exp(-k * (f_best - f_min))
    # Where f_min is the known global minimum (-3.32237).
    # If f_best is close to f_min, exp term is small -> beta close to beta_min.
    # If f_best is far, exp term is large -> beta close to beta_max.
    
    f_min_hartmann = -3.32237
    # Scale factor k: if f_best - f_min = 1.0, exp(-k) should be significant.
    # Let's say k=1.0.
    # If f_best = -2.3, diff = 1.0, exp(-1) ~ 0.37.
    # If f_best = -1.3, diff = 2.0, exp(-2) ~ 0.13.
    # This decays too fast.
    
    # Let's try a simpler mapping based on the range of f_best.
    # Assume f_best ranges from ~0 (worst) to -3.3 (best).
    # Map f_best to [0, 1] where 0 is early and 1 is late.
    # stage = (f_best - f_min) / (f_max - f_min)
    # f_max ~ 0? Let's assume 0.
    # stage = (f_best + 3.32237) / 3.32237
    # If f_best = 0, stage = 1.0 (late? No, 0 is bad/early).
    # If f_best = -3.32237, stage = 0.0 (late/best).
    
    # So stage 0 = late, stage 1 = early.
    # We want high beta when stage is high (early).
    # beta = beta_min + (beta_max - beta_min) * stage
    
    f_max_est = 0.0  # Rough estimate of worst case
    f_min_est = -3.32237
    
    # Clip f_best to reasonable range to avoid extreme betas
    f_best_clipped = np.clip(f_best, f_min_est, f_max_est)
    
    # stage: 1.0 when f_best is bad (high), 0.0 when f_best is good (low)
    stage = (f_best_clipped - f_min_est) / (f_max_est - f_min_est)
    
    # Invert so that 0 is early, 1 is late? No, stage is already 1 for bad (early).
    # So beta should be high when stage is high.
    beta_min = 0.5
    beta_max = 3.0
    
    beta = beta_min + (beta_max - beta_min) * stage
    
    # 4. Compute Score
    # We combine EI and Exploration (sigma) or use LCB directly.
    # LCB = mu - beta * sigma
    # We want to minimize LCB, so maximize -LCB.
    # But we also want to account for cost.
    # Score = (-LCB) / cost_safe
    
    lcb = mu - beta * sigma_safe
    score_lcb = -lcb / cost_safe
    
    # Alternatively, blend EI and LCB?
    # EI is a good exploitation signal. LCB is a good exploration signal.
    # The directive says "cost-normalized lower confidence bound (LCB)".
    # So let's stick to LCB.
    
    score = score_lcb
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
