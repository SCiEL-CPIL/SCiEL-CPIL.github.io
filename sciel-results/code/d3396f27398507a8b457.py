import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
TIME_LIMIT = 290.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to simple EI/cost if running out of time
        eps = 1e-9
        z = (f_best - mu) / (sigma + eps)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / np.maximum(cost, eps)

    eps = 1e-9
    sigma_safe = sigma + eps
    cost_safe = np.maximum(cost, eps)
    f_best_safe = np.maximum(f_best, eps)

    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z

    # Compute Lower Confidence Bound score (higher = more promising)
    # LCB = mu + beta * sigma (we want to minimize, so lower LCB is better)
    # Score = -LCB = -(mu + beta * sigma)
    beta = 2.0
    lcb_score = -(mu + beta * sigma)

    # Combine EI and LCB: take maximum to allow both exploitation and exploration
    combined = np.maximum(ei, lcb_score)

    # Cost-aware: divide by cost
    score = combined / cost_safe

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
