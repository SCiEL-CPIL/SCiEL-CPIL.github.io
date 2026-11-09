# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard
    t_start = time.time()
    DEADLINE = t_start + 280  # 300s limit, 20s margin

    sigma_safe = np.maximum(sigma, 1e-8)
    diff = f_best - mu

    # Compute Expected Improvement (EI)
    z = diff / sigma_safe
    # Use vectorized erf for CDF
    cdf_val = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf_val = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    ei = diff * cdf_val + sigma_safe * pdf_val

    # Compute Upper Confidence Bound (UCB)
    # UCB = mu + beta * sigma (for minimization, we want to maximize this score)
    # Actually, for minimization, UCB score = mu - beta * sigma is wrong.
    # We want to explore where sigma is high.
    # Standard UCB for minimization: score = mu - beta * sigma? No.
    # We want to minimize f.
    # Lower bound of f is mu - beta*sigma. We want to find low values.
    # So we should prioritize points with low mu and high sigma.
    # A common acquisition function for minimization is:
    # Score = mu - beta * sigma (optimistic/pessimistic?)
    # Let's think:
    # If we use EI, we are exploiting.
    # If we use UCB for minimization, we often use:
    # q(x) = mu(x) - beta * sigma(x)
    # This encourages exploring regions with high sigma (uncertainty) because it lowers the score, making it more attractive for minimization.
    # Wait, the task says "higher meaning evaluate next".
    # So we want to return scores where higher is better.
    # For minimization, we want to find points with low f.
    # So a point is "good" if its expected f is low.
    # EI measures how much better than f_best we can expect. Higher EI is better.
    # UCB for minimization:
    # If we define UCB_score = -mu + beta*sigma, then high score means low mu and/or high sigma.
    # Let's use a hybrid or just UCB.
    # The inspiration mentions UCB with beta=2.5 improved score.
    
    beta = 2.5
    ucb_score = -mu + beta * sigma_safe

    # We can blend EI and UCB or just use UCB.
    # Let's try a weighted sum.
    # Normalize EI and UCB to comparable scales?
    # Or just use UCB since it showed promise.
    
    # Let's try pure UCB first as suggested by the finding.
    result = ucb_score

    # Check deadline
    if time.time() > DEADLINE:
         # If we are close to deadline, just return what we have
         pass

    # Ensure finite results
    result[~np.isfinite(result)] = -1e10

    return result
# EVOLVE-BLOCK-END
