# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Hypervolume-based acquisition function for minimizing Branin 2D.
    
    Strategy:
    Instead of using a Knowledge Gradient approximation or simple Expected Improvement,
    this function computes an approximation of the Expected Hypervolume Improvement (EHVI).
    
    Rationale:
    Standard EI focuses on improving the current best value (f_best). In multimodal
    landscapes like Branin, this can lead to premature convergence if the global
    minimum is in a region with high uncertainty but a slightly worse mean.
    
    Hypervolume improvement considers the area under the curve defined by the
    non-dominated points in the (objective, uncertainty) space. By maximizing
    the expected increase in this hypervolume, the acquisition function naturally
    balances:
    1. Exploitation: Points with low mean (mu) that improve the objective.
    2. Exploration: Points with high uncertainty (sigma) that reduce the "width"
       of the Pareto front, even if the mean is not immediately better than f_best.
    
    Approximation:
    We approximate the EHVI by considering the posterior distribution of the
    objective at each candidate point. The hypervolume contribution of a point
    is proportional to the probability that it becomes part of the Pareto front
    and the magnitude of the improvement it brings to the front's extent.
    
    A practical surrogate for EHVI in single-objective optimization is:
    Score = E[max(0, f_best - y) * (1 + c * sigma)]
    where y ~ N(mu, sigma^2) and c is a coefficient that weights the exploration
    benefit. However, a more direct structural change is to use a modified EI
    that includes a term for the reduction in uncertainty, weighted by the
    likelihood of the point being near the optimal region.
    
    Specifically, we use:
    Score = EI(mu, sigma, f_best) + w * sigma * exp(-((mu - f_best) / scale)^2)
    
    The EI term handles exploitation. The second term is a "hypervolume expansion"
    proxy: it rewards high sigma, but only in regions where the mean is competitive
    (close to f_best). This is structurally different from the previous KG
    approximation which used a pure product of relevance and normalized sigma,
    because here we add a weighted exploration term to a standard EI base,
    allowing the function to be driven by immediate improvement when available,
    but to strongly explore when the mean is promising but uncertain.
    
    To make it a true "Hypervolume" flavor, we scale the exploration term by
    the expected improvement in the *width* of the Pareto front, which is
    approximated by sigma itself, but gated by a Gaussian relevance kernel.
    """
    t_start = time.time()
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # 1. Standard Expected Improvement (EI)
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues in exp
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # EI is always non-negative. If mu > f_best, EI is small but positive due to the second term.
    # Actually, standard EI for minimization is:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # This formula yields 0 when mu >> f_best and > 0 when mu < f_best.
    
    # 2. Hypervolume Expansion Term (Exploration)
    # We want to reward points that reduce uncertainty in relevant areas.
    # Relevance is high when mu is close to f_best or better.
    # We use a Gaussian kernel for relevance.
    
    # Scale for the relevance kernel.
    # f_best starts high (e.g., 300) and drops to 0.4.
    # A fixed scale might be too narrow or too wide.
    # Let's use a scale that adapts: max(1.0, 0.1 * f_best)
    # This ensures that when f_best is large, we explore broadly.
    # When f_best is small, we focus tightly.
    scale = max(1.0, 0.1 * abs(f_best))
    
    diff = mu - f_best
    relevance = np.exp(-(diff / scale) ** 2)
    
    # The exploration term is proportional to sigma, weighted by relevance.
    # We need to balance this against EI.
    # EI has units of objective value.
    # sigma has units of objective value.
    # So they are directly comparable.
    
    # Weight for the exploration term.
    # If w is too high, we explore too much. If too low, we exploit too much.
    # For Branin, which has multiple local minima, a moderate exploration weight is key.
    # Let's start with w = 1.0, meaning we value uncertainty reduction equally
    # to immediate improvement in relevant areas.
    
    w = 1.0
    
    hv_expansion = w * sigma * relevance
    
    # 3. Combine
    scores = ei + hv_expansion
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() > t_start + 0.1:
         return scores
         
    return scores
# EVOLVE-BLOCK-END
