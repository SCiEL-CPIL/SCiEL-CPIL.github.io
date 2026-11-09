# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for minimizing hartmann 3d.
    
    Uses a Knowledge Gradient (KG) style acquisition that estimates the
    probability that a candidate point is the global minimum, combined
    with a cost-aware scaling. This is structurally different from simple
    LCB/EI approaches as it explicitly models the probability of finding
    the global optimum.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate

    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin before 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)

    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    n = len(mu_safe)
    
    # Knowledge Gradient approach:
    # The KG for minimization estimates the expected reduction in the
    # best value by evaluating at point x. A common approximation is:
    # KG(x) ≈ P(f(x) < f_best) * (f_best - E[f(x) | f(x) < f_best])
    # 
    # For a normal distribution N(mu, sigma^2):
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # E[f(x) | f(x) < f_best] = mu - sigma * phi(z) / Phi(z)
    # where z = (f_best - mu) / sigma
    #
    # So KG(x) = Phi(z) * (f_best - mu + sigma * phi(z) / Phi(z))
    #          = Phi(z) * (f_best - mu) + sigma * phi(z)
    #
    # This is actually the Expected Improvement (EI) formula!
    # But the key insight for KG is to use it in a way that accounts
    # for the global nature of the problem.
    
    # Compute z-scores
    z = (f_best - mu_safe) / sigma_safe
    z = np.clip(z, -10, 10)  # numerical stability
    
    # Standard EI
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Avoid division by zero
    cdf_z_safe = np.where(cdf_z > 1e-10, cdf_z, 1e-10)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu_safe) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # KG-style: The expected improvement in the best value
    # We can enhance this by considering the probability that this point
    # is the global minimum among all candidates
    # 
    # A practical KG approximation: weight EI by the probability that
    # this point is the best among all 2000 candidates
    # P(x is best) ≈ 1 / (1 + sum_{j != i} P(f(j) < f(i)))
    # 
    # For efficiency, we use a simpler approximation:
    # The relative rank of the EI value determines the KG weight
    # Points with higher EI are more likely to be the global minimum
    
    # Normalize EI to [0, 1] for ranking
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_normalized = np.ones_like(ei) / n
    else:
        ei_normalized = (ei - ei_min) / ei_range
    
    # KG weight: use a softmax-like weighting that emphasizes top candidates
    # Temperature controls how peaked the distribution is
    temp = 0.1
    kg_weights = np.exp(ei_normalized / temp)
    kg_weights /= np.sum(kg_weights)
    
    # Combine EI with KG weight
    # The KG score is the expected improvement weighted by the probability
    # that this point is the global minimum
    kg_scores = ei * (1 + 5 * kg_weights)
    
    # Cost-aware adjustment
    # Cost is in [1, 2], so we divide by cost to prefer cheaper evaluations
    # But we also want to account for the fact that expensive evaluations
    # might be more informative if they're in high-uncertainty regions
    cost_adjusted_scores = kg_scores / cost_safe
    
    # Ensure finite
    cost_adjusted_scores = np.where(np.isfinite(cost_adjusted_scores), cost_adjusted_scores, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return cost_adjusted_scores
    
    # Alternative: Use a hybrid approach that combines EI with a
    # probability-based KG term
    if n >= 100 and time.time() < deadline - 10.0:
        # Compute the probability that each point is the global minimum
        # using a pairwise comparison approximation
        # P(i is best) = prod_{j != i} P(f(i) < f(j))
        # 
        # For efficiency, we use a rank-based approximation
        # Sort by mu + sigma (optimistic bound)
        optimistic = mu_safe - 2 * sigma_safe
        ranks = np.argsort(optimistic)
        
        # Assign probabilities based on rank
        # Top ranks get higher probability
        rank_probs = np.zeros_like(optimistic)
        for r, idx in enumerate(ranks):
            rank_probs[idx] = 1.0 / (r + 1)
        
        # Normalize
        rank_probs /= np.sum(rank_probs)
        
        # Combine with EI
        hybrid_scores = ei * (1 + 3 * rank_probs)
        hybrid_cost = hybrid_scores / cost_safe
        hybrid_cost = np.where(np.isfinite(hybrid_cost), hybrid_cost, 0.0)
        
        # Compare the two approaches
        # Use the one with higher mean score in the top 10%
        top_k = max(1, n // 10)
        idx_kg = np.argsort(cost_adjusted_scores)[-top_k:]
        idx_hybrid = np.argsort(hybrid_cost)[-top_k:]
        
        mean_kg = np.mean(cost_adjusted_scores[idx_kg])
        mean_hybrid = np.mean(hybrid_cost[idx_hybrid])
        
        if mean_hybrid > mean_kg:
            cost_adjusted_scores = hybrid_cost
    
    # Final safety
    cost_adjusted_scores = np.where(np.isfinite(cost_adjusted_scores), cost_adjusted_scores, 0.0)
    
    return cost_adjusted_scores
# EVOLVE-BLOCK-END
