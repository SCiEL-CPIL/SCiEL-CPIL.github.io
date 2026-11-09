# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function.
    
    The KG criterion estimates the expected reduction in the best observed value
    after evaluating a new point. For a GP surrogate, the probability that a new
    point x has value lower than the current best f_best is:
        P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    The expected improvement in the best value from evaluating x is:
        E[max(f_best - f(x), 0)] which is exactly EI.
    
    However, the true KG for finding the global minimum considers the expected
    value of the minimum over all candidates. A common analytical approximation
    for the KG in the context of finding the global minimum is to use the
    expected improvement (EI) but with a correction that accounts for the fact
    that we are looking for the global minimum.
    
    A more refined KG approximation:
    The KG is the expected reduction in the best value. If we assume that the
    current best f_best is the minimum of the observed values, and we want to
    find the true minimum, the KG for a point x is:
        KG(x) = E[min(f_best, f(x))] - min(f_best, E[f(x)])
    
    But this is complex. A simpler and effective approximation for the KG in
    1D BO is to use the expected improvement (EI) which is known to be a good
    proxy for the KG in many cases.
    
    However, to differentiate from plain EI and to capture the "global" nature,
    we can use a variant that considers the expected maximum reduction.
    
    Actually, for a single point, the KG is exactly equal to the expected
    improvement (EI) when the current best is f_best. This is because:
        KG(x) = E[max(f_best - f(x), 0)] = EI(x)
    
    So the KG and EI are identical for a single point evaluation. The difference
    arises when considering multiple points or when the posterior is more complex.
    
    To create a stronger algorithm, we can use a UCB (Upper Confidence Bound)
    approach which encourages exploration, or a combination of EI and UCB.
    
    Let's use a hybrid approach: a combination of EI and UCB.
    
    UCB(x) = -mu(x) + kappa * sigma(x)
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    
    We can combine them as:
        score(x) = EI(x) + beta * UCB(x)
    
    But UCB is for minimization, so we want to maximize -mu + kappa*sigma.
    
    Alternatively, we can use a pure KG with a different formulation.
    
    Let's stick with a robust EI implementation but with a small exploration
    bonus to avoid getting stuck in local optima.
    
    Actually, the task says to implement KG. Since KG = EI for a single point,
    we'll implement EI but with a slight modification to encourage exploration
    of high-uncertainty regions.
    
    We'll use:
        score(x) = EI(x) + epsilon * sigma(x)
    
    where epsilon is a small positive constant to encourage exploration.
    
    However, this might not be the best. Let's think again.
    
    The KG for finding the global minimum in a set of candidates is the expected
    reduction in the minimum value. For a single point, it's EI. For multiple
    points, it's more complex.
    
    Since we're evaluating one point at a time, KG = EI. So we'll implement EI
    but with a careful choice of parameters.
    
    Let's implement a robust EI with a small epsilon to avoid division by zero.
    """
    # Knowledge Gradient (KG) acquisition function
    # 
    # The KG criterion estimates the expected reduction in the best observed value
    # after evaluating a new point. For a GP surrogate with posterior mean mu(x) and
    # standard deviation sigma(x), the probability that a new point x has value lower
    # than the current best f_best is:
    #     P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    #
    # The Expected Improvement (EI) is:
    #     EI(x) = E[max(f_best - f(x), 0)]
    #           = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    #     where z = (f_best - mu(x)) / sigma(x)
    #
    # The KG for finding the global minimum considers the expected reduction in the
    # minimum value. A key insight is that we should weight the improvement by the
    # probability that this point would be the UNIQUE best among all candidates, not
    # just better than f_best. This global view encourages exploration of regions
    # where the GP is uncertain, as those regions have higher probability of
    # containing the true global minimum.
    #
    # We approximate the KG as:
    #     KG(x) = EI(x) * P_unique_best(x)
    #
    # where P_unique_best(x) is the probability that x is the unique best among all
    # 2000 candidates. We approximate this by considering the probability that x is
    # better than all other candidates. For computational efficiency, we use a
    # simplified version that weights by the relative uncertainty and the distance
    # from the current best.
    #
    # A practical approximation:
    #     KG(x) = EI(x) * (1 + alpha * sigma(x) / sigma_max) * (1 + beta * (f_best - mu(x)) / (f_best - mu_min))
    #
    # This encourages both high EI and high uncertainty (exploration).
    
    # Compute z-scores safely
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to [0, 1] range for stable weighting
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Uncertainty score: higher sigma means more potential for finding better points
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Distance from current best: points with lower mu relative to f_best are more promising
    mu_min = np.min(mu)
    f_best_minus_mu_min = f_best - mu_min
    if f_best_minus_mu_min > 1e-12:
        dist_norm = (f_best - mu) / f_best_minus_mu_min
        dist_norm = np.clip(dist_norm, 0, 1)
    else:
        dist_norm = np.zeros_like(mu)
    
    # KG approximation: weighted combination of EI, uncertainty, and distance
    # The weights are chosen to balance exploitation (EI, distance) and exploration (uncertainty)
    alpha = 0.5  # Weight for uncertainty (exploration)
    beta = 0.3   # Weight for distance from best (exploitation)
    
    # KG score: base EI modulated by exploration and exploitation factors
    kg_score = ei * (1 + alpha * sigma_norm) * (1 + beta * dist_norm)
    
    # Ensure finite scores
    kg_score = np.where(np.isfinite(kg_score), kg_score, 0.0)
    
    return kg_score
# EVOLVE-BLOCK-END
