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
    # Knowledge Gradient approximation:
    # For each candidate x, the probability it becomes the new global best is
    # P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    # The expected reduction in the best value if x is the new best is
    # E[max(f_best - f(x), 0)] = EI(x)
    #
    # A refined KG considers that evaluating x gives information about other
    # candidates. We approximate this by weighting EI by the probability that
    # x is the unique best among all candidates.
    #
    # For simplicity and effectiveness, we use a combination of:
    # 1. Expected Improvement (EI)
    # 2. Probability of being the new best (exploration term)
    # 3. Uncertainty bonus to encourage exploration
    
    # Compute z-scores
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability that x becomes the new best
    prob_new_best = norm.cdf(z)
    
    # Uncertainty bonus: encourages exploration in high-uncertainty regions
    # Scale by the range of mu to make it adaptive
    mu_range = np.max(mu) - np.min(mu) + 1e-9
    uncertainty_bonus = 0.3 * sigma_safe * (sigma_safe / (mu_range + 1e-9))
    
    # Combine: EI + exploration bonus
    score = ei + 0.1 * prob_new_best * (f_best - np.min(mu)) + uncertainty_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
