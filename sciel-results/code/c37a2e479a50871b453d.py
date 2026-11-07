"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Knowledge Gradient (KG) acquisition for minimization
    # KG estimates the expected reduction in the global minimum value.
    # For a candidate x_i with predictive distribution N(mu_i, sigma_i^2):
    #   The expected improvement in the global minimum if we evaluate at x_i is:
    #   E[max(0, f_best - Y_i)] where Y_i ~ N(mu_i, sigma_i^2)
    #   But this is just EI. True KG considers the change in the expected minimum over all points.
    #   A common approximation is:
    #   KG_i = E[max(0, f_best - Y_i)] - E[max(0, f_best - min_j Y_j)]
    #   where the second term is the expected minimum of the predictive distribution over all candidates.
    #   Since we have 2000 candidates, computing the exact expected minimum is expensive.
    #   We use a simpler but effective approximation:
    #   score_i = E[max(0, f_best - mu_i - sigma_i * Z)] where Z ~ N(0,1)
    #   This is equivalent to EI, but we can enhance it by considering the relative uncertainty.
    
    # A more principled KG approximation:
    # For each candidate i, the probability that it becomes the new best is:
    #   P(Y_i < f_best) = Phi((f_best - mu_i) / sigma_i)
    #   where Phi is the standard normal CDF.
    #   The expected reduction if it becomes the new best is:
    #   E[Y_i | Y_i < f_best] = mu_i - sigma_i * phi((f_best - mu_i)/sigma_i) / Phi((f_best - mu_i)/sigma_i)
    #   So the expected reduction is:
    #   E[max(0, f_best - Y_i)] = (f_best - mu_i) * Phi((f_best - mu_i)/sigma_i) + sigma_i * phi((f_best - mu_i)/sigma_i)
    
    # However, a simpler and often effective approach is to use a weighted combination of
    # exploitation (low mu) and exploration (high sigma), but with a focus on points
    # that are likely to beat f_best.
    
    # Let's compute the standard EI for minimization:
    # EI_i = E[max(0, f_best - Y_i)]
    #      = (f_best - mu_i) * Phi((f_best - mu_i)/sigma_i) + sigma_i * phi((f_best - mu_i)/sigma_i)
    #      when sigma_i > 0
    
    # For the KG, we want to prioritize points that are likely to significantly reduce f_best.
    # A common heuristic is to use EI but with a penalty for points where the expected
    # improvement is small relative to the uncertainty.
    
    # Let's compute EI:
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized distance: how many sigma below f_best is the mean?
    # If mu < f_best, then (f_best - mu) / sigma > 0, meaning the mean is below f_best.
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    from scipy.special import erf, ndtr, ndtri
    # ndtr is the CDF of the standard normal
    prob_improve = ndtr(z)
    
    # phi(z) = (1/sqrt(2*pi)) * exp(-z^2/2)
    phi_z = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * prob_improve + sigma_safe * phi_z
    
    # The KG can be approximated by EI, but we can add a term that rewards
    # points with high uncertainty that are close to f_best.
    # Let's use a weighted combination:
    # score = EI + alpha * sigma * prob_improve
    # where alpha controls the exploration-exploitation tradeoff.
    
    # Adaptive alpha: higher when f_best is not yet close to 0
    # Since the minimum value is 0, if f_best is large, we need more exploration.
    # If f_best is close to 0, we can focus on exploitation.
    if f_best > 1.0:
        alpha = 1.0
    elif f_best > 0.1:
        alpha = 0.5
    else:
        alpha = 0.2
    
    scores = ei + alpha * sigma_safe * prob_improve
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
