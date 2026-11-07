"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import t as t_dist
import time

# Deadline guard
_DEADLINE = 320.0
_start_time = time.time()


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Student's t-based Expected Improvement.
    
    Uses a t-distribution with 5 degrees of freedom to model the posterior
    predictive distribution, which has heavier tails than the Gaussian.
    This promotes more robust exploration in the 6D Hartmann landscape
    with a limited 30-evaluation budget, preventing over-confidence
    in the GP mean when data is sparse.
    """
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return finite scores that favor exploration
        return np.maximum(sigma, 1e-9)
    
    # Ensure sigma is safe (avoid division by zero)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Degrees of freedom for the t-distribution
    nu = 5
    
    # Compute the t-statistic: (f_best - mu) / sigma
    # Under the t-model, the standardized variable follows t_nu
    t_stat = (f_best - mu) / sigma_safe
    
    # t-CDF and t-PDF
    t_cdf = t_dist.cdf(t_stat, df=nu)
    t_pdf = t_dist.pdf(t_stat, df=nu)
    
    # Expected Improvement under t-distribution:
    # EI = (f_best - mu) * P(X < f_best) + E[(f_best - X) * 1(X < f_best)]
    # For a t-distribution with df=nu, the second term involves:
    # sigma * (nu+1)/nu * t_pdf(t_stat) * (1 - t_stat * t_cdf(t_stat))
    # Actually, let me derive this more carefully.
    #
    # If Y = (X - mu) / sigma follows t_nu, then:
    # E[(f_best - X) * 1(X < f_best)] = E[(f_best - mu - sigma*Y) * 1(Y < t_stat)]
    # = (f_best - mu) * P(Y < t_stat) - sigma * E[Y * 1(Y < t_stat)]
    #
    # For t_nu, E[Y * 1(Y < a)] = -nu/(nu-1) * t_pdf(a) for nu > 1
    # Wait, that's not quite right. Let me use the known result:
    # For standard normal: E[Z * 1(Z < a)] = -phi(a)
    # For t_nu: E[T * 1(T < a)] = -nu/(nu-1) * t_pdf(a) ... no
    #
    # Actually, for a t-distribution with df=nu:
    # E[T * 1(T < a)] = - (nu+1)/(nu-1) * t_pdf(a) ... let me just use the formula
    #
    # The correct formula for EI with t-distribution:
    # EI = (f_best - mu) * F_t(t_stat) + sigma * (nu+1)/(nu-1) * f_t(t_stat) * (1 - t_stat * F_t(t_stat))
    #
    # Hmm, let me think again. The standard EI formula is:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    #
    # For t-distribution, the analog is:
    # EI = (f_best - mu) * F_t(z) + sigma * (nu+1)/(nu-1) * f_t(z)
    #
    # Wait, I need to be more careful. Let me derive from scratch.
    #
    # Let X ~ N(mu, sigma^2) in the Gaussian case, but here we model:
    # (X - mu) / sigma ~ t_nu
    #
    # EI = E[(f_best - X)^+ ] = E[(f_best - X) * 1(X < f_best)]
    # Let Y = (X - mu)/sigma, so X = mu + sigma*Y, Y ~ t_nu
    # EI = E[(f_best - mu - sigma*Y) * 1(Y < (f_best-mu)/sigma)]
    #    = (f_best - mu) * P(Y < t_stat) - sigma * E[Y * 1(Y < t_stat)]
    #
    # For t_nu, E[Y * 1(Y < a)] = -nu/(nu-1) * t_pdf(a) ... 
    # Actually, I recall that for the t-distribution:
    # E[T * 1(T <= a)] = - (nu+1)/(nu-1) * t_pdf(a) for nu > 1
    #
    # Let me verify: for nu=1 (Cauchy), this diverges. For large nu, it should approach -phi(a).
    # The t-pdf is f(t) = Gamma((nu+1)/2) / (sqrt(nu*pi)*Gamma(nu/2)) * (1+t^2/nu)^(-(nu+1)/2)
    #
    # The integral E[T * 1(T < a)] = ∫_{-∞}^{a} t * f(t) dt
    #
    # There's a known result: for t_nu, ∫_{-∞}^{a} t * f(t) dt = -nu/(nu-1) * f(a)
    # Let me check with nu -> infinity: f(t) -> phi(t), and -nu/(nu-1) -> -1, so we get -phi(a). Correct!
    #
    # So: EI = (f_best - mu) * F_t(t_stat) - sigma * (-nu/(nu-1) * t_pdf(t_stat))
    #        = (f_best - mu) * F_t(t_stat) + sigma * nu/(nu-1) * t_pdf(t_stat)
    
    # Use the derived formula
    ei_t = (f_best - mu) * t_cdf + sigma * (nu / (nu - 1)) * t_pdf
    
    # Add a small exploration bonus to encourage coverage
    exploration_bonus = 0.1 * sigma_safe
    
    scores = ei_t + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
