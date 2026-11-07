"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a
    noise-free lower-bound term to encourage exploitation of very low mu.
    """
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)

    # Standardized improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Lower confidence bound (LCB) component: favors points with low mu
    # and moderate sigma. We use a small beta to not overwhelm EI.
    lcb = mu - 0.05 * sigma_safe

    # Combine: primary is EI, but add a small bonus for low LCB
    # This helps break ties or guide toward the minimum when EI is flat.
    # We scale EI to be comparable to LCB terms if needed, but here
    # EI is the main driver. The LCB term is a secondary nudge.
    # Actually, for minimization, we want high score.
    # EI is higher when mu is low and sigma is low (exploit) or sigma is high (explore).
    # LCB is lower when mu is low. We want to MAXIMIZE score.
    # So we can add a term that rewards low mu.
    # score = EI + gamma * ( - LCB ) ? No, LCB = mu - beta*sigma.
    # If we maximize EI, we already get low mu.
    # Let's try a pure EI but with a slightly modified sigma floor or
    # a hybrid that penalizes high sigma when mu is already very low.
    
    # Alternative: Use EI but ensure numerical stability and maybe
    # a slight preference for exploitation when close to optimum.
    
    # The current best score is 1.0. Let's try a pure EI with a very small
    # noise floor to avoid division by zero issues, which is standard.
    # However, the directive asks for a STRUCTURAL change.
    # Let's try Upper Confidence Bound (UCB) for minimization, which is
    # equivalent to minimizing LCB.
    # UCB score = -LCB = -(mu - beta*sigma) = -mu + beta*sigma
    # This is a linear function of mu and sigma.
    # EI is nonlinear.
    # Often EI is better for BO.
    
    # Let's try a weighted sum of EI and a "probability of improvement" (PI)
    # or just stick to EI but with a different beta parameter if it were LCB.
    # Since we only have mu and sigma, let's try a different acquisition:
    # Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is purely exploratory/exploitative based on probability.
    
    # Let's try a mix: 0.9 * EI + 0.1 * PI
    # Or just EI.
    
    # Given the stagnation, let's try a different structural approach:
    # Use the "Expected Regret" or just a pure LCB (UCB for minimization)
    # with a tuned beta.
    
    # Let's try: Score = -mu + beta * sigma
    # This is UCB for minimization.
    # If beta is too large, it explores too much. If too small, it exploits.
    # Let's try beta = 0.1.
    
    # Wait, the task says "higher meaning evaluate next".
    # For minimization of sphere function (min at 0):
    # If we use UCB = -mu + beta*sigma:
    # Low mu (good) -> high score (because -mu is high).
    # High sigma (uncertain) -> high score (because beta*sigma is high).
    
    # Let's compare EI vs UCB.
    # EI is generally superior.
    # Maybe the issue is that EI gets stuck if sigma is very small and mu is not 0.
    # With UCB, if sigma is small, the score is dominated by -mu.
    
    # Let's try a hybrid:
    # If mu is very close to f_best (within sigma), EI is good.
    # If mu is far, EI is small.
    
    # Let's try a different acquisition: Knowledge Gradient? No, too complex.
    
    # Let's try "Expected Improvement" but with a slightly larger sigma floor
    # or a different combination.
    
    # Actually, let's look at the "STAGNATION" hint.
    # "Make a STRUCTURAL change: a different design or modelling family..."
    # We are only writing the acquisition function. We cannot change the GP.
    # So we must change the acquisition strategy.
    
    # Let's try LCB (Lower Confidence Bound) for minimization.
    # We want to minimize f. We pick the point that minimizes LCB = mu - beta*sigma.
    # The score should be HIGHER for better points.
    # So Score = -LCB = -mu + beta*sigma.
    
    # Let's try this with beta = 0.2.
    
    # Also, consider that f_best is the current minimum.
    # If we use UCB, we might explore points that are far from the current best
    # if their sigma is high.
    
    # Let's implement UCB with beta=0.2.
    
    beta = 0.2
    score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
