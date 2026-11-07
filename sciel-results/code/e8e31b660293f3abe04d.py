import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Hybrid acquisition: max(PI, EI)
    # PI favors high probability of improvement (exploration of uncertainty)
    # EI favors high expected improvement (exploitation of promising areas)
    # Taking the max ensures that points with high uncertainty (high PI) are considered
    # even if their mean is not sufficiently low to yield high EI.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Probability of Improvement (PI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Expected Improvement (EI)
    # E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Normalize EI to be comparable to PI?
    # PI is in [0, 1]. EI is in [0, infinity].
    # For a sphere function, f_best starts around 25 and goes to 0.
    # Early on, EI can be large. PI is bounded.
    # If we just take max, EI will dominate early on because EI > 1 easily.
    # However, the prompt says "favors points with a high likelihood of surpassing... providing a distinct exploration bias".
    # Pure PI is often too exploratory. Pure EI is too exploitative.
    # A common heuristic is to use PI when sigma is large, or blend them.
    # But the directive specifically asks for PI strategy to test a structural change.
    # Let's look at the "Inspiration" note: "PI ... favors points with a high likelihood ... distinct exploration bias compared to EI".
    # If I use pure PI, it might score worse than EI (which scored 1.0).
    # The observed change data shows a child with score 1.0 from a parent with 0.43.
    # The current program is EI.
    # The directive says: "Strategy to try in this change: Use a Probability of Improvement (PI) acquisition function...".
    # So I should implement PI.
    # But wait, the "Knowledge from this run" says: "Switching from PI to EI ... is typically more effective".
    # This suggests PI might be worse.
    # However, the directive is explicit: "Strategy to try in this change: Use a Probability of Improvement (PI)..."
    # I will implement PI as requested by the directive, as it is the specified structural change.
    # To make it slightly more robust, I could add a small exploration bonus or use the max, but the directive is specific about PI.
    # Let's stick to pure PI to clearly test the "PI vs EI" structural difference.
    
    # Ensure finite scores
    pi = np.where(np.isfinite(pi), pi, 0.0)
    
    return pi
