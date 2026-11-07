"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # GP-UCB with time-varying beta for minimization.
    # Since we are minimizing, we want to find low values.
    # UCB for maximization: mu + sqrt(beta) * sigma
    # For minimization (lower is better), we can maximize -mu + sqrt(beta) * sigma
    # Or equivalently, we want high scores for points that are likely to be low (good)
    # and have high uncertainty.
    # So score = -mu + sqrt(beta) * sigma
    
    # We need to estimate t (number of evaluations so far).
    # The problem states: "starts from 5 uniform random points, then at each step... until 30 evaluations"
    # So t ranges from 5 to 30.
    # We don't have explicit t passed in. We can infer t from the length of the data?
    # No, mu and sigma have shape (2000,), which are the candidates.
    # We don't know the current step count directly.
    # However, the prompt says "f_best is the lowest observed value".
    # We can try to infer the stage based on how good f_best is? No, that's unreliable.
    
    # Alternative: Use a fixed beta or a beta that doesn't depend on t explicitly,
    # but the directive suggests time-varying.
    # Since we can't pass t, maybe we can use a heuristic.
    # Or perhaps we can just use a fixed beta that is reasonably high for exploration.
    # The standard EI (score 1.0) is effectively a specific balance.
    # GP-UCB with beta ~ sqrt(log(t)) is standard.
    # Let's assume t is roughly the number of evaluations.
    # Can we estimate t? No.
    
    # Let's look at the "INSPIRATIONS" and "LESSONS".
    # "Replacing the fixed 0.7/0.3 EI/UCB blend with an adaptive beta UCB component..."
    # This suggests previous attempts tried blending.
    
    # Since we cannot know t, let's use a fixed beta that is known to work well for small n.
    # For n=5..30, log(t) is small.
    # sqrt(log(30)) ~ sqrt(3.4) ~ 1.85
    # sqrt(log(5)) ~ sqrt(1.6) ~ 1.26
    # A common choice is beta = 2 * sqrt(log(t)).
    # Let's try a fixed beta = 2.0 or 3.0?
    # Actually, EI is often better than UCB with constant beta for small budgets.
    # But the directive says "GP-UCB ... theoretically superior ... for minimizing regret in multimodal landscapes".
    
    # Let's try to infer t from the variance of sigma?
    # Early on, sigma is high everywhere. Later, sigma is low near evaluated points.
    # Maybe use the mean sigma as a proxy for exploration?
    
    # Let's stick to the directive: "Use a Gaussian Process Upper Confidence Bound (GP-UCB) acquisition function with a time-varying, decreasing exploration parameter beta_t"
    # Wait, "decreasing"? Usually beta_t increases to allow more exploration as uncertainty decreases?
    # No, standard GP-UCB uses beta_t = sqrt(log(t)). It increases slowly.
    # The directive says "decreasing exploration parameter"? That might be a typo in my interpretation or the prompt.
    # "systematically shift from broad exploration ... to exploitation"
    # If beta decreases, we exploit more.
    # If we start with high beta and decrease it, we explore first.
    # But we don't know t.
    
    # Let's assume the prompt implies we should use a strategy that adapts.
    # Since we can't track t, let's use a fixed beta that balances exploration/exploitation well.
    # A value of beta=2.0 is a common starting point.
    
    # Let's try: score = -mu + beta * sigma
    # With beta = 2.0
    
    # To be safe and potentially improve over EI, let's use a slightly higher beta to encourage exploration of the multimodal landscape.
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
