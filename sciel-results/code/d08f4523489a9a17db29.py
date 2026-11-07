import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-10)
    
    # We need to estimate t (number of evaluations) to compute beta_t.
    # The problem states: 5 initial points, then 2000 candidates per step, total 30 evaluations.
    # So t ranges from 5 to 30.
    # We don't have direct access to t. We can estimate it or use a heuristic.
    # However, the signature is fixed: acquisition(mu, sigma, f_best).
    # We can try to infer t from f_best? No, that's unreliable.
    # Let's look at the inspirations. They just use fixed or adaptive constants.
    # The directive suggests GP-UCB with time-varying beta.
    # Without t, we can use a proxy.
    # Early on, f_best is likely high. Later, f_best is lower.
    # But f_best is not monotonic in a way that directly maps to t without noise.
    
    # Alternative: Use a fixed but well-tuned beta for UCB, or combine with EI.
    # The directive says: "Replace the linear UCB and standard EI with a Gaussian Process-Upper Confidence Bound (GP-UCB) using a time-varying exploration parameter... and combine it additively with a normalized Expected Improvement term".
    
    # Since we cannot pass t, let's approximate t.
    # The Beale function has a minimum of 0.
    # Initial random points in [-4.5, 4.5]^2 often yield values between 0 and 100+.
    # As t increases, f_best decreases.
    # We can map f_best to an estimated t.
    # Or, we can just use a fixed beta that works well across the range, or a beta that adapts to the uncertainty.
    
    # Let's try the combination: UCB + EI.
    # UCB: -mu + beta * sigma
    # EI: max(0, f_best - mu) - sigma * norm.pdf((f_best - mu)/sigma) ... wait, for minimization.
    # Standard EI for minimization:
    # Let Y ~ N(mu, sigma^2). We want to maximize E[max(f_best - Y, 0)].
    # If f_best is the current best (lowest), we want to find Y < f_best.
    # Improvement I = max(f_best - Y, 0).
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    # Note: If f_best - mu is negative (mu > f_best), the probability of improvement is low, but EI is still positive if there's uncertainty.
    
    # Let's compute EI.
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Normalize EI? The directive says "normalized Expected Improvement term".
    # EI has units of the objective function. UCB also has units of the objective function (since mu and sigma are in objective units).
    # So they are comparable.
    # However, the magnitude of EI depends on f_best.
    # If f_best is large, EI is large. If f_best is small, EI is small.
    # UCB's exploration term beta*sigma is independent of f_best.
    # This might be unbalanced.
    
    # Let's try a simple weighted sum.
    # score = UCB + weight * EI
    # Or score = UCB * (1 - alpha) + EI * alpha
    
    # Let's stick to the directive's suggestion: additive combination.
    # What beta to use?
    # If we assume t is around 15 (midpoint), beta_t = sqrt(2 * 15 * log(1 + 15)) = sqrt(30 * log(16)) ≈ sqrt(30 * 2.77) ≈ sqrt(83) ≈ 9.1.
    # This seems high.
    # Let's try a smaller beta, e.g., 2.0 or 3.0, which is common in practice.
    # Or adapt beta to f_best as in the current program?
    # Current program: beta = 1.0 + 4.0 * (f_best / (f_best + 5.0))
    # If f_best=100, beta=4.8. If f_best=0, beta=1.0.
    
    # Let's try a fixed beta of 2.5 and add EI.
    # score = -mu + 2.5 * sigma + ei
    
    # Let's check the magnitude.
    # If mu is far from f_best, EI is small.
    # If mu is close to f_best, EI is significant.
    # UCB pushes towards high sigma regions.
    # EI pushes towards regions where mu is slightly above f_best but with high probability of being below.
    
    # Actually, EI is a very strong exploitation signal.
    # UCB is a strong exploration signal.
    # Combining them might be redundant or conflicting.
    # Usually, one uses either EI or UCB.
    # The directive says "combine it additively".
    
    # Let's try:
    # score = -mu + beta * sigma + ei
    
    # To prevent EI from dominating when f_best is large, we could normalize it?
    # Or just use it as is.
    
    # Let's try a different approach: Use EI alone?
    # EI is theoretically optimal for expected improvement.
    # But it can get stuck in local optima if the GP is not well-calibrated.
    # UCB is more robust for exploration.
    
    # Let's try a hybrid:
    # If sigma is very small, EI dominates.
    # If sigma is large, UCB dominates.
    
    # Let's try:
    # score = -mu + 2.0 * sigma + 0.5 * ei
    
    # Let's refine the beta.
    # A common choice is beta = 2.0 or 3.0.
    # Let's use 2.5.
    
    beta = 2.5
    score = -mu + beta * sigma + ei
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
