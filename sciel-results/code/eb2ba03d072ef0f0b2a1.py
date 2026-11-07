import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
        # Calculate Expected Improvement (which is equivalent to KG for 1 step)
        # mu, sigma, f_best are arrays/scalars
        # Ensure sigma is positive
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # z-score
        z = (f_best - mu) / sigma_safe
        
        # EI formula
        # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
        # For minimization, improvement is f_best - y.
        # y ~ N(mu, sigma^2)
        # E[max(0, f_best - y)]
        
        # Handle numerical stability for large z
        # If z > 8, cdf ~ 1, pdf ~ 0
        # If z < -8, cdf ~ 0, pdf ~ 0
        
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        
        ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
        
        # Cost-awareness
        # cost is the cost of evaluating this point
        # We want to maximize EI per unit cost
        
        # Adaptive exploration
        # Total budget is 30.
        # cost is the cumulative cost so far.
        # Remaining budget R = 30 - cost.
        # Note: 'cost' input is described as "cost" in signature.
        # "mu, sigma and cost have shape (2000,)."
        # Wait. "mu, sigma and cost have shape (2000,)."
        # Does 'cost' here mean the cost of *each candidate* or the *total cost so far*?
        
        # Task description: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
        # "mu, sigma and cost have shape (2000,)."
        # This implies `cost` is an array of the *individual costs* of the 2000 candidates.
        # It does NOT pass the total accumulated cost.
        
        # Re-reading: "Each function runs... While total cost is below 30... f_best is the lowest observed value."
        # "mu, sigma and cost have shape (2000,). Return 2000 finite scores..."
        # "No coordinates are passed."
        
        # So `cost` is the cost of evaluating *that specific candidate x*.
        # We do NOT know the total accumulated cost or the remaining budget.
        # We only know `f_best`.
        
        # So we cannot use a time-dependent alpha based on remaining budget.
        # We must rely on `f_best` and the distribution of `mu, sigma`.
        
        # However, we can infer the stage of the search from `f_best`.
        # If `f_best` is close to 0, we are in late stage (exploitation).
        # If `f_best` is high, we are in early stage (exploration).
        
        # Ackley minimum is 0.
        # Initial f_best might be ~5-10.
        # Late f_best might be ~1e-3.
        
        # We can define an exploration weight based on f_best.
        # alpha = max(0, 1 - f_best / f_initial_estimate)
        
        # But we don't know f_initial_estimate exactly.
        # Let's assume a threshold.
        # If f_best < 0.1, we are close to optimum. Reduce exploration.
        # If f_best > 1.0, we are far. Increase exploration.
        
        # Let's try:
        # Score = (EI + beta * sigma) / Cost
        # beta = 0.5 * exp(-f_best) ?
        
        # Or simpler:
        # Just use EI / Cost, but with a better numerical implementation?
        # The seed is already simple.
        
        # What if we use **Probability of Improvement**?
        # PI = cdf_z
        # Score = PI / Cost
        
        # Let's compare EI vs PI.
        # EI emphasizes *magnitude* of improvement.
        # PI emphasizes *likelihood* of improvement.
        # In the end game, we want magnitude.
        # In the start game, we want likelihood (to find a good region).
        
        # Let's try a blend:
        # Score = (EI + 0.1 * PI) / Cost ? No, units differ.
        
        # Let's stick to the "KG" directive.
        # If KG == EI, maybe the "improvement" is just to ensure we are using the *exact* KG formula which might handle the "min" operation differently in a multi-point context? No, it's single point.
        
        # Let's look at the "REFINEMENT" hint again: "smooth surrogate for a nonsmooth min/max".
        # The min in EI is nonsmooth at y = f_best.
        # We can use a soft-min.
        # min(a, b) approx a + b - soft_max(a, b)
        # soft_max(a, b) = log(exp(a/T) + exp(b/T)) * T
        # This smooths the kink.
        
        # However, the analytic solution for EI is exact for the hard min.
        # Smoothing it would make it an approximation.
        
        # Let's try **Expected Improvement with a noise term** (EI-noise).
        # If there is observation noise, the "true" function value is $y - \epsilon$.
        # But noise is 1e-6.
        
        # Let's try **UCB** with a tuned beta.
        # Score = (-mu + beta * sigma) / Cost
        # The inspiration said UCB -> 0.23. So UCB alone is bad.
        
        # What if we combine EI and UCB?
        # Score = (EI + 0.5 * sigma) / Cost
        
        # Let's try this.
        
        # Also, ensure `cost` is not zero.
        
        # One more idea: **Variance-aware EI**.
        # If sigma is very high, EI is high, but the risk is also high.
        
        # Let's go with the **Adaptive Exploration** based on `f_best`.
        # If f_best is large, we are lost. Explore more.
        # If f_best is small, we are close. Exploit more.
        
        # beta = 1.0 if f_best > 1.0 else 0.0
        # Smooth transition:
        # beta = 1.0 / (1.0 + f_best)
        
        # Score = (EI + beta * sigma) / Cost
        
        # Let's implement this.
        
        beta = 1.0 / (1.0 + f_best)
        
        # To avoid dominating EI, scale beta down.
        # EI is in objective units. sigma is in objective units.
        # So they are comparable.
        
        # Let's try beta = 0.5 / (1.0 + f_best)
        
        scores = (ei + 0.5 * beta * sigma_safe) / cost
        
        return scores
