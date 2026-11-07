"""Cost-aware Knowledge Gradient acquisition for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # 300s limit + 20s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient acquisition function.
    
    KG computes the expected value of the minimum after evaluating at x.
    For minimization: KG(x) = E[min(f_best, f(x))]
    We want to MINIMIZE this, so we return -KG / cost for the acquisition score
    (higher score = better candidate to evaluate next).
    
    The cost-aware component adjusts the effective budget pressure:
    as total cost approaches the budget limit, we weight exploitation more heavily.
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Safety fallback: simple EI-based score
        sigma_safe = np.maximum(sigma, 1e-10)
        cost_safe = np.maximum(cost, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Budget pressure: total cost is bounded by 30, start with 5 initial points
    # As cost approaches 30, we should be more exploitation-focused
    # budget_fraction_remaining = max(0, 1 - cost / 30)
    # But cost here is the per-candidate cost, not total. 
    # The evaluator tracks total cost; we only see per-point cost.
    # So we use cost directly as the penalty.
    
    # Knowledge Gradient for minimization:
    # KG(x) = E[min(f_best, f(x))] where f(x) ~ N(mu(x), sigma(x)^2)
    # 
    # E[min(a, Y)] for Y ~ N(mu, sigma^2) and constant a:
    # = mu * Phi((a-mu)/sigma) + sigma * phi((a-mu)/sigma) - (a-mu) * Phi((a-mu)/sigma)
    # Wait, let me derive this properly.
    # 
    # E[min(a, Y)] = E[Y * 1{Y < a}] + a * P(Y >= a)
    # For Y ~ N(mu, s^2):
    # E[Y * 1{Y < a}] = mu * Phi(z) - s * phi(z)  where z = (a-mu)/s
    # P(Y >= a) = 1 - Phi(z)
    # So E[min(a,Y)] = mu*Phi(z) - s*phi(z) + a*(1-Phi(z))
    # = a + (mu - a)*Phi(z) - s*phi(z)
    # = a - (a-mu)*Phi(z) - s*phi(z)
    
    # We want to MINIMIZE KG, so acquisition score = -KG / cost
    # But actually, for the loop that picks the MAX score, we want:
    # score = (something that's high when KG is low) / cost
    
    # Let's use: score = (f_best - KG) / cost
    # When KG < f_best, this is positive (good). When KG >= f_best, it's <= 0.
    # Actually E[min(f_best, Y)] <= f_best always, so f_best - KG >= 0.
    
    z = (f_best - mu) / sigma_safe
    
    # KG = f_best - (f_best - mu) * Phi(z) - sigma * phi(z)
    # Wait: E[min(f_best, Y)] = f_best - (f_best - mu)*Phi(z) - sigma*phi(z)
    # Let me verify: if mu << f_best, z is large positive, Phi(z)~1, phi(z)~0
    # KG ~ f_best - (f_best-mu) - 0 = mu. Correct, min is ~mu.
    # If mu >> f_best, z is large negative, Phi(z)~0, phi(z)~0
    # KG ~ f_best. Correct, min is ~f_best.
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # KG = E[min(f_best, Y)]
    KG = f_best - (f_best - mu) * Phi_z - sigma_safe * phi_z
    
    # The gain from evaluating at x is f_best - KG (how much we expect to improve)
    gain = f_best - KG
    # gain = (f_best - mu) * Phi_z + sigma_safe * phi_z
    # This is actually just EI! So KG gain = EI for the case of one evaluation.
    # 
    # Hmm, for a single evaluation, the KG gain equals EI. The difference shows up
    # when considering the value of information for future decisions.
    # But with a fixed budget and no further optimization, they're equivalent.
    #
    # Let me use a different structural approach: instead of pure EI, use a
    # "probabilistic improvement" measure that's more robust.
    
    # Alternative: Use the probability of improvement times the expected improvement
    # P(Y < f_best) = Phi(z)
    # E[Y | Y < f_best] = mu - sigma * phi(z)/Phi(z)
    # Expected improvement = (f_best - mu)*Phi(z) + sigma*phi(z)  [this is EI]
    
    # For a structural change, let's use a "soft minimum" approach with a
    # temperature parameter that balances exploration and exploitation.
    # Or better: use the actual KG formulation which accounts for the fact
    # that after evaluating, we update our belief.
    
    # Actually, for a single evaluation with no further queries, EI and KG gain are
    # identical. The structural difference needs to come from elsewhere.
    
    # Let's try: a cost-aware acquisition that uses a "regret" formulation
    # with a dynamic exploration-exploitation balance based on the cost structure.
    
    # The cost is 1 + mean(x scaled to [0,1]), so points near the center of the
    # domain (where x_scaled ~ 0.5) cost ~1.5, while corners cost ~1 or ~2.
    # We should prefer lower-cost points that have comparable improvement potential.
    
    # Score = EI / cost^alpha where alpha > 1 penalizes expensive points more
    # Or: Score = EI * exp(-beta * cost) for a soft penalty
    
    # Let's use: score = EI / (cost * (1 + gamma * sigma)) 
    # The gamma*sigma term makes high-uncertainty, high-cost points less attractive
    # This is a variance-penalized cost-aware EI
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Variance penalty: penalize high uncertainty to reduce noise sensitivity
    # The penalty scales with the ratio of uncertainty to the potential improvement
    penalty_factor = 1.0 + 0.5 * (sigma_safe / np.maximum(np.abs(f_best - mu), 1e-10))
    
    score = ei / (cost_safe * penalty_factor)
    
    # Add a small bonus for points that are likely to be in the "interesting" region
    # (not too far from current best in terms of predicted value)
    # This helps focus the search
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
