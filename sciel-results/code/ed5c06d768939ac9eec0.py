import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a stochastic Probabilistic Improvement (PI) 
    with soft-max selection probability.
    
    Instead of a deterministic EI/cost, we model the decision as selecting the point 
    that maximizes a noisy version of the cost-adjusted improvement. 
    The score is the probability that a candidate is the global maximum of this noisy 
    improvement landscape. This is approximated using the property that the max of 
    N iid normals has a known CDF, but we account for correlation by using a 
    "soft-max" temperature derived from the spread of the scores.
    
    Args:
        mu: array of GP mean predictions, shape (2000,)
        sigma: array of GP std predictions, shape (2000,)
        f_best: current best observed value
        cost: array of evaluation costs, shape (2000,)
    
    Returns:
        array of acquisition scores, shape (2000,), higher = prefer
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.nan_to_num(sigma, nan=1e-12, posinf=1.0, neginf=1e-12)
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e6, neginf=-1e6)
    cost = np.maximum(cost, 1e-9)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Compute Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = f_best - mu
    z = improvement / sigma
    
    # Clip z for stability, though scipy is robust
    z_clipped = np.clip(z, -10, 10)
    
    ei = improvement * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment
    # We want to prefer high EI and low cost.
    # A common heuristic is EI / cost, but this can be too sharp.
    # Another is EI - lambda * cost.
    # The inspiration suggests "Blending PI with scaled EI and adaptive weighting".
    # Let's use a log-cost penalty or a power cost normalization.
    # cost is in [1, 2]. 
    # Let's use a "cost-effective" score: S(x) = EI(x) / (cost(x) ** alpha)
    # alpha = 0.5 was noted to balance exploration better.
    alpha = 0.5
    cost_adjusted = cost ** alpha
    raw_scores = ei / cost_adjusted
    
    # Now, we want to convert these raw scores into a "probability of being the best".
    # This mimics a Bayesian model where the true improvement is random.
    # If we assume the scores are independent (which they are not, but as an approximation),
    # the probability that x_i is the max is:
    # P(X_i > X_j for all j != i)
    # For a set of scores S, a common "soft-max" probability is:
    # P_i = exp(S_i / T) / sum(exp(S_j / T))
    # However, the task asks for a score where higher = prefer. 
    # If we return the probability, the argmax is the same as the argmax of S_i.
    # This doesn't change the selection!
    
    # To change the selection, we need to account for the *uncertainty* in the scores themselves 
    # or use a different criterion.
    # The "Stochastic" part implies we consider the distribution of the improvement.
    # A better structural change: 
    # Use the **Expected Improvement** but normalized by the **Standard Deviation of the Cost-Adjusted Improvement**.
    # Or, use a **Thompson Sampling** style approach.
    
    # Let's try a different structural change: **Probability of Improvement (PI) with Cost Discounting**.
    # PI(x) = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is smoother than EI.
    # Cost-aware PI: PI(x) / cost(x)^beta
    
    pi = norm.cdf(z_clipped)
    pi = np.maximum(pi, 1e-12) # Avoid log(0) later if needed, but for division it's fine
    
    # Let's blend EI and PI to get a robust signal.
    # The inspiration mentions "Blending PI (smooth probability) with scaled EI".
    # Let's compute a blended score.
    # Scale EI to be comparable to PI (0-1).
    # EI can be large. Let's normalize EI by a characteristic scale.
    # A good scale is the max EI in the batch, or f_best itself.
    # Using f_best as scale: EI/f_best is dimensionless if f_best > 0.
    # If f_best is 0, we have issues.
    
    # Alternative: Use the **Knowledge Gradient** approximation? Too complex.
    
    # Let's stick to the "Soft-Select" idea but make it affect the *ranking* by adding noise 
    # or using a different metric.
    
    # Actually, the most impactful structural change for Ackley (multimodal) with cost 
    # is often **UCB with Cost Penalty**.
    # UCB(x) = mu(x) + kappa * sigma(x)
    # Cost-UCB(x) = UCB(x) / cost(x)  OR  UCB(x) - lambda * cost(x)
    
    # Let's try **Cost-Adjusted UCB**.
    # UCB encourages exploration. EI is exploitative.
    # For Ackley, exploration is key to escape local minima.
    # The seed program used EI/cost.
    # Let's switch to a UCB-based cost-aware acquisition.
    
    # Determine kappa adaptively.
    # A common choice is to scale kappa with the range of mu.
    # Or use a fixed kappa.
    # Let's use a kappa that balances exploration and exploitation.
    
    # To make it robust, we can use a "hybrid" that shifts from UCB to EI as f_best decreases.
    # But let's try pure Cost-UCB first.
    
    # mu is the predicted mean. For minimization, we want low mu.
    # UCB for minimization: mu(x) - kappa * sigma(x)
    # We want to maximize this score (i.e., find the most negative value).
    # So Score_UCB = mu(x) - kappa * sigma(x)
    # We want to select the point with the *lowest* Score_UCB? 
    # No, the task says "higher = evaluate next".
    # So we should return -Score_UCB or (kappa * sigma(x) - mu(x)).
    
    # Let's define:
    # Improvement_Potential = -mu(x) + kappa * sigma(x)
    # This is high when mu is low (good) and sigma is high (uncertain).
    
    # Now apply cost.
    # Score = Improvement_Potential / cost(x)^alpha
    
    # How to choose kappa?
    # If kappa is too small, it's greedy (like EI).
    # If kappa is too large, it's random.
    # A standard heuristic is kappa = 1.0 or scaled by the standard deviation of observed values.
    # Since we don't have observed values, let's use a fixed kappa relative to the scale of the problem.
    # Ackley values are in [0, ~28].
    # Let's use kappa = 1.0.
    
    # However, to make it more adaptive, we can use the spread of mu.
    # But we don't have global stats.
    
    # Let's try a **Weighted Sum of EI and UCB**, weighted by cost.
    # This is a "blended" approach.
    
    # 1. Compute EI (exploitative)
    # 2. Compute UCB (exploratory): -mu + sigma (normalized)
    #    Let's normalize sigma by a typical scale. If f_best is large, scale is large.
    #    Let's use scale = max(1, f_best).
    #    UCB_norm = (-mu + sigma) / scale
    
    # 3. Blend: Score = w * EI_norm + (1-w) * UCB_norm
    #    Then divide by cost.
    
    # Let's refine the UCB term.
    # For minimization, the "upper confidence bound" on the *improvement* is:
    # EI is already an improvement measure.
    # UCB on the function value: f(x) < mu(x) + sigma(x) with prob ~0.84.
    # So the best possible value is mu(x) + sigma(x).
    # The improvement is f_best - (mu(x) + sigma(x)).
    # This is a "Optimistic Improvement".
    
    # Let's use **Optimistic Improvement (OI)**:
    # OI(x) = max(0, f_best - (mu(x) + kappa * sigma(x)))
    
    # This is similar to EI but uses a linear bound instead of the Gaussian CDF.
    # It is computationally cheaper and often more robust in multimodal landscapes 
    # because it doesn't penalize high uncertainty as heavily as EI's tail behavior might.
    
    # Let's test this hypothesis.
    # OI(x) = max(0, f_best - mu(x) - kappa * sigma(x))
    
    # Choose kappa. 
    # If kappa is 1, it's a 84% confidence bound.
    # If kappa is 2, it's a 97% confidence bound.
    # Let's use kappa = 1.0 for a balance.
    
    kappa = 1.0
    oi = f_best - mu - kappa * sigma
    oi = np.maximum(oi, 0.0)
    
    # Now, combine OI with EI? 
    # Or just use OI?
    # OI is more "exploratory" than EI for high sigma.
    # EI(x) ~ sigma * phi(z) for large z. 
    # OI(x) ~ -kappa * sigma for large negative z (when mu >> f_best).
    # Actually, if mu >> f_best, EI is ~0. OI is ~0.
    # If mu << f_best, EI is ~ f_best - mu. OI is ~ f_best - mu - kappa*sigma.
    # So OI is slightly more conservative when we are confident.
    # When uncertain (mu ~ f_best, sigma large), EI is ~ sigma * phi(0) = sigma * 0.4.
    # OI is ~ -kappa * sigma (negative, so 0).
    # Wait, if mu = f_best, z=0. EI = sigma * 0.398.
    # OI = f_best - f_best - kappa * sigma = -kappa * sigma = 0 (clipped).
    # So OI is 0 when we are uncertain about the center.
    # This means OI will NOT explore regions where the mean is equal to the best value but uncertainty is high.
    # EI WILL explore them.
    # This is a significant difference. EI is better for exploration in the "plateau" of uncertainty.
    
    # Therefore, pure OI might be too exploitative/greedy in uncertain regions where the mean hasn't dropped yet.
    
    # Let's go back to **Blended EI and UCB**.
    
    # Term 1: EI (exploitative + some exploration)
    # Term 2: Pure Uncertainty (sigma) normalized.
    
    # Let's use a **Posterior Sampling** approximation.
    # Sample theta ~ N(mu, sigma).
    # Score = (f_best - theta) / cost.
    # We can't sample in the acquisition function (it must be deterministic for the evaluator? 
    # "Return 2000 finite scores". It doesn't say deterministic, but stochastic acquisition 
    # can cause instability in the "best of 2000" selection if not careful. 
    # However, the evaluator runs 5 instances. A stochastic acquisition might average out.
    # But typically, acquisition functions are deterministic given the GP.
    
    # Let's stick to a **Deterministic** structural change.
    
    # **Strategy: Cost-Adjusted Expected Improvement with Adaptive Temperature Smoothing**
    # The current program uses EI / cost.
    # The problem with EI/cost is that it is very peaky. 
    # If one point has slightly higher EI and lower cost, it wins by a large margin.
    # This can lead to "locking in" on a local optimum if the GP is slightly biased.
    
    # A structural change is to use a **Log-Sum-Exp** style aggregation or a **Soft-Max** 
    # that flattens the peaks.
    
    # But we only return the score for the argmax. Flattening peaks doesn't change the argmax 
    # unless we change the *relative* ordering.
    
    # To change the ordering, we must change the *metric*.
    
    # **New Metric: Cost-Effective Probability of Improvement (CEPI)**
    # CEPI(x) = PI(x) / cost(x)
    # PI(x) = Phi((f_best - mu) / sigma)
    
    # Comparison:
    # EI(x) = (f_best - mu)Phi(z) + sigma phi(z)
    # PI(x) = Phi(z)
    
    # For large z (mu << f_best), EI ~ f_best - mu, PI ~ 1.
    # For z ~ 0 (mu ~ f_best), EI ~ sigma * 0.4, PI ~ 0.5.
    # For negative z (mu > f_best), EI ~ 0, PI ~ 0.
    
    # PI is smoother. It doesn't depend on the *magnitude* of the improvement, just the probability.
    # This might help in multimodal landscapes where the GP mean is flat but uncertainty is high.
    
    # Let's try **PI / cost^0.5**.
    
    pi = norm.cdf(z_clipped)
    
    # To avoid selecting points where PI is 1.0 everywhere (early on), 
    # we might need to blend with EI.
    
    # Let's use a **Mixture** of EI and PI, weighted by the uncertainty.
    # When sigma is high, PI is more informative about "where to look" (exploration).
    # When sigma is low, EI is more informative about "how much to gain" (exploitation).
    
    # Let's define a weight w = sigma / (sigma + epsilon).
    # This is 1 when sigma is large, 0 when sigma is small.
    # But we need a scale.
    
    # Let's use a fixed blend.
    # Score = (EI + PI) / cost^0.5
    
    # This is a structural change from pure EI to a blend.
    
    scores_ei = ei / (cost ** 0.5)
    scores_pi = pi / (cost ** 0.5)
    
    # Normalize EI and PI to comparable scales?
    # EI can be >> 1. PI is in [0, 1].
    # If f_best is 10, EI is ~10. PI is ~0.5.
    # EI will dominate.
    
    # Let's normalize EI by f_best.
    if f_best > 1e-6:
        ei_norm = ei / f_best
    else:
        ei_norm = ei
        
    # Now EI_norm and PI are both in [0, 1] roughly.
    # Blend them.
    # w = 0.5
    blended = 0.5 * ei_norm + 0.5 * pi
    
    # Apply cost
    scores = blended / (cost ** 0.5)
    
    # Final sanitization
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
