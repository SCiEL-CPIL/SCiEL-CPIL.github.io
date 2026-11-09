"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard: 330s wall-clock limit, use 320s as safety margin
_T0 = time.time()
_DEADLINE = _T0 + 320.0


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return simple POI/cost
    if time.time() > _DEADLINE:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        poi = norm.cdf(z)
        score = poi / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score

    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-8)

    # 1. Compute the Gram matrix K = sigma @ sigma.T
    # This represents the covariance between the 2000 candidate points under the GP prior.
    # Note: In a standard GP, we would use the kernel matrix K(x_i, x_j).
    # Here, we only have mu and sigma. However, for Thompson Sampling, we need to sample
    # from the joint posterior.
    # A common approximation when only marginals are available is to assume a specific
    # correlation structure. But wait, the task says "GP: Matern-5/2...".
    # The input mu and sigma are already the GP posterior mean and std.
    # To sample y ~ N(mu, Sigma), we need Sigma.
    # We do not have the full covariance matrix of the candidates.
    # However, we can approximate the joint distribution.
    # A simpler and very effective strategy for "different design family" is to use
    # a **Lower Confidence Bound (LCB)** with an adaptive beta, or a **Gaussian Process
    # based Thompson Sampling** approximation.
    
    # Actually, without the full covariance matrix between the 2000 points, we cannot
    # perform exact joint Thompson Sampling. 
    # But we can use a **single-sample Thompson Sampling** approximation where we assume
    # the candidates are independent (which is false, but often works surprisingly well 
    # for acquisition if we just want to pick one).
    # Wait, if we assume independence, we sample y_i ~ N(mu_i, sigma_i) and pick argmin y_i/cost_i.
    # This is a valid "different design family" (Monte Carlo sampling vs deterministic heuristic).
    
    # Let's try **Thompson Sampling with Independent Approximation**.
    # This explores regions with high sigma because the sample y will be more spread out.
    # It exploits regions with low mu because the sample y will be lower.
    
    # To make it more robust, we can use a **Gaussian Process based Thompson Sampling**
    # if we could compute the kernel. But we don't have x.
    # So we are stuck with marginals.
    
    # Alternative: **Knowledge Gradient (KG)** approximation.
    # KG(x) = E[ max(0, f_best - y(x)) - max(0, f_best - y*(x)) ] where y*(x) is the best of the rest.
    # This is hard to compute without joint distribution.
    
    # Let's stick to **Thompson Sampling (Independent Approximation)**.
    # It is computationally O(N) and structurally different from POI.
    
    # Generate random samples
    # To ensure reproducibility within the loop if needed, we could use a fixed seed, 
    # but random is fine for TS.
    # However, random TS can be noisy. 
    # Let's add a bit of structure: **Expected Improvement (EI)** is deterministic.
    # **Thompson Sampling** is stochastic.
    
    # Let's try a **Hybrid** that is structurally different: 
    # **Adaptive UCB** (Upper Confidence Bound) is for maximization. 
    # For minimization, it's **LCB** (Lower Confidence Bound).
    # LCB(x) = mu(x) - beta * sigma(x)
    # Score = -LCB(x) / cost(x)
    # This is deterministic. The current program is POI. 
    # LCB is a different family (optimistic bias vs probability of improvement).
    
    # But the directive says "Do not micro-tune constants". 
    # LCB requires beta. 
    # We can set beta adaptively based on the number of evaluations? 
    # We don't know the number of evaluations directly, but we have cost.
    # Total cost < 30. Initial cost ~ 5 * 1.5 = 7.5. 
    # So we have ~22.5 budget left. 
    # We can estimate the number of evaluations taken. 
    # But we don't have a counter. 
    # We can estimate "phase" from the spread of mu/sigma? 
    # Or just use a fixed beta that is "standard" like 1.0 or 2.0. 
    # Is that micro-tuning? 
    # The prompt says "Make a STRUCTURAL change". 
    # Switching from POI to LCB is structural. 
    # Switching to Thompson Sampling is structural.
    
    # Let's implement **Thompson Sampling** as it is a "different optimization or search method" 
    # (sampling-based vs heuristic-based).
    
    # To reduce variance of TS, we can use a **Gaussian Process based** approach if we had x. 
    # Since we don't, let's use a **Bayesian Optimization with Thompson Sampling** 
    # that samples from the posterior.
    
    # One issue: TS can pick a point with high cost if the sample is very good. 
    # We divide by cost.
    
    # Let's refine the TS approach:
    # 1. Sample y_i ~ N(mu_i, sigma_i) for all i.
    # 2. Score_i = -y_i / cost_i (since we want to minimize y, we maximize -y).
    #    Wait, the acquisition score should be "higher = evaluate next".
    #    If we sample y, we want the y that is likely to be the minimum.
    #    So we want to maximize the probability that y is the minimum?
    #    No, TS says: sample y, then pick argmin y.
    #    So the "score" for the loop is just the sample value? 
    #    No, the loop picks the argmax of the returned score.
    #    So we should return a score such that the best candidate has the highest score.
    #    If we sample y_i, the best candidate is argmin y_i.
    #    So we can return score_i = -y_i.
    #    But we must account for cost.
    #    score_i = -y_i / cost_i.
    
    # Let's try this.
    
    # To make it more robust, we can blend with POI? No, keep it pure.
    
    # One potential issue: If sigma is very small, y is close to mu.
    # If sigma is large, y is spread out.
    # This naturally balances exploration and exploitation.
    
    # Let's implement it.
    
    # Note: Using np.random.normal is fine.
    
    # However, pure TS can be unstable. 
    # Let's try **Expected Improvement (EI)** which is the deterministic counterpart.
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # This is a different family from POI (which is just Phi(z)).
    # EI accounts for the *magnitude* of the improvement, not just the probability.
    # This is a structural change: from probability of improvement to expected magnitude of improvement.
    # And it is cost-aware: EI / cost.
    
    # Let's compare POI vs EI.
    # POI = P(y < f_best).
    # EI = E[max(0, f_best - y)].
    # EI is generally considered superior to POI because it favors larger improvements.
    # The current program uses POI.
    # Switching to EI is a structural change in the acquisition function family.
    
    # Let's implement EI.
    
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    cdf = norm.cdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * phi
    
    # Ensure ei is non-negative (numerical errors can make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    score = ei / cost_safe
    
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
