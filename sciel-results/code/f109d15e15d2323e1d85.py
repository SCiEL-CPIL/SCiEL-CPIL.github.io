"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a hybrid of Expected Improvement and
    a UCB-like exploration bonus, with a dynamic exploration coefficient that
    decays as the total cost increases.
    
    Strategy:
    1. Compute Expected Improvement (EI) for minimization.
    2. Compute a UCB-like term: -mu + beta * sigma, where beta is dynamic.
    3. Combine them: Score = (EI + gamma * UCB_term) / cost.
       Actually, a simpler and often more effective approach for this specific
       setup (2000 candidates, fixed GP) is to use a modified EI that includes
       a small exploration bonus directly in the improvement calculation, or
       simply use a UCB with a well-tuned beta that accounts for the cost.
       
       Let's try a different structural approach: **Probability of Improvement (PI)**
       combined with **Cost**. PI is more robust to outliers in sigma than EI,
       and when divided by cost, it favors cheap points that have a reasonable
       chance of being better. However, PI is very exploitative.
       
       Let's stick to EI but add a **Thompson Sampling-inspired noise** or a
       **UCB hybrid**.
       
       Alternative Structural Change: **Kernel Density Estimation of the cost landscape**?
       No, we don't have x.
       
       Let's try **Expected Simple Regret (ESR)**?
       ESR is similar to EI.
       
       How about **Batched EI**? We are selecting 1 point from 2000.
       
       Let's try **Upper Confidence Bound (UCB)** with a cost penalty.
       UCB(x) = -mu(x) + beta * sigma(x)
       Score = UCB(x) / cost(x)
       
       The current program uses EI/cost. EI is known to be more robust.
       However, EI can be too conservative.
       
       A structural change: **Adaptive Beta for UCB**.
       Start with a high beta (exploration) and decay it as the budget is consumed.
       We can estimate the "progress" from f_best. If f_best is improving, we might
       want to exploit more. But we don't have history.
       
       We have `cost` which is the cumulative cost? No, `cost` has shape (2000,).
       "cost has shape (2000,)". "Evaluating x costs 1 + mean(x scaled to [0, 1])."
       So `cost[i]` is the cost of evaluating candidate i.
       The total cost is tracked externally. We don't know the total cost inside `acquisition`
       unless we infer it? No, we just get `mu`, `sigma`, `f_best`, `cost`.
       
       Wait, the contract says: "While total cost is below 30...".
       We don't have access to the total cost history.
       
       However, we can look at the distribution of `mu` and `sigma`.
       
       Let's try a **Hybrid EI/UCB** with a fixed, slightly higher exploration factor than pure EI implies.
       Pure EI is equivalent to a specific UCB with a non-linear beta.
       
       Let's try **Probability of Improvement (PI)** with a small xi (optimism).
       PI(x, xi) = P(f(x) < f_best - xi)
       For minimization: Z = (f_best - xi - mu) / sigma
       PI = Phi(Z)
       Score = PI / cost
       
       PI is very sharp. It might be too exploitative.
       
       Let's try **Expected Improvement with a small noise term** added to sigma to encourage exploration?
       
       Actually, the most promising structural change from the inspirations is **Thompson Sampling**.
       But we only have mu and sigma. We can sample from N(mu, sigma^2) for each candidate?
       That would be 2000 samples.
       Score = -sample / cost.
       This is stochastic.
       
       Let's implement **Thompson Sampling** (TS) for the acquisition.
       For each candidate i:
       f_sample = mu[i] + sigma[i] * randn()
       score[i] = -f_sample / cost[i]
       
       This is a very different design family (Bayesian posterior sampling vs. deterministic heuristic).
       It naturally balances exploration and exploitation.
       It is also very fast to compute (just one random draw per candidate).
       
       However, TS can be too random. We can use a "temperature" or scale the sigma.
       
       Let's combine TS with a deterministic EI component to stabilize it?
       Score = alpha * EI/cost + (1-alpha) * (-mu + beta*sigma)/cost
       
       Or just pure TS.
       
       Let's try pure TS first. It's a distinct structural change.
       To make it cost-aware, we divide by cost.
       
       One issue: TS samples can be very large positive (bad for minimization) or very large negative.
       We want to maximize the score, so we minimize the sample.
       score = -sample / cost.
       
       Let's add a small epsilon to cost to avoid division by zero.
       
       We also need to ensure finite outputs.
       
       Let's implement this.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return np.zeros(n, dtype=np.float64)
    
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Thompson Sampling: sample from the posterior
    # We use a single random draw for all candidates to maintain consistency within the batch scoring
    # Actually, each candidate should have its own sample to reflect independent uncertainty.
    # But here, the GP posterior is fixed for the 2000 candidates.
    # Sampling independently for each is the standard TS approach.
    
    # Use a fixed seed for reproducibility in the evaluator? No, the evaluator calls this repeatedly.
    # We should use a random seed based on time or just use global random.
    # Using global random is fine.
    
    # Generate random samples
    # To avoid extreme outliers affecting the choice too much, we could clip the samples,
    # but let's try raw TS first.
    
    # Optimization: use standard normal and scale
    z_samples = np.random.standard_normal(n)
    f_samples = mu + sigma_safe * z_samples
    
    # We want to minimize f, so we maximize -f_samples
    # Cost-aware: divide by cost
    scores = -f_samples / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if time.time() > deadline:
        pass
        
    return scores
# EVOLVE-BLOCK-END
