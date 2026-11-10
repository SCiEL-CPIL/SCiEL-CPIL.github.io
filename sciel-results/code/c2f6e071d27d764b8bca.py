"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats as stats

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Multi-Start Local Refinement Acquisition.
    
    Instead of just picking the single best EI point, this strategy:
    1. Identifies multiple non-redundant high-potential regions (basins) using EI.
    2. Estimates the potential improvement in each basin via a local search simulation.
    3. Returns a score for the candidate that maximizes the expected improvement 
       achievable within the remaining budget, effectively steering the global 
       search towards regions where local refinement is most cost-effective.
    
    Since we only return a single vector of scores to guide the *next* global 
    evaluation, we use the scores to highlight the "entry point" of the best 
    basin. However, to truly leverage local refinement, we would need to modify 
    the loop. Since we cannot modify the loop, we approximate by boosting the 
    EI score of points that are in "unexplored" high-value regions relative to 
    their cost, while penalizing points that are close to already evaluated 
    good points (if we had them, which we don't directly, but we can infer 
    from sigma).
    
    Actually, a simpler structural change that fits the interface:
    Use a **Cost-Adjusted Expected Improvement with a Dynamic Temperature**.
    The temperature controls the exploration/exploitation trade-off.
    As the budget is consumed (inferred from f_best improvement rate or just 
    a heuristic decay), we shift from exploration to exploitation.
    
    Wait, the directive says "Different design or modelling family".
    Let's try **Knowledge Gradient (KG)** approximation or **Probability of Improvement (PI)** 
    with a cost-aware twist.
    
    Let's implement a **Cost-Aware Probability of Improvement (PAI)**.
    PI = Phi((f_best - mu) / sigma)
    PAI = PI / cost
    
    This is simpler than EI and often more robust to noise, but we have low noise.
    EI is generally better.
    
    Let's try **Expected Improvement with a Cost Penalty that scales with the 
    "distance" from the current best in the input space**.
    But we don't have input coordinates.
    
    Let's look at the "Cross" operator hint. It suggests that combining features 
    from different parents (or different regions) is beneficial.
    In the context of a single acquisition function, this might imply that 
    we should encourage **diversity** in the search.
    
    A structural change: **Batch Acquisition via Greedy Selection Simulation**.
    We simulate selecting K next points (e.g., K=3) from the 2000 candidates 
    using a greedy EI strategy, estimating the total improvement.
    Then, we score each candidate based on how often it appears in the top 
    K selections across different random restarts or perturbations.
    
    However, K=3 simulation inside acquisition() for 2000 points might be slow 
    if done naively. But we only have 2000 points.
    
    Let's try a simpler but structurally different approach:
    **Variance-Weighted Cost-Adjusted LCB**.
    LCB = mu - beta * sigma
    Score = LCB / cost
    
    The current program uses EI / cost.
    LCB is more aggressive in exploration.
    
    Let's try **Thompson Sampling with Cost**.
    Sample from posterior: y ~ N(mu, sigma)
    Score = -y / cost  (since we minimize, lower y is better, so higher score is better)
    Actually, we want to maximize the probability of finding a low value.
    TS is stochastic, which might be bad for reproducibility if not seeded.
    
    Let's stick to a deterministic but different formulation:
    **Expected Improvement per Unit of Cost, but with a "Discount Factor" for 
    points that are likely to be "wasted" due to high cost.**
    
    The current program uses: EI / (cost * (1 + cost)).
    
    Let's try: **EI / cost** but with a **dynamic beta** in the EI calculation?
    No, EI is standard.
    
    How about **Maximin LCB**?
    We don't have distances.
    
    Let's implement **Cost-Aware Expected Improvement with a Soft Threshold**.
    Instead of standard EI, use a "soft" EI that only counts improvement 
    if it is significant relative to sigma.
    
    Actually, the most distinct structural change available without coordinates 
    is to change from **Pointwise Acquisition** to **Set-based Acquisition** 
    (even if we only return one score per point, we can compute the score 
    based on its marginal contribution to a hypothetical set).
    
    Approximation:
    1. Sort candidates by EI/cost.
    2. The top candidate gets the highest score.
    3. Subsequent candidates get penalized if they are "redundant".
    But we don't know redundancy without coordinates.
    
    Let's try a different family: **Bayesian Optimization with Uncertainty (BOU)** 
    or **Confidence-Weighted Acquisition**.
    
    Let's try **LCB with a cost-dependent beta**.
    beta = alpha / log(1 + cost)
    Score = (mu - beta * sigma) / cost
    
    If cost is high, beta is smaller, so we exploit more.
    If cost is low, beta is larger, so we explore more.
    This is a structural inversion of the usual intuition (usually high cost -> explore less, 
    but here high cost -> exploit more because we can't afford to waste it on exploration 
    unless the potential is huge).
    
    Let's implement this:
    1. Compute LCB = mu - beta * sigma.
    2. beta = 2.0 / (1.0 + cost)  # High cost -> small beta -> exploitation. Low cost -> large beta -> exploration.
    3. Score = LCB / cost.
    
    This is materially different from EI / cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Dynamic Beta for LCB
    # We want to explore more when cost is low, exploit when cost is high.
    # Base beta around 2.0 (typical for UCB/LCB)
    # Scale beta inversely with cost.
    # If cost is 1 (min), beta = 2.0 / (1+1) = 1.0? 
    # If cost is 2 (max, since cost = 1 + mean(x_scaled) in [0,1] -> cost in [1,2]), 
    # beta = 2.0 / (1+2) = 0.66.
    
    # Let's adjust:
    # cost_min = 1, cost_max = 2.
    # We want beta to be high for cost=1, low for cost=2.
    # beta = base_beta * (cost_max / cost) ?
    # beta = 2.0 * (2.0 / cost) = 4.0 / cost.
    # cost=1 -> beta=4.0 (High exploration)
    # cost=2 -> beta=2.0 (Lower exploration)
    
    # Or simply:
    beta = 3.0 / cost
    
    # Compute LCB
    lcb = mu - beta * sigma
    
    # Normalize LCB to be positive? No, LCB can be negative.
    # We want to maximize LCB / cost.
    
    scores = lcb / cost
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
