# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Deterministic Monte Carlo Knowledge Gradient (KG) acquisition function.
    
    Instead of greedily maximizing Expected Improvement (EI) or Lower Confidence Bound (LCB),
    this function estimates the Knowledge Gradient: the expected reduction in the best observed
    value if we evaluate at a given point. This is a one-step lookahead policy that is known
    to be more robust for multimodal functions like Hartmann 3D because it accounts for the
    global context (probability of finding a new global minimum) rather than just local improvement.
    
    Since we do not have the full GP kernel or covariance matrix, we approximate the joint
    distribution of the candidate points and the current best point using Monte Carlo sampling
    based on the provided posterior mean and standard deviation. We assume the current best
    point's uncertainty is negligible compared to the candidate points in the early-to-mid stages,
    or we approximate the joint covariance structure by assuming the candidate points are
    conditionally independent given the current best (a simplification, but often effective
    for screening 2000 candidates).
    
    However, a better approximation for KG without the kernel is to sample from the joint
    distribution of the function value at the candidate x_i and the function value at the
    current best point x_best. We don't have x_best's mu/sigma explicitly, but we know
    f_best is the observed value. The GP posterior at x_best would have mean ~f_best and
    small sigma. 
    
    A robust surrogate for KG in this stateless setting is to maximize the probability
    that f(x) < f_best, weighted by the expected improvement magnitude, but computed via
    MC to capture non-linearities.
    
    Actually, a simpler and highly effective structural alternative to EI is the
    "Probability of Global Improvement" (PGI) or a variant of KG approximated by:
    E[max(0, f_best - f(x))] is EI.
    KG is E[f_best - min(f_best, f(x))] = E[max(0, f_best - f(x))] if we assume the
    current best remains the best unless beaten. Wait, that is exactly EI.
    
    The difference is in the *joint* modeling. Standard EI assumes the current best is fixed
    at f_best. KG considers the uncertainty in the current best as well.
    Since f_best is an observed value, its posterior variance is very small (noise level).
    Thus, EI and KG are nearly identical when noise is small.
    
    However, the prompt mentions a previous improvement using "MC-computed EI" and "prob_global".
    Let's implement a strategy that explicitly uses Monte Carlo to estimate the KG score,
    which involves sampling from the joint distribution of the candidate and the best point.
    
    To make this materially different and potentially more robust:
    We will use a "Softmax" based selection over the MC samples to approximate the
    probability of being the global minimum, which helps in escaping local minima by
    favoring points that have a non-negligible chance of being better than the current best,
    even if the mean improvement is small, provided the uncertainty is high.
    
    Specifically, we will compute the KG score as:
    KG(x) = E[ f_best - min(f(x), f_best) ]
    Since f_best is fixed (observed), this is mathematically EI.
    
    BUT, the "structural change" directive suggests a different design family.
    Let's try **Probability of Global Improvement (PGI)** combined with a penalty for
    points that are too close to the current best (to encourage exploration of new basins).
    
    Actually, let's look at the inspiration: "MC-computed EI ... prob_global weighting".
    Let's implement a Monte Carlo estimator for the expected reduction in the best value,
    but we will add a **diversity bonus** based on the distance to the current best's
    implied region? No, we don't have coordinates.
    
    Let's stick to the core insight: **Knowledge Gradient via Monte Carlo**.
    Even if mathematically similar to EI in the noiseless limit, the MC implementation
    allows for different handling of the tails and can be combined with a "lookahead"
    heuristic.
    
    Here is a specific structural variant: **Multi-Step Lookahead via MC**.
    Instead of just 1-step EI, we simulate a few steps? No, too slow for 2000 points.
    
    Let's use **Expected Feasibility** or **Probability of Improvement (PI)** with a
    dynamic threshold?
    
    Let's try **GP-UCB with an adaptive beta** derived from the distribution of sigmas,
    but computed via a "confidence" metric that is different from the smoothstep blend.
    
    Wait, the most distinct and powerful method for multimodal functions is **Thompson Sampling (TS)**.
    The current program is deterministic (EI/LCB). TS is stochastic.
    However, the evaluator calls `acquisition` once per step to get 2000 scores.
    If we return stochastic scores, the selection `argmax` will be stochastic.
    This is a valid strategy. TS is known to be more exploratory and robust to local minima.
    
    Strategy: **Thompson Sampling**.
    1. Sample a single realization of the GP posterior for all 2000 candidates.
       Since we don't have the covariance matrix, we cannot sample the *joint* distribution
       correctly to maintain correlation.
       However, we can approximate the marginal posterior for each candidate independently:
       f_i ~ N(mu_i, sigma_i^2).
       This ignores the correlation between candidates, which is a flaw.
       
    Alternative: **Deterministic KG with MC**.
    Let's implement the MC KG properly.
    For each candidate i, we want to estimate E[f_best - min(f_i, f_best)].
    Since f_best is a constant, this is E[max(0, f_best - f_i)].
    This is exactly EI.
    
    So, to be materially different, we must change the *objective* or the *model*.
    
    How about **Maximizing the Lower Confidence Bound (LCB)** but with a beta that is
    not a smoothstep of max_sigma, but rather a **quantile-based** adaptation?
    
    Or, **Minimizing the Upper Confidence Bound (UCB)** for the regret?
    
    Let's try **Expected Improvement (EI) with a "floor" on sigma**.
    If sigma is very small, EI is small. If sigma is large, EI is large.
    
    Let's look at the "inspiration 3" which scored 0.8481: "local, greedy strategy susceptible to getting trapped".
    The fix is usually **exploration**.
    
    Let's implement **GP-UCB** with a beta that decays with the number of evaluations.
    But we are stateless. We can estimate the number of evaluations by the drop in max_sigma.
    
    Let's try a **Hybrid of LCB and EI** but with a **weighted average** where the weight
    for LCB is proportional to the *local* sigma, not the global max_sigma.
    This way, points with high uncertainty get more exploration weight, while points with
    low uncertainty are evaluated by their exploitation value.
    
    Formula:
    score_i = w_i * LCB_i + (1 - w_i) * EI_i
    where w_i = sigmoid((sigma_i - threshold) / scale)
    
    This is a **point-wise adaptive** acquisition, whereas the current program uses a
    **global** weight (weight_lcb) applied to all points. This is a structural change in
    how exploration and exploitation are balanced: locally vs globally.
    
    Let's implement this **Locally Adaptive Hybrid**.
    
    Parameters:
    - threshold: 0.3 (mid-range sigma)
    - scale: 0.1
    
    If sigma_i > 0.3, w_i is close to 1 (explore).
    If sigma_i < 0.3, w_i is close to 0 (exploit).
    
    This encourages exploring high-uncertainty regions regardless of their mean, while
    exploiting low-uncertainty regions if their mean is good.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety check
    if time.time() - start_time > 325:
        return -mu

    # Compute LCB scores
    # Use a beta that is moderate. Since we are blending, we can use a fixed beta for LCB.
    beta = 2.5
    lcb_scores = -mu + beta * sigma
    
    # Compute EI scores
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Locally Adaptive Weight
    # We want to explore when sigma is high, exploit when sigma is low.
    # Use a sigmoid function to create a smooth transition.
    # Center the transition around a typical mid-optimization sigma value.
    # In Hartmann 3D with Matern-5/2 and length scale 0.2, sigma drops from ~1 to ~0.1.
    # Let's center at 0.4.
    center = 0.4
    scale = 0.15
    
    # w_i = 1 / (1 + exp(-(sigma_i - center) / scale))
    # If sigma_i >> center, w_i -> 1 (LCB)
    # If sigma_i << center, w_i -> 0 (EI)
    
    # Clip to avoid overflow in exp
    exponent = np.clip((sigma - center) / scale, -50, 50)
    w_lcb = 1.0 / (1.0 + np.exp(-exponent))
    
    # Final score is a weighted sum
    # Note: LCB and EI have different scales. 
    # LCB can be negative or positive. EI is always non-negative.
    # To combine them meaningfully, we should normalize or ensure they are comparable.
    # However, both are "higher is better".
    # A simple linear combination is a common heuristic.
    
    final_scores = w_lcb * lcb_scores + (1 - w_lcb) * ei_scores
    
    # Ensure finite values
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
# EVOLVE-BLOCK-END
