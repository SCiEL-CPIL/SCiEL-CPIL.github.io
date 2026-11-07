"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We use a stateless approach that estimates the "search phase" from f_best.
# For Ackley 2D, the optimum is 0.
# Early phase: f_best is large (e.g., > 2.0). We should explore to find a promising basin.
# Late phase: f_best is small (e.g., < 0.1). We should exploit to refine the minimum.

def acquisition(mu, sigma, f_best):
    # Safety guards
    if len(mu) != len(sigma):
        return np.zeros(len(mu))
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    # In minimization, we want low mu.
    # z = (f_best - mu) / sigma
    # If z is high, mu is low relative to f_best -> likely improvement.
    z = (f_best - mu) / sigma_safe
    
    # Core acquisition functions
    # 1. Probability of Improvement (PI)
    # P(y < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # 2. Expected Improvement (EI)
    # E[max(0, f_best - y)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 3. Upper Confidence Bound (UCB) for minimization
    # We want to maximize -mu + beta*sigma
    # Standard UCB for minimization often uses -mu + beta*sigma
    ucb_beta = 2.5
    ucb = -mu + ucb_beta * sigma_safe
    
    # 4. Hypervolume Improvement (HVI) approximation? 
    # For 1D output, HVI is similar to EI but with a reference point.
    # Let's stick to PI, EI, UCB.
    
    # Strategy:
    # If f_best is high, we haven't found a good region. 
    # The GP is likely uncertain everywhere or has broad peaks.
    # UCB is good for exploring.
    # However, if we just pick the max UCB, we might ignore areas where mu is already decent.
    # 
    # Let's look at the distribution of scores.
    # A robust approach in short-horizon BO is to use a "racing" or "tournament" selection 
    # or simply a weighted combination that adapts.
    
    # But the simplest structural change from the current "static beta UCB" is 
    # to use a **Gaussian Process-based Local Improvement** heuristic embedded in the scoring.
    # Since we can't do local optimization, we can use the posterior to guide a "virtual" local search.
    
    # Alternative Structural Change: **Multi-Objective style weighting**
    # Let's try a dynamic blend of EI and UCB.
    
    # Normalize EI and UCB to [0, 1] range to combine them fairly?
    # EI is in objective units. UCB is in objective units (roughly).
    # They are comparable if mu and sigma are in objective units.
    
    # Let's try: Score = EI + lambda * UCB
    # Where lambda decreases as we get closer to 0.
    
    # Heuristic for lambda:
    # If f_best > 1.0: lambda = 1.0 (High exploration)
    # If f_best < 0.1: lambda = 0.2 (High exploitation)
    # Linear interpolation in log space? Or just simple linear.
    
    # Let's map f_best to a "confidence" level.
    # Ackley values drop rapidly.
    # f_best > 5: very uncertain.
    # f_best < 0.01: very certain.
    
    # Use a smooth transition.
    # t = sigmoid(k * (f_best - center))
    # We want high weight on UCB when f_best is high.
    
    # Let's try a different structural approach: **Thompson Sampling**
    # Draw samples from the posterior and pick the argmin?
    # But we need to return scores for ALL 2000 points to let the evaluator pick.
    # The evaluator picks the point with the highest score.
    # If we want to implement Thompson Sampling via scoring:
    # We can't easily sample from the GP for all 2000 points efficiently in Python without vectorized sampling,
    # but we can approximate the "expected" regret or use the quantiles.
    
    # Actually, the standard "best score" selection is deterministic given the scores.
    # To emulate stochasticity (which helps escape local minima in short runs),
    # we can add noise to the scores. But that's risky.
    
    # Let's go back to the **Dynamic Weighting** of EI and UCB.
    # This is structurally different from the single UCB in the current program.
    
    # Calculate a weight for exploration.
    # If f_best is large, we are likely in a bad region or haven't converged.
    # Exploration is key.
    # If f_best is small, exploitation is key.
    
    # Normalization of f_best for the weight.
    # Typical max Ackley 2D is ~10-15.
    # Let's say f_best=0 is fully exploited, f_best=10 is fully explored.
    
    # w_explore = exp(-f_best / scale) ?
    # If f_best = 0, w_explore = 1.0 (Wait, we want LOW exploration at 0).
    # So w_explore = exp(-f_best / scale) is wrong.
    # We want w_explore HIGH when f_best is HIGH.
    # w_explore = 1.0 - exp(-f_best / scale)?
    # If f_best = 0, w_explore = 0.
    # If f_best = 10, w_explore ~ 1.0 (if scale=5).
    
    scale = 2.0
    w_explore = 1.0 - np.exp(-f_best / scale)
    
    # Clamp between 0 and 1
    w_explore = np.clip(w_explore, 0.1, 1.0) # Never fully zero exploration
    
    w_exploit = 1.0 - w_explore
    
    # Combine
    # EI is naturally in "improvement" units.
    # UCB is in "value" units (negative mean + sigma).
    # They have different scales.
    # To combine them, we should normalize them.
    
    # Min-max normalize EI and UCB?
    # This depends on the spread of the GP predictions, which changes.
    # A safer way: Use Z-score normalization for the combination?
    # Or just rely on the fact that both drive towards low mu.
    
    # Let's try a different structural change: **Kriging Believer**
    # Kriging Believer uses the posterior mean (mu) to simulate future improvements.
    # But for a single step, it's just -mu.
    # That's too greedy.
    
    # Let's try **Variance-Weighted Mean**.
    # score = -mu + alpha * sigma
    # This is UCB.
    
    # What if we use **Quantile of Improvement**?
    # Instead of EI (mean improvement), use the 90th percentile of improvement?
    # This is more robust to outliers in the improvement distribution.
    # Improvement I = f_best - Y, Y ~ N(mu, sigma^2)
    # I ~ N(f_best - mu, sigma^2)
    # We want E[max(0, I)].
    # The "robust" version might be max(0, quantile_90(I)).
    # But we need a smooth score.
    
    # Let's stick to the **Dynamic Weighted Combination** but handle the scale mismatch.
    # EI is roughly of magnitude sigma.
    # UCB is roughly of magnitude sigma (if beta ~ 1-3).
    # So they are comparable.
    
    # Score = w_exploit * EI + w_explore * UCB
    
    # However, EI can be 0 if mu >> f_best. UCB can be large if sigma is large.
    # This encourages exploring high sigma areas even if mu is high.
    # In the early phase, we WANT this.
    # In the late phase, w_exploit is high, so EI dominates. EI is 0 for bad points, positive for good.
    
    scores = w_exploit * ei + w_explore * ucb
    
    # One issue: EI and UCB are not normalized to the same scale across the domain.
    # If one point has high sigma, its UCB is high. Its EI is also high.
    
    # Let's try a **Standardized UCB** (Student's T-like)
    # This is the z-score itself.
    # score = z = (f_best - mu) / sigma
    # This is Probability of Improvement (monotonic).
    # Max z is equivalent to max PI.
    # This is a very strong exploitation strategy.
    # It ignores the magnitude of the potential improvement, only the likelihood.
    # For Ackley, this might get stuck in local minima because it doesn't account for "how much" better it could be.
    
    # Let's mix z-score (PI) and EI.
    # z is unitless. EI is objective.
    
    # Let's try **Log-EI**? No.
    
    # Let's go with the **Dynamic Weighted EI/UCB** but add a small constant to EI to ensure
    # that points with zero EI (mu > f_best) don't have strictly 0 score if they have high sigma?
    # No, if mu > f_best, EI is small/0. UCB is high if sigma is high.
    # w_explore * UCB will still give it score.
    
    # Let's refine the weights.
    # Use a steeper transition to switch from exploration to exploitation faster.
    # scale = 1.0
    
    scale = 1.0
    w_explore = 1.0 - np.exp(-f_best / scale)
    w_explore = np.clip(w_explore, 0.05, 1.0)
    w_exploit = 1.0 - w_explore
    
    # Recalculate with new weights
    scores = w_exploit * ei + w_explore * ucb
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
