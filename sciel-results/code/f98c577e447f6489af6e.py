import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    # If f_best is significantly higher than the last observed min, assume new run
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Total expected optimization steps is 25 (30 total evals - 5 initial)
    step = min(_step_counter, 24)
    total_steps = 25.0
    
    # Handle zero sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute standard Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Use standard EI as the base score
    ei_scores = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    ei_scores = np.nan_to_num(ei_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # --- Structural Change: Diversity-Aware Selection ---
    # Instead of just returning EI, we penalize candidates that are too close to
    # other high-EI candidates. This mimics batch acquisition logic (like Kriging Believer
    # or simple diversity penalties) without needing to know the batch size explicitly,
    # by assuming a "virtual batch" of the top candidates.
    
    # We don't have coordinates, but we can't compute distance in input space.
    # HOWEVER, the prompt says "No coordinates are passed".
    # Wait, if no coordinates are passed, how do we do diversity?
    # We CANNOT do spatial diversity if we don't have x coordinates.
    # 
    # Re-reading the prompt: "mu and sigma have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # This implies we cannot compute distances between candidates.
    # Therefore, the "Diversity-Aware" strategy based on spatial distance is IMPOSSIBLE with the given interface.
    
    # I must pivot. The "different strategy" must be achievable with only mu, sigma, f_best.
    
    # Alternative Structural Change: **Thompson Sampling via Sampled Functions**
    # Instead of deterministic EI/LCB, we can simulate the posterior uncertainty.
    # However, we only have mu and sigma, not the full GP kernel or data.
    # We cannot sample from the GP directly.
    
    # Let's look at the "Search-based" inspiration.
    # What if we use a **Hyperparameter-Adaptive EI**?
    # The current program uses fixed EI.
    # We can estimate the "quality" of the current GP fit by looking at the distribution of mu and sigma.
    
    # A robust alternative to static blending is **UCB with a learned/adaptive beta**.
    # Or, better: **Expected Regret** vs **Expected Improvement**.
    
    # Let's try a **Greedy Batch-Approximation via Local Penalties** is not possible.
    
    # Let's try **Softmax-based Selection**? No, we return scores.
    
    # Let's look at the failure of the current program (score 2.25).
    # The current program blends LCB and EI.
    # The baseline (score 1.0) is likely pure EI or pure LCB.
    # Why is 2.25 better than 1.0? (Wait, lower is better for regret, but score is seed/value? 
    # "Scoring (lower is better): mean over the seeds of the simple regret... The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # So higher score is better. Current is 2.25. Baseline is 1.0.
    # So the current program is already 2x better than baseline.
    
    # The directive says: "Make a STRUCTURAL change... A first try in a new direction may score worse; that is acceptable."
    # But I should aim for something that might be better.
    
    # Since I cannot use coordinates, I cannot do spatial diversity.
    # What other structural change is possible?
    # 1. **Optimistic Initial Values**: The GP is standardized. Early on, sigma is high everywhere.
    # 2. **Posterior Sampling Approximation**: Since we can't sample the GP, we can approximate the "random function" view by adding noise to the acquisition? No.
    # 3. **Adaptive Exploration via Uncertainty Statistics**:
    #    Instead of a time-based schedule, use the statistics of the current mu/sigma distribution.
    #    If the mean sigma is high, we are in early stages -> explore more.
    #    If the mean sigma is low, we are in late stages -> exploit more.
    
    # Let's implement **Adaptive UCB based on Global Uncertainty**.
    # This is structurally different from the time-based blend. It reacts to the actual state of the GP.
    
    # Calculate the mean uncertainty of the candidates.
    mean_sigma = np.mean(sigma_safe)
    # Normalize mean_sigma to a 0-1 range. 
    # In this problem, the function values are in [-2.5, 0]. 
    # Initial sigma is likely around 0.5-1.0. Later it drops to 0.01-0.1.
    # We can map mean_sigma to a beta parameter for UCB.
    
    # Let's assume a typical range for sigma in this problem.
    # If mean_sigma > 0.5, we are very early.
    # If mean_sigma < 0.05, we are very late.
    
    # Use a sigmoid or linear mapping to determine the exploration weight.
    # Let's use a robust mapping.
    # beta = beta_min + (beta_max - beta_min) * (mean_sigma / sigma_ref)
    
    # sigma_ref: A reference value for "high uncertainty". Let's say 0.2.
    # If mean_sigma is 0.2, beta is at the midpoint.
    
    beta_min = 0.5
    beta_max = 3.0
    sigma_ref = 0.2
    
    # Clip the ratio to [0, 1] to avoid extreme values if sigma is unexpectedly large/small
    ratio = np.clip(mean_sigma / sigma_ref, 0.0, 1.0)
    
    # Linear interpolation of beta
    beta = beta_min + (beta_max - beta_min) * ratio
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # We want to minimize f. So we look for low f.
    # UCB for minimization: -mu + beta * sigma
    # (We maximize the score -mu + beta*sigma)
    
    ucb_scores = -mu + beta * sigma_safe
    
    # To ensure we don't completely ignore exploitation if the GP is overconfident,
    # we can blend UCB with EI slightly, but the directive says structural change.
    # Pure adaptive UCB is a distinct family from the fixed-schedule blend.
    
    # However, UCB is known to be more exploratory than EI.
    # Given the current program (Blend) scores 2.25, maybe pure EI (Baseline 1.0) was too conservative?
    # Or maybe the Blend was lucky?
    
    # Let's try **Pure EI with a "Gap" Penalty**?
    # No, let's stick to **Adaptive UCB**. It is a standard, robust method that adapts to the data.
    
    # One refinement: Ensure the scores are finite.
    ucb_scores = np.nan_to_num(ucb_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ucb_scores
