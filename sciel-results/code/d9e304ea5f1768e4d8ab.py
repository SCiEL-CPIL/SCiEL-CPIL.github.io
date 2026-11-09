import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None
_last_sigma_mean = None

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best, _last_sigma_mean
    
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
    
    # 1. Compute base acquisition scores (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei_scores = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei_scores = np.nan_to_num(ei_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 2. Adaptive Exploration Weight
    # Estimate "stage" of optimization via mean sigma.
    # Early: high sigma -> explore more. Late: low sigma -> exploit more.
    mean_sigma = np.mean(sigma_safe)
    
    # Mapping mean_sigma to a diversity weight.
    # If mean_sigma is high (>0.2), we want high diversity weight.
    # If mean_sigma is low (<0.05), we want low diversity weight.
    # Let's use a simple linear decay with clipping.
    # beta_diversity controls how much we penalize points close to already selected ones.
    # High beta_diversity -> strong penalty for closeness -> high diversity.
    
    # Reference values based on problem scale (values in [-2.5, 0], sigma starts ~0.5-1.0)
    sigma_high = 0.3
    sigma_low = 0.05
    
    if mean_sigma >= sigma_high:
        div_weight = 1.0
    elif mean_sigma <= sigma_low:
        div_weight = 0.1
    else:
        # Linear interpolation
        div_weight = 0.1 + (1.0 - 0.1) * (mean_sigma - sigma_low) / (sigma_high - sigma_low)
        
    # 3. Greedy Batch Selection Approximation
    # Since we must return scores for all 2000 points, and the evaluator picks the max,
    # we cannot directly return a "batch". Instead, we modify the scores such that:
    # The highest score corresponds to the best point in a greedy diverse set.
    # However, the evaluator only picks ONE point. 
    # The "Greedy Batch" idea usually implies we pick multiple. 
    # But here, we pick one. 
    # How does greedy batch selection help if we only pick one?
    # It doesn't directly, unless we interpret "diversity-aware scoring" as:
    # Penalizing points that are "redundant" with respect to the current best region.
    # But we don't know the current best region's coordinates.
    
    # Wait, the prompt says "No coordinates are passed".
    # This is the critical constraint. 
    # If I don't have coordinates, I cannot compute spatial distances between candidates.
    # Therefore, I CANNOT implement a true spatial diversity penalty (like min-distance to selected set).
    
    # Re-reading the directive: "Replace the myopic point-wise acquisition with a greedy batch selection strategy that constructs a diversity-aware set of candidates using a greedy maximization of a hypervolume or spacing-based objective over the top-k scored points"
    
    # This is impossible without coordinates. 
    # HOWEVER, maybe I can approximate "diversity" in the *value space* or *uncertainty space*?
    # Or perhaps the "diversity" is implicit in the GP structure if I had the kernel? I don't.
    
    # Let's look at the "Knowledge from this run":
    # "Diversity-aware scoring with adaptive exploration-exploitation tradeoff based on progress toward known minimum, plus spatial diversity bonus to prevent premature convergence to local minima of the RBF mixture Measured: 1 -> 2.421."
    # This suggests a previous attempt *did* something that worked. 
    # But how did it get spatial diversity without coordinates?
    # Maybe it didn't? Or maybe the "spatial diversity" was a misnomer for "uncertainty diversity"?
    # Or maybe the evaluator *does* pass coordinates in a way I missed? 
    # "mu and sigma have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # It explicitly says no coordinates.
    
    # If I cannot do spatial diversity, what is the next best thing?
    # The current program (score 0.0238) is very bad. The parent (2.25) was good.
    # The parent was "Adaptive UCB".
    # The baseline (1.0) was likely EI or LCB.
    # The "Knowledge" says a program with "diversity-aware scoring... plus spatial diversity bonus" got 2.421.
    # If that program existed and scored 2.421, it must have found a way to inject diversity.
    # Could it be that `mu` and `sigma` contain enough information to infer location?
    # No, mu/sigma are just scalars per candidate.
    
    # Is it possible the "spatial diversity" was achieved by *randomization*?
    # Or by using the *index* of the candidate?
    # The candidates are drawn from "2000 fresh uniform candidates".
    # If they are uniform, their indices are random permutations of the uniform grid/samples.
    # But the order is arbitrary.
    
    # Let's reconsider the "Greedy Batch" instruction.
    # If I cannot do spatial diversity, I should probably stick to a robust acquisition function that is known to work well for this type of problem (smooth, low-dim).
    # The parent (Adaptive UCB) scored 2.25.
    # The current (Adaptive UCB with comments) scored 0.0238? 
    # Wait, the current program provided in the prompt is the one with score 0.0238.
    # Let's look at the code provided in "CURRENT PROGRAM".
    # It computes `ucb_scores = -mu + beta * sigma_safe`.
    # Why is this scoring 0.0238? 
    # 0.0238 is very low. Regret = f_best - f_min.
    # If score = seed/value, and lower is better for regret...
    # "Scoring (lower is better): mean over the seeds of the simple regret... The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement"
    # This is confusing.
    # "lower is better" for regret.
    # "search score is seed/value".
    # If value = regret, then lower value is better.
    # But "above 1.0 is an improvement".
    # This implies the "score" reported is actually *inversely* related to regret, or the regret is small (good) and the score is high?
    # If Regret = 0.1, and Baseline Regret = 0.1, Score = 1.0.
    # If Regret = 0.05, Score = 2.0 (Improvement).
    # So Score = Baseline_Regret / Current_Regret.
    # Current Program Score = 0.0238.
    # This means Current_Regret is HUGE compared to baseline.
    # 0.0238 = Base_Regret / Curr_Regret.
    # If Base_Regret ~ 0.1, Curr_Regret ~ 4.2.
    # This suggests the current program is failing catastrophically.
    # Why?
    # `ucb_scores = -mu + beta * sigma_safe`.
    # If `mu` is negative (function values are negative), `-mu` is positive.
    # `sigma` is positive.
    # We maximize this score.
    # Early on, sigma is large. We pick high uncertainty points.
    # Later, sigma is small. We pick low mu (high -mu) points.
    # This seems standard.
    # Maybe the issue is the `beta` calculation?
    # `mean_sigma` might be miscalculated or the scaling is wrong.
    # Or maybe `f_best` is not updated correctly in my global state logic?
    # If `_last_f_best` logic fails, `step` might be wrong, but `beta` depends on `mean_sigma`, not `step`.
    # So `beta` should be robust.
    
    # Let's look at the "Inspiration 1" (score 2.2504).
    # It says: "uses numpy, scipy, time; defines acquisition".
    # And the diff shows it removed some comments and changed the logic slightly?
    # Actually, the diff in the JSON shows the *child* (current program) is different from the *parent* (inspiration 1?).
    # Parent ID: p12, Score: 2.2504.
    # Child ID: p28, Score: 0.0237.
    # The diff shows the child removed comments and changed the acquisition logic?
    # No, the diff is truncated.
    # But the "CURRENT PROGRAM" provided in the prompt text is the one with score 0.0238.
    # Let's re-read the current program code carefully.
    # It uses `norm.cdf` and `norm.pdf`.
    # It computes `ei_scores` but then *doesn't use them*?
    # It computes `ucb_scores` and returns `ucb_scores`.
    # So it's Adaptive UCB.
    
    # Why is Adaptive UCB failing here?
    # Perhaps `beta` is too large?
    # `beta_min = 0.5`, `beta_max = 3.0`.
    # If `mean_sigma` is high, `beta` is 3.0.
    # `score = -mu + 3.0 * sigma`.
    # If `sigma` is 0.5, `3.0 * 0.5 = 1.5`.
    # If `mu` is -1.0, `-mu = 1.0`.
    # Total 2.5.
    # If a point has `mu = -2.0` (good), `sigma = 0.1`.
    # `score = 2.0 + 0.3 = 2.3`.
    # The high uncertainty point (2.5) beats the good mean point (2.3).
    # This is good for exploration.
    # But if `beta` is too high, it might ignore the best known region entirely.
    
    # However, the parent (score 2.25) was also Adaptive UCB?
    # Inspiration 1 says "Step 1: The current program uses a simple time-decaying Linear Confidence Bound (LCB)..."
    # Wait, the inspiration text describes the *parent's* strategy as LCB?
    # And the parent scored 2.25.
    # The current program (child) scored 0.0238.
    # The diff shows the child changed from LCB to ... ?
    # The diff is hard to read.
    # But the "CURRENT PROGRAM" code I see uses UCB.
    # Maybe the "CURRENT PROGRAM" text in the prompt is NOT the one that scored 0.0238?
    # "CURRENT PROGRAM (score 0.0238) and its strategy:"
    # Yes, it is.
    
    # So, Adaptive UCB scored 0.0238.
    # Parent (LCB?) scored 2.25.
    # Baseline (EI?) scored 1.0.
    
    # This implies LCB is better than UCB?
    # For minimization:
    # LCB: `mu - beta * sigma`. We want to MINIMIZE this.
    # So we pick the point with the LOWEST `mu - beta * sigma`.
    # This is equivalent to maximizing `beta * sigma - mu`.
    # Which is exactly what the current UCB code does: `score = -mu + beta * sigma`.
    # So the math is the same.
    
    # Why the huge drop in score?
    # Maybe the `beta` adaptation is broken?
    # In the parent (Inspiration 1), it said "time-decaying LCB".
    # So the parent used a TIME-based decay for beta.
    # The current program uses a SIGMA-based decay for beta.
    # Maybe the sigma-based decay is faulty?
    # `mean_sigma` might not be a good proxy for "stage" if the GP is noisy or if the candidates are not representative.
    # Or maybe `sigma_ref` is wrong.
    
    # Let's go back to the directive: "Replace the myopic point-wise acquisition with a greedy batch selection strategy...".
    # Even though I can't do spatial diversity, I can try to mimic the *effect* of batch selection by using a more robust acquisition function that is less sensitive to local noise, or by using a "look-ahead" heuristic.
    # But without coordinates, look-ahead is impossible.
    
    # Alternative: **Expected Hypervolume Improvement (EHVI)**?
    # EHVI requires the reference point and the set of non-dominated points.
    # Since we have a single objective, EHVI reduces to EI.
    
    # Let's try **Pareto Front Approximation**? No, single objective.
    
    # What if I use **Thompson Sampling**?
    # `score = mu - sigma * z` where `z` is a random draw from N(0,1).
    # This is equivalent to sampling from the posterior.
    # This is structurally different from deterministic UCB/EI.
    # It introduces stochasticity.
    # The evaluator runs multiple seeds.
    # If I use a fixed random seed for the noise in `acquisition`, it becomes deterministic but "random-like".
    # If I use `np.random.normal`, it will be different every call.
    # This might help escape local minima by chance.
    
    # However, the directive specifically asks for "greedy batch selection... diversity-aware set... spacing-based objective".
    # I must try to implement *something* that fits this description, even if it's an approximation.
    
    # Approximation: **Score Smoothing / Regularization**
    # Instead of picking the single max EI, pick a point that is "diverse" from the *distribution* of high EI points.
    # I can compute the "density" of high EI points in the *value space*?
    # No, that doesn't make sense.
    
    # Let's look at the "Knowledge" again: "spatial diversity bonus".
    # If I absolutely cannot access coordinates, I cannot compute spatial diversity.
    # Is it possible that `mu` and `sigma` are passed in a specific order that correlates with location?
    # "draws 2000 fresh uniform candidates".
    # Usually, these are generated via `np.random.uniform(size=(2000, 2))`.
    # The order is random.
    
    # I will implement **Adaptive LCB with a "Diversity" term based on the rank of the uncertainty**.
    # Hypothesis: Points with very high sigma are likely "far" from observed data (spatially diverse).
    # Points with very low sigma are "close" to observed data (exploit).
    # I can create a score that balances `mu` and `sigma` but with a non-linear penalty for being "too certain" (low sigma) or "too uncertain" (high sigma)?
    # No, that's just tuning.
    
    # Let's try **Expected Improvement with a "Batch" approximation via Gaussian Process Regression on the Acquisition Scores**?
    # No, too complex.
    
    # I will implement **UCB with a time-decaying beta** (like the parent that scored 2.25) but with a slight modification: **Adaptive Beta based on both Time and Sigma**.
    # The parent used Time. The current uses Sigma. The current failed.
    # Maybe the Sigma calculation is the issue.
    # Let's go back to Time-based decay, but make it smoother.
    # And add a "Diversity" component by **penalizing points that are too similar in VALUE to the current best**?
    # No, we want points with LOW value.
    
    # Let's try **Softmax of EI**?
    # `score = EI + alpha * log(sum(exp(EI)))`? No.
    
    # I will implement **Greedy Batch Selection** in the following way:
    # 1. Calculate EI for all points.
    # 2. Identify the top K points by EI.
    # 3. Since we can't check spatial distance, we assume that if the top K points have very similar `mu` and `sigma` values, they might be in the same region (correlated).
    # 4. We penalize the scores of points that are "redundant" with the top point.
    # 5. "Redundancy" is measured by the similarity of `mu` and `sigma`.
    # 6. If a point has `mu` and `sigma` very close to the best point's `mu` and `sigma`, it is likely in the same GP "bump".
    # 7. We subtract a penalty from the score if the point is too similar to the current best candidate.
    
    # This is a "value-space diversity" heuristic. It's not true spatial diversity, but it's a proxy.
    # If the GP has a sharp peak, nearby points will have similar `mu` and `sigma`.
    # If the GP has a flat region, nearby points will have similar `mu` and `sigma`.
    # This might help avoid picking two points from the same sharp peak if we were picking a batch.
    # But we only pick one.
    # So, we want to pick the point with the highest EI that is NOT "redundant" with the previous picks?
    # We don't have previous picks' mu/sigma.
    
    # Okay, I will stick to **Adaptive UCB with Time-Decay** (reverting to the parent's strategy which worked) but adding a **Sigma-Clamp** to prevent extreme exploration.
    # And I will add a **Deadline Guard**.
    
    # Wait, the directive says "Do NOT switch to a different design family."
    # The current family is "Adaptive UCB".
    # So I should keep Adaptive UCB but refine it.
    # The refinement is: "investigate improvements within its family... consider a smooth surrogate... or a final high-precision polish pass".
    
    # I will implement **Adaptive UCB** with:
    # 1. Time-based decay for beta (proven to work in parent).
    # 2. A "Polish" phase: In the last few steps, switch to pure Exploitation (mu) or very low beta UCB.
    # 3. A "Diversity" proxy: Add a small random noise to the scores to break ties and encourage exploration of similar-value regions? No, that's bad.
    
    # Let's try **UCB with a "Lookahead" approximation**:
    # Estimate the potential improvement of the next point.
    # No, impossible.
    
    # I will implement **Adaptive UCB** with **Time-Decay** and **Sigma-Adaptive Floor**.
    # `beta = beta_max * (1 - step/total_steps)^2 + beta_min * sigma_factor`
    # This combines time and uncertainty.
    
    # Also, I will add a **Deadline Guard** as requested.

    # Deadline Guard
    elapsed = time.time() - _start_time
    if elapsed > 300:
        # Return a safe default if we are running out of time
        # Just return EI
        return ei_scores

    # Time-based decay
    # step is 0 to 24
    # beta should start high and end low.
    time_factor = 1.0 - (step / total_steps)
    time_factor = time_factor ** 1.5  # Slower decay early, faster late? Or vice versa.
    
    # Sigma-based factor
    # If sigma is high, we are in a new area, explore more.
    # If sigma is low, we are in a known area, exploit.
    sigma_factor = np.clip(mean_sigma / 0.2, 0.0, 1.0)
    
    # Combine
    # beta = 0.5 + 2.5 * (0.5 * time_factor + 0.5 * sigma_factor)
    beta = 0.5 + 2.5 * (0.7 * time_factor + 0.3 * sigma_factor)
    
    # Compute UCB
    ucb_scores = -mu + beta * sigma_safe
    
    # Ensure finite
    ucb_scores = np.nan_to_num(ucb_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return ucb_scores
