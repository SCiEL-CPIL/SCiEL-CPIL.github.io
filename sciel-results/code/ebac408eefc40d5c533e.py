"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    if time.time() - _START_TIME > 320:
        # Return a neutral score if we are running out of time
        return np.zeros(len(mu))
    
    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid log(0) or division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Strategy: Two-stage nested acquisition
    # The standard loop evaluates 1 point per iteration.
    # We have 2000 candidates. 
    # Instead of scoring all 2000 with a single complex function and picking the max,
    # we simulate a local search within the candidate set to find a better representative.
    #
    # Stage 1: Coarse Selection using Cost-Aware EI
    # We calculate a preliminary score for all 2000 points.
    # Score_1 = EI(x) / cost(x)
    # This identifies regions that are both promising (low mu) and cheap.
    
    # Calculate Z-score
    z = (f_best - mu) / sigma
    
    # Calculate EI (Expected Improvement) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Aware Score
    # Using log-cost or inverse cost? 
    # Cost is in [1, 2] (since x in [0,1]^4, mean(x) in [0,1], cost = 1 + mean(x) in [1,2]).
    # Division by cost is linear. 
    # Let's try a slightly more aggressive penalty for expensive points to ensure we stay within budget.
    # score_coarse = ei / (cost ** 1.5)
    # This heavily favors cheap points if EI is comparable.
    
    score_coarse = ei / (cost ** 1.5)
    
    # Stage 2: Fine Selection using Pure EI (or UCB)
    # The top candidates from Stage 1 are likely in good regions.
    # However, pure EI might be too greedy. 
    # Let's select the top 100 candidates from the coarse score.
    # Among these top 100, we re-score them with a different acquisition function
    # that is less sensitive to cost but more sensitive to uncertainty, 
    # to ensure we don't miss a slightly more expensive point that has much higher potential.
    
    # Get indices of top 100 candidates from coarse score
    n_candidates = len(mu)
    n_top = min(100, n_candidates)
    
    # Use argpartition for efficiency
    # We want the indices of the largest n_top elements in score_coarse
    top_indices = np.argpartition(score_coarse, -n_top)[-n_top:]
    
    # Initialize final scores to 0 (or very low)
    # We will return scores such that the max score corresponds to the best candidate in the fine stage.
    # But the interface requires returning scores for ALL 2000 points, and the evaluator picks the max.
    # So we need to assign scores to ALL points, but effectively "downweight" the non-top candidates
    # so that the winner is chosen from the fine-tuned set.
    
    # Let's refine the top candidates with a UCB-like score.
    # UCB = mu - beta * sigma (for minimization, we want low value, so we maximize (f_best - mu + beta*sigma)?)
    # Actually, for minimization, UCB is often defined as mu - beta*sigma.
    # We want to MAXIMIZE the score.
    # So Score_Fine = (f_best - mu) + beta * sigma
    # This is optimistic: if mu is low, score is high. If sigma is high, score is high.
    
    beta = 2.0  # Exploration constant
    
    # Calculate UCB-like score for all points (just in case, or for the top ones)
    ucb_score = (f_best - mu) + beta * sigma
    
    # We want the final returned array `score` to have its maximum at the best point from the fine stage.
    # However, the evaluator picks `np.argmax(score)`.
    # If we just return `ucb_score` for the top 100 and 0 for others, the max will be in the top 100.
    # But what if a point outside the top 100 of `score_coarse` has a higher `ucb_score`?
    # The coarse stage was designed to filter out bad points.
    # Let's ensure the final score is a combination that respects the coarse ranking but allows fine-tuning.
    
    # Alternative Structural Change: 
    # Use the coarse score to define a "gate".
    # If score_coarse is below the 100th percentile, set final score to 0.
    # If above, use the UCB score.
    
    threshold = np.partition(score_coarse, -n_top)[-n_top]
    
    # Create final score
    final_score = np.zeros(n_candidates)
    mask = score_coarse >= threshold
    
    # For the masked points, use the UCB score
    # But we must ensure that the UCB score doesn't accidentally make a point with lower coarse score 
    # (but just above threshold) beat a point with much higher coarse score if their UCB scores are similar.
    # Actually, since we only evaluate 1 point, and the evaluator picks argmax, 
    # we just need the global max of `final_score` to be the point we want to evaluate.
    
    # Let's simply set the final score to be the UCB score for the top candidates,
    # and a very low value for the rest.
    final_score[mask] = ucb_score[mask]
    
    # To be safe, let's scale the UCB score so it's comparable to EI if needed, 
    # but since we are taking argmax, relative scale within the mask is what matters.
    # However, if all ucb_score are negative (possible if f_best is very low and mu is high?), 
    # we might pick a "least bad" one. 
    # f_best is the min observed. mu is the prediction.
    # If mu > f_best, f_best - mu is negative.
    # UCB = (f_best - mu) + beta * sigma.
    # Since sigma > 0, this can be positive or negative.
    
    # Let's add a small offset to ensure positive scores for the candidates we care about,
    # or just rely on the fact that the evaluator picks the max.
    # If all scores are negative, the max is the one closest to 0 (least negative), which is fine.
    
    # One risk: If the top 100 coarse candidates all have very low UCB scores, 
    # and a candidate outside the top 100 has a slightly higher UCB score but lower coarse score, 
    # we ignore it. This is the intended "nested" behavior: trust the coarse filter.
    
    # Ensure finite
    final_score = np.nan_to_num(final_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_score
# EVOLVE-BLOCK-END
