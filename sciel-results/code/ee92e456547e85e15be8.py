import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Upper Confidence Bound
    # mu: mean of GP (2000,)
    # sigma: std of GP (2000,)
    # f_best: best observed value (scalar)
    # cost: cost of evaluating each candidate (2000,)
    
    # Exploration parameter
    beta = 5.0
    
    # UCB score
    ucb = mu + beta * sigma
    
    # Cost-aware score: higher is better
    # We want to maximize improvement per cost.
    # Since this is a minimization problem, lower mu is better.
    # UCB (mu + beta*sigma) is a "value" where higher means "more promising" 
    # in a maximization context. But here we are minimizing.
    # Wait. Standard UCB for minimization is usually:
    # Lower Confidence Bound (LCB): mu - beta * sigma
    # We want to minimize f(x). So we look for low mu and high sigma (uncertainty).
    # The score should be high for points likely to have low f(x).
    # So we should maximize (mu - beta * sigma) / cost ?
    # Or minimize (mu - beta * sigma) / cost ?
    # The return is "higher = evaluate next".
    # So we want the score to be high for good candidates.
    # Good candidates have low f(x).
    # LCB = mu - beta * sigma. Lower LCB is better.
    # So we can return -LCB / cost ? Or 1 / (LCB + eps) ?
    # Or simply: Score = (mu - beta * sigma) is NOT monotonic with "goodness" in a simple "higher is better" way if we don't negate it.
    # If mu=0, sigma=0, LCB=0. If mu=10, sigma=0, LCB=10.
    # We prefer LCB=0.
    # So we want a score that is high when LCB is low.
    # Score = -LCB / cost ?
    # Let's check the EI logic in the seed.
    # EI = (f_best - mu) * ...
    # If mu < f_best, EI > 0.
    # The seed returns EI / cost.
    # This implies that "higher score" corresponds to "higher expected improvement".
    # For UCB, we want to maximize the probability of finding a new minimum.
    # A common cost-aware UCB for minimization is:
    # Score = (mu - beta * sigma) / cost ? No, that would be lower is better.
    # We need to invert it.
    # Score = - (mu - beta * sigma) / cost
    # Let's verify.
    # Candidate A: mu=0, sigma=0. LCB=0. Score = 0.
    # Candidate B: mu=1, sigma=0. LCB=1. Score = -1.
    # A is better (lower f). Score A (0) > Score B (-1). Correct.
    # Candidate C: mu=1, sigma=10. LCB = -9. Score = 9.
    # C is very uncertain, might be low. Score C (9) > Score A (0). Correct, encourages exploration.
    
    # So, Score = - (mu - beta * sigma) / cost
    # Which is (beta * sigma - mu) / cost
    
    lcb = mu - beta * sigma
    scores = -lcb / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
