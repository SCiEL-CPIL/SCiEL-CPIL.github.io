"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Thompson Sampling: Sample from posterior and select best sampled value per cost
    # To make this deterministic per call (as required by the interface), we use a fixed random seed
    # based on the current state or simply use the mean of the posterior samples.
    # However, standard Thompson Sampling for acquisition functions often uses the posterior mean + beta * sigma (UCB) 
    # or actually samples. Since we must return 2000 scores, we can simulate the "sample" by using a fixed random draw 
    # or, more robustly for a static acquisition function, use a UCB-like strategy which is the deterministic equivalent 
    # of Thompson Sampling in expectation, but let's try a different angle: 
    # The prompt asks for a "different strategy". EI is greedy. 
    # Let's use a UCB (Upper Confidence Bound) style acquisition, which is distinct from EI.
    # UCB = mu + kappa * sigma.
    # But wait, the inspiration 1 said "Switch to Thompson Sampling". 
    # A true Thompson Sampling approach in this context (where we return scores for 2000 points) is tricky 
    # because TS usually picks one point. 
    # However, we can interpret "score" as the probability of being the best, or the expected value of the sample.
    # Actually, a common alternative to EI is UCB.
    # Let's try UCB with a tuned kappa.
    
    # But let's look at the cost-aware aspect.
    # Current: EI / cost.
    # New: (mu + kappa * sigma) / cost? Or maybe (mu + kappa * sigma) - lambda * cost?
    # The problem states: "Evaluating x costs 1 + mean(x scaled to [0, 1])."
    # The acquisition function should prioritize points that are likely to be good AND cheap.
    
    # Let's try a different metric: Probability of Improvement (PI) per cost?
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma).
    # This is different from EI which weighs the magnitude of improvement.
    
    # Or, let's try UCB per cost.
    # kappa controls exploration.
    kappa = 2.0
    score = (mu + kappa * sigma) / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
