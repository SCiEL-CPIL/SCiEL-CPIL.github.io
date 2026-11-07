"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    UCB is chosen over Expected Improvement (EI) because it provides a more
    balanced exploration-exploitation trade-off for multimodal functions like
    Hartmann 6d, especially in the early stages of optimization. It explicitly
    rewards high uncertainty (sigma) which helps escape local minima.
    
    The score is normalized by the evaluation cost to prioritize cheaper points,
    which is critical given the tight budget (total cost < 30).
    """
    # Ensure sigma is positive to avoid numerical issues, though GP sigma should be > 0
    sigma = np.maximum(sigma, 1e-9)
    
    # Exploration parameter beta.
    # A value of 2.0 is a standard heuristic for UCB in Bayesian Optimization.
    # It encourages exploration when uncertainty is high.
    beta = 2.0
    
    # UCB Score: (f_best - mu) is the exploitation term (improvement over best).
    # beta * sigma is the exploration term (uncertainty).
    # We maximize this value.
    ucb_score = f_best - mu + beta * sigma
    
    # Cost-aware normalization: Divide by cost to prefer cheaper evaluations.
    # Cost is defined as 1 + mean(x), ranging from 1.0 to 2.0.
    score = ucb_score / cost
    
    return score
# EVOLVE-BLOCK-END
