"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # Kill switch at 330s, keep safety margin
_MARGIN = 10.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB (Upper Confidence Bound).
    UCB(x) = mu(x) + beta * sigma(x)
    Score = UCB(x) / cost(x)
    
    Since we are minimizing, a lower mu is better. 
    Wait, standard UCB for minimization is usually Lower Confidence Bound (LCB):
    LCB(x) = mu(x) - beta * sigma(x)
    
    The prompt asks for "higher = evaluate next".
    If we use LCB, lower is better. So we should return -LCB / cost.
    Or, if we interpret UCB in the context of "maximizing improvement", 
    but the prompt explicitly suggests: "UCB (Upper Confidence Bound) acquisition function, UCB(x) = mu(x) + beta sigma(x)... divide this value by a linear cost penalty".
    
    Let's re-read carefully: 
    "Strategy to try... Replace the standard Expected Improvement with a UCB ... UCB(x) = mu(x) + beta sigma(x) ... and divide this value by a linear cost penalty"
    
    However, for minimization, maximizing mu + beta*sigma is wrong. 
    Maximizing mu means going to high values. We want low values.
    So we should minimize mu - beta*sigma.
    To fit "higher score = evaluate next", we should return:
    Score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    Let's check the inspiration/lessons. The current program uses EI which is positive when mu < f_best.
    If I follow the directive literally "UCB(x) = mu(x) + beta sigma(x)", and I am minimizing, that directive might be slightly misphrased for minimization, or it implies maximizing the negative of the objective.
    
    Let's assume the standard BO practice for minimization:
    We want to find low values.
    Exploitation: low mu.
    Exploration: high sigma.
    LCB = mu - beta * sigma.
    We want to minimize LCB.
    Acquisition score (higher is better) = -LCB / cost = (beta * sigma - mu) / cost.
    
    Let's verify the "UCB" naming in the directive. Sometimes people call the term "UCB" regardless of sign, or they mean the confidence bound on the maximum (if maximizing). Here we minimize.
    If I use (beta * sigma - mu) / cost, I am prioritizing high uncertainty and low mean.
    
    Let's add a deadline guard.
    """
    # Deadline check
    if time.time() > _DEADLINE - _MARGIN:
        # Return a safe, finite array. 
        # Ideally we'd return something that doesn't crash, but since we can't change the loop,
        # we just return zeros or a simple heuristic to ensure termination if called repeatedly?
        # Actually, the acquisition function is called once per iteration of the loop.
        # If we are close to deadline, the loop will likely stop or timeout.
        # Returning zeros is safe.
        return np.zeros_like(mu)
    
    # Clip sigma to avoid issues, though GP sigma should be > 0
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Beta for exploration. 
    # Hartmann 6d is multimodal. We need sufficient exploration.
    # Typical beta values are 1.0 to 3.0. 
    # Since cost varies, and we want to explore cheaply, 
    # a moderate beta is good.
    beta = 2.0
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we negate it.
    lcb = mu - beta * sigma_safe
    score = -lcb / cost
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
