"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a combination of Expected Improvement (EI)
    and a penalty for high uncertainty to balance exploration/exploitation.
    
    We use a modified EI that accounts for the fact that we are minimizing.
    To encourage more exploration early on (when sigma is large relative to the range),
    we can use a lower bound on sigma or a different acquisition like
    Upper Confidence Bound (UCB) for minimization.
    
    However, EI is generally robust. A common improvement is to use a
    "noise-free" EI or to adjust the exploration parameter.
    
    Let's try a hybrid: EI with a small exploration bonus based on sigma,
    or simply a well-tuned EI. Since the seed program uses standard EI,
    let's try something different: the "Lower Confidence Bound" (LCB) for minimization,
    which is equivalent to maximizing -f with a confidence bound.
    LCB for minimization: mu - kappa * sigma
    Higher scores are better, so we want to maximize the score.
    If we want to minimize f, we want low mu and high sigma (exploration).
    So score = -mu + kappa * sigma? No, that would favor high sigma always.
    
    Actually, for minimization:
    We want to find points where f is likely to be low.
    LCB (Lower Confidence Bound) for the function value is mu - kappa * sigma.
    We want to minimize this lower bound? No, we want to find the minimum.
    The standard UCB for maximization is mu + kappa * sigma.
    For minimization, we can maximize -mu + kappa * sigma?
    Or minimize mu - kappa * sigma?
    
    Let's stick to a variant of EI that is known to work well, perhaps with a
    slight modification to handle the specific characteristics of Styblinski-Tang.
    
    Alternative: Probability of Improvement (PI) is simpler but often less effective.
    
    Let's try a "Weighted" acquisition or a different kernel-based heuristic.
    Actually, the prompt asks for a *new* approach family.
    Current: EI.
    New: UCB (Upper Confidence Bound) adapted for minimization.
    
    For minimization, the "Upper Confidence Bound" on the negative function?
    Let's define the score to maximize.
    We want to select x that minimizes f(x).
    A common heuristic is to maximize: -mu + beta * sigma.
    Here, -mu encourages low predicted values, and beta * sigma encourages
    exploring areas with high uncertainty.
    
    We need to choose beta. A common choice is related to the standard deviation
    of the observations or a fixed value. Since we don't have access to the
    hyperparameters directly in this function signature, we can use a heuristic.
    
    Another option: The "Expected Hypervolume Improvement" is too complex.
    
    Let's try UCB with a beta that adapts slightly or is fixed.
    beta = 2.0 is a common starting point.
    
    However, EI is often superior to UCB. Let's look at another family:
    "Thompson Sampling" is not applicable here as we must return a score for each point.
    
    What about "Knowledge Gradient"? Too expensive to compute exactly.
    
    Let's try a different EI variant: "Local Expected Improvement" or
    simply tuning the EI with a different exploration constant.
    Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    
    Maybe the issue with the seed program is that it doesn't explore enough
    or exploits too much.
    
    Let's try the "UCB" approach:
    score = -mu + beta * sigma
    
    We need to pick beta. If beta is too high, we explore randomly.
    If beta is too low, we exploit greedily.
    A good heuristic for beta in BO is often based on the standard deviation
    of the observed values, but we only have f_best.
    
    Let's try a fixed beta, say 2.0 or 3.0.
    Or, we can use a time-based beta, but we don't have the step count.
    
    Let's try a different acquisition: "Probability of Improvement" (PI).
    PI = Phi((f_best - mu) / sigma)
    This is simpler and might behave differently.
    
    However, PI is known to be less effective than EI.
    
    Let's go with UCB (Upper Confidence Bound) for minimization.
    Score = -mu + beta * sigma
    We will set beta = 2.0. This encourages exploring areas with high uncertainty
    while still favoring low mean predictions.
    
    To make it more robust, we can normalize sigma or use a log-scale.
    But let's start with a simple linear UCB.
    
    Another idea: Use a "Gaussian Process Upper Confidence Bound" but with
    a beta that depends on the number of evaluations. Since we don't have
    the count, we can estimate it or use a fixed value.
    
    Let's try beta = 2.0.
    
    Wait, the task says "design a new program ... from an approach family that none
    of the current programs use". The current program uses EI. So UCB is a valid
    new family.
    
    Let's implement UCB for minimization.
    """
    # UCB for minimization: maximize -mu + beta * sigma
    # We choose beta = 2.0 as a heuristic exploration parameter.
    beta = 2.0
    score = -mu + beta * sigma
    return score
# EVOLVE-BLOCK-END
