"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to track the number of calls to implement time-decaying UCB.
# The evaluator calls acquisition 25 times (30 total evaluations - 5 initial).
# We reset the counter when the function is first called in a new session?
# No, the evaluator runs seeds 0, 1, 2 independently.
# However, the function state is global. We need to be careful.
# The prompt says "For each seed... the evaluator starts from 5 uniform random points... until 30 evaluations".
# This implies 3 separate runs.
# If we use a global counter, it will accumulate across seeds unless we reset.
# How to detect a new seed?
# The first call for a seed will have a specific mu/sigma pattern? No.
# We can assume the evaluator imports the module once and runs the 3 seeds sequentially.
# We need to reset the step counter for each seed.
# A common trick: if the number of points in mu/sigma changes, or if we detect a "restart".
# But mu/sigma always has shape (2000,).
# Let's look at the problem: "For each seed 0, 1 and 2 the evaluator starts from 5 uniform random points...".
# This means the loop runs 25 times per seed.
# If we just use a global counter, seed 1 will start at step 25, seed 2 at step 50.
# The beta(t) decay would be wrong for seeds 1 and 2.
# We need to reset the counter.
# One heuristic: if the current f_best is "very good" or if the mu/sigma distribution suggests a fresh start?
# Actually, the simplest robust way is to check if the function is being called for the first time in a "long while" or if we can infer the step.
# But we don't have access to the step count directly.
# Wait, the prompt says "Keep the function names and signatures...".
# Can we use a closure or a global variable? Yes.
# How to reset?
# If we assume the evaluator runs seeds sequentially in the same process, we can track the number of calls.
# If we see 25 calls, we reset? That's brittle if the evaluator stops early or something.
# However, the task says "until 30 evaluations". So exactly 25 acquisition calls per seed.
# Let's try to detect a reset by checking if the global counter has reached 25.
# If counter >= 25, reset to 0.
# This assumes exactly 25 calls per seed. The contract says "until 30 evaluations including the first 5", so 25 acquisition calls.
# This seems safe enough for this specific contract.

_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # Increment and handle reset
    _call_count += 1
    if _call_count > 25:
        _call_count = 1
    
    # Calculate beta(t)
    # t is the current step (1 to 25)
    t = _call_count
    # Beta decays from 2.0 to 0.0 over the 25 steps?
    # The directive says: beta(t) = 2.0 * (1 - t/30)
    # If t=1, beta = 2.0 * (29/30) ~ 1.93
    # If t=25, beta = 2.0 * (5/30) ~ 0.33
    # This provides strong exploration early, shifting to exploitation later.
    beta = 2.0 * (1.0 - t / 30.0)
    
    # UCB acquisition: -mu + beta * sigma
    # We want to maximize this score (higher means evaluate next).
    # mu and sigma are in objective units.
    # sigma is standard deviation, so it's non-negative.
    # Ensure sigma is not zero to avoid numerical issues, though UCB handles 0 sigma fine (just -mu).
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
# EVOLVE-BLOCK-END
