"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level state to track time and potentially other things
# We need to be careful with state across calls since the evaluator might
# reuse the same process or restart. The contract says "for each seed 0, 1 and 2",
# implying 3 separate runs. However, if the module is imported once and
# acquisition is called 30 times for seed 0, then 30 times for seed 1, etc.,
# global state persists.
# We should reset state if we detect a new "run".
# A new run is likely detected if f_best increases significantly or if we can track
# that the sequence of calls is restarting.
# However, the simplest robust approach for a single function call is to avoid
# complex state if possible, or reset based on heuristics.
# Let's look at the budget: 330s total. 30 evaluations per seed, 3 seeds.
# Each evaluation draws 2000 candidates.
# The acquisition function is called once per evaluation step.
# Total calls per seed: 30.
# Total calls total: 90.
# Time per call must be very small.
# The deadline guard is for the *program* (the whole evaluation loop? or just this function?)
# "Each evaluation must finish within 300 s."
# "The evaluator KILLS the program after 330s wall-clock."
# This implies the *entire* script execution (all 3 seeds) must finish in 330s.
# Since we only provide `acquisition`, we cannot control the loop.
# However, we can add a time check inside `acquisition` to ensure we don't hang.
# But `acquisition` is just a math op on 2000 floats. It's very fast.
# The bottleneck is likely the GP computation in the evaluator, not our function.
# So the deadline guard might be less critical for *this specific function* unless
# we do heavy lifting. We will keep it simple.

# Let's try a strategy that adapts exploration.
# The inspiration suggests blending EI and UCB.
# The current program is pure EI.
# Pure EI can get stuck in local minima for multimodal functions like Styblinski-Tang.
# Adding a small UCB term helps escape.
# A fixed UCB term (like 0.05 * sigma) might be too much or too little.
# An adaptive approach: start with more exploration, decay to exploitation.
# But we don't know the step number easily without state.
# Let's try a simple, robust blend.
# score = EI + kappa * sigma
# If kappa is too high, it explores too much. If too low, it exploits too much.
# Let's try a moderate kappa.
# Also, we can normalize sigma if needed, but sigma is already in objective units.
# The scale of f for Styblinski-Tang is around -39 to 40.
# sigma might be small or large depending on the GP fit.

# Let's stick to a simple, well-tuned EI with a small exploration bonus.
# The inspiration 1 used 0.05. Let's try a slightly different approach:
# Use EI, but ensure we don't ignore high uncertainty regions completely.
# A common trick is to use EI but with a "floor" or a specific weighting.

# Let's try: score = EI + 0.1 * sigma * (1.0 / (1.0 + EI))
# This adds more exploration when EI is low (uncertain or not promising)
# and less when EI is high (very promising).
# Actually, simpler: score = EI + 0.05 * sigma.

# Let's refine the "adaptive" idea from the inspiration but make it safer.
# If we assume the function is called sequentially for a single seed,
# we can track the number of calls.
# However, if the state persists across seeds, we need to reset.
# How to detect a new seed?
# f_best starts high (worst observed) and decreases.
# If f_best suddenly increases, it's likely a new seed.
# Let's implement this state tracking.

_state = {
    'n_calls': 0,
    'prev_f_best': None,
    'start_time': None
}

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with a small
    UCB exploration term that decays over time.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Detect new seed/run
    # If f_best increases, we likely started a new run (new random points)
    if _state['prev_f_best'] is not None and f_best > _state['prev_f_best'] + 1e-6:
        _state['n_calls'] = 0
        _state['start_time'] = time.time() # Reset time if needed, though not strictly necessary for this logic
    
    _state['prev_f_best'] = f_best
    n = _state['n_calls']
    _state['n_calls'] = n + 1
    
    # Total expected evaluations per seed is 30.
    # We want to explore more in the first few, exploit later.
    # Let's say we consider the first 10 calls as "exploration phase".
    # Decay factor for exploration: starts at 1.0, goes to 0.0 by call 30.
    # Or a smooth decay.
    # Let's use a simple linear decay for the UCB weight.
    # weight = max(0, 1.0 - n / 30.0)
    # ucb_term = weight * 0.1 * sigma
    
    # However, a fixed small exploration term is often more robust than a decaying one
    # if the decay is not tuned well.
    # Let's try a constant small exploration term first, as in inspiration 1,
    # but maybe tune the coefficient.
    
    # Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration term
    # Let's try a constant 0.05 * sigma as in the inspiration, which scored low (0.0025).
    # Wait, the inspiration score was 0.0025, which is *worse* than 1.0.
    # So adding 0.05 * sigma made it worse.
    # This suggests that pure EI (score 1.0) is better than EI + 0.05*UCB.
    # Maybe the exploration term was too strong, or the function is well-captured by EI.
    # Or maybe the normalization was off.
    
    # Let's try a *very* small exploration term, or a different formulation.
    # Perhaps the issue is that EI already accounts for some exploration via the sigma term in the PDF.
    # Adding more sigma might over-explore.
    
    # Alternative: Use "Probability of Improvement" (PI) blended with EI?
    # PI = norm.cdf(z). This is less aggressive than EI.
    
    # Let's try to stick close to the current program (which is good) but make it slightly
    # more robust to numerical issues or local minima.
    # The current program is pure EI.
    # Let's try "EI with a small epsilon" or just keep it pure EI but ensure stability.
    
    # Actually, let's look at the "Inspiration" again. It scored 0.0025.
    # The current program scores 1.0.
    # So the inspiration is BAD.
    # We should NOT blindly copy the inspiration.
    # We should combine the *structure* (state tracking, time guard) with the *current* logic (pure EI)
    # and maybe a *tiny* refinement.
    
    # What if we use a slightly modified EI that is more robust?
    # Some sources suggest using `EI = (f_best - mu) * Phi(z) + sigma * phi(z)` is standard.
    # Maybe the issue is that for Styblinski-Tang, we need to explore more *early*.
    # But the inspiration's early exploration (via constant UCB) failed.
    # Maybe the UCB term should be *negative*? No, higher score is better.
    
    # Let's try a different blend:
    # Score = EI * (1 + 0.1 * norm.pdf(z))
    # This boosts points that are in the "shoulder" of the distribution?
    
    # Let's try the simplest improvement:
    # Keep pure EI, but add a time guard to prevent any potential hangs (though unlikely).
    # And maybe ensure that if sigma is very large, we don't get numerical issues.
    
    # Let's try a "safe" EI that caps the score? No, we want to maximize.
    
    # Let's try a small exploration term that *decays* very quickly, so it only affects
    # the very first few steps.
    # If n < 5, add 0.05 * sigma. Else, add 0.
    # This might help escape initial bad local minima without hurting later exploitation.
    
    if n < 5:
        ucb_weight = 0.05
    else:
        ucb_weight = 0.0
        
    ucb_term = ucb_weight * sigma
    score = ei + ucb_term
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
