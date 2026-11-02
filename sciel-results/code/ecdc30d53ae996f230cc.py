import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a nonnegative step function with 600 equal intervals on [-1/4, 1/4].
    Strategy: Return a uniform distribution.
    
    Reasoning:
    The seed program (spike) scores 1.0.
    Inspirations using gradient descent from a spike or similar initializations
    achieved ~1.1, suggesting that spreading the mass improves the score.
    The metric likely involves a ratio where a uniform distribution minimizes the
    "normalized maximum" (e.g., if it's max/mean or max/L1^2, uniform is optimal
    or near-optimal compared to spikes).
    
    A uniform function has the lowest possible "peakiness" in its autocorrelation
    relative to its energy or mass, depending on the exact normalization.
    Given the stagnation of gradient methods, a deterministic global optimum
    candidate like the uniform distribution is a strong structural change.
    """
    start_time = time.time()
    deadline = 145.0
    
    # Uniform distribution
    heights = [1.0] * N
    
    # Check time (instant)
    if time.time() > deadline:
        # Fallback (unlikely)
        return [1.0] * N
        
    return heights
# EVOLVE-BLOCK-END
