import numpy as np

N = 400

def construct():
    # Seed value is 2.0
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try a few random configurations to see if we can beat 2.0
    # We assume the theoretical minimum is 2.0, but let's verify with a small search
    # to be safe against any misinterpretation of the max/conv interaction.
    
    rng = np.random.default_rng(42)
    
    for _ in range(100):
        # Random walk or random uniform
        h = rng.uniform(-1, 1, N)
        s = np.sum(h)
        if np.abs(s) < 1e-6:
            continue
            
        # Scale to make sum 1 to simplify calculation, but value is scale invariant
        # So we can just compute directly
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2 * N * max_conv / (s ** 2))
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
