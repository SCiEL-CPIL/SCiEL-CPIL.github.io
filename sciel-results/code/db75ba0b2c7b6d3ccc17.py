import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    MAX_WORDS = 1 << N
    
    # Pre-generate all candidates
    candidates = list(range(MAX_WORDS))
    
    best_code = []
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer for evaluation overhead
    
    rng = np.random.default_rng(42)
    
    # We will try multiple random orderings
    # To speed up, we can also try starting with a known good linear code?
    # But the linear code is size 256. If we start with it, we can only add words that are distance >=6 from ALL 256 words.
    # This is a very strict constraint. It might be easier to find a large nonlinear code from scratch.
    
    while time.time() - start_time < time_limit:
        # Shuffle candidates
        rng.shuffle(candidates)
        
        current_code = []
        # Optional: Start with 0 to break symmetry? 
        # If we start with 0, we force 0 to be in the code. This is fine as we can translate any code to include 0.
        current_code.append(0)
        
        added_count = 0
        # We iterate through the shuffled list
        # Optimization: If current_code is large, checking distance is expensive.
        # We can break early if we've gone through many candidates without adding any?
        # Let's just do a full pass.
        
        for w in candidates:
            if w == 0:
                continue
            # Check distance to all existing codewords
            valid = True
            for c in current_code:
                if (w ^ c).bit_count() < MIN_DIST:
                    valid = False
                    break
            if valid:
                current_code.append(w)
                added_count += 1
        
        if len(current_code) > len(best_code):
            best_code = current_code
            # If we found a very large code, we might want to try to refine it?
            # For now, just keep the best.
            
    return best_code
