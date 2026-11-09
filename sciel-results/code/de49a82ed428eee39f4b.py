import time
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 600.0
    safety_margin = 5.0
    
    m = 12
    
    try:
        # Get the 12 roots of the Laguerre polynomial L_12(x)
        # These are positive, distinct, and sorted
        roots = roots_laguerre(m)
        roots = sorted(roots.real.tolist())
        
        # Try a range of scales to find the one that minimizes the C4 upper bound.
        # The bound likely decreases with scale, but there may be an optimal point
        # where numerical issues or constraint violations start to matter.
        # We'll test a geometric progression of scales.
        
        best_roots = None
        best_score = float('inf')
        
        # Test scales from 1000 to 100000 in log-spaced steps
        scales = np.logspace(np.log10(1000), np.log10(100000), 20)
        
        for scale in scales:
            if time.time() - start_time > deadline - safety_margin:
                break
            
            scaled_roots = [r * scale for r in roots]
            
            # Verify basic constraints: positive and strictly increasing
            valid = True
            for i in range(len(scaled_roots) - 1):
                if scaled_roots[i] <= 0 or scaled_roots[i] >= scaled_roots[i+1]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Heuristic: the C4 bound likely decreases with scale.
            # Use the maximum root as a proxy for the "spread" that affects the bound.
            # Larger spread -> smaller bound (based on empirical observation).
            # The score is inversely related to the bound, so we want to maximize spread.
            # Use the product of all roots as a more robust measure of spread.
            log_spread = sum(np.log(scaled_roots))
            
            # We want to maximize log_spread (equivalent to maximizing product of roots)
            if log_spread > -best_score:  # Note: best_score is -inf initially, so this works
                best_score = -log_spread
                best_roots = scaled_roots
        
        if best_roots is None:
            # Fallback: use scale=10000
            scale = 10000.0
            best_roots = [r * scale for r in roots]
        
        return best_roots
        
    except Exception as e:
        # Fallback: scaled arithmetic progression (known to work at scale=1000)
        scale = 1000.0
        return [4.0 * scale * (i + 1) for i in range(m)]

# EVOLVE-BLOCK-END
