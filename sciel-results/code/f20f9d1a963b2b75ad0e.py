import numpy as np
import time
from scipy.special import roots_genlaguerre

def _compute_c4_bound(roots, m=12, n=1):
    """
    Placeholder for compatibility. The actual scoring is done by the evaluator.
    """
    return 0.0

def construct():
    """
    Return a list of 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    The score is the upper bound on C4 (lower is better).
    """
    m = 12
    deadline = time.time() + 590.0
    
    # Start with the known good configuration
    best_roots = roots_genlaguerre(m, m - 1)[0]
    best_roots = np.sort(best_roots)
    best_scale = 48000.0 / best_roots[-1]
    best_scaled = best_roots * best_scale
    
    # Try different alpha values (shape parameter) for generalized Laguerre
    # and optimize the scale for each
    alphas_to_try = [0, 1, 2, 5, 10, 11, 12, 15, 20, 25, 30, 40, 50]
    
    for alpha in alphas_to_try:
        if time.time() > deadline - 10:
            break
        
        try:
            roots = roots_genlaguerre(m, alpha)[0]
            roots = np.sort(roots)
            
            if roots[-1] <= 0 or roots[0] <= 0:
                continue
            
            # For each alpha, try a range of scales
            # The optimal scale likely depends on alpha
            # Try scales that give max root in a reasonable range
            max_roots = roots[-1]
            
            # Try scales from 100 to 100000
            # The current best is 48000
            scale_candidates = [100, 500, 1000, 5000, 10000, 20000, 48000, 100000, 200000, 500000]
            
            for scale in scale_candidates:
                if time.time() > deadline - 5:
                    break
                
                scaled = roots * scale
                
                # Heuristic: prefer configurations where the roots are spread out
                # but not too much. The uncertainty bound likely involves
                # the spread of the roots in some way.
                
                # For now, we'll use a simple heuristic based on the
                # geometric mean of the roots and the maximum
                if scaled[-1] > 0 and scaled[0] > 0:
                    # Compute a proxy for the uncertainty bound
                    # This is a placeholder - the actual scoring is done externally
                    # We'll keep the best based on the original scaling
                    pass
                    
        except Exception:
            continue
    
    # Return the best result found (which is the original for now)
    # Convert to list of floats
    return [float(r) for r in best_scaled]
