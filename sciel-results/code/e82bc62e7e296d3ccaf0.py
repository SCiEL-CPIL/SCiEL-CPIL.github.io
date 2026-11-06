"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_set = list(range(1, 20))  # {1, ..., 19}, size 19
    
    # Strategy 1: All small numbers
    # Max k such that T_k < 200
    # T_19 = 190, T_20 = 210
    # So {1, ..., 19} is valid.
    
    # Strategy 2: Small numbers + one large number
    # S = {1, ..., k} U {x}
    # Condition: 200 - x > T_k
    # Maximize k + 1
    
    for k in range(1, 20):
        t_k = k * (k + 1) // 2
        # We need 200 - x > t_k => x < 200 - t_k
        # x must be > k and <= 100
        # To maximize k, we just check if there exists such x
        # Actually, for a fixed k, any x in (k, 200 - t_k) works.
        # We want to maximize k.
        # If k=18, T_18=171. x < 29. x in {19, ..., 28}.
        # Size 18 + 1 = 19.
        
    # It seems 19 is the limit.
    # Let's verify {1, ..., 19} is valid.
    # Sum of any subset is <= 190.
    
    # Check if we can do better with non-contiguous smalls?
    # Unlikely to exceed 19 quickly.
    
    if time.time() > deadline:
        return best_set
        
    return best_set
# EVOLVE-BLOCK-END
