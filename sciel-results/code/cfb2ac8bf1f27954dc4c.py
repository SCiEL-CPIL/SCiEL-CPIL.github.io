"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        g = hi - lo
        
        # If gap is 2, only one integer m.
        # Condition: LPF(m) >= 2.
        # All integers >= 2 have LPF >= 2.
        # So if g=2, the single integer is always a witness.
        # If g > 2, we search.
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m has any divisor d in [2, g-1]
            # If yes, LPF(m) < g.
            # If no, LPF(m) >= g.
            valid = True
            # Optimization: only check up to g-1
            # Since g is small, this is fast.
            for d in range(2, g):
                if m % d == 0:
                    valid = False
                    break
            
            if valid:
                result[i] = m
                found = True
                break
                
        if found:
            continue
        else:
            result[i] = None
            
    return result
# EVOLVE-BLOCK-END
