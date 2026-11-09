"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    # We need a nonempty subset of [1, 255] containing 255,
    # such that gcd(a, b) > 1 for all distinct pairs. Maximize cardinality.
    #
    # 255 = 3 * 5 * 17.
    # Analysis shows that the set of multiples of the smallest prime factor
    # (3) provides the maximum cardinality.
    # Size = floor(255/3) = 85.
    # Mixing with other prime factors reduces the total size significantly.
    
    if time.time() > deadline:
        return [255]
        
    result = [i for i in range(1, 256) if i % 3 == 0]
    
    return result
# EVOLVE-BLOCK-END
