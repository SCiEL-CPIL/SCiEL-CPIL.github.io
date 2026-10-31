import sys
import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Fast path for small n or to ensure speed
    c0 = 0
    c1 = 0
    c2 = 0
    
    # Unroll loop for speed? No, standard loop is fine for n<=20
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    target = n / 3.0
    # Penalty for deviation from balance
    # We want to maximize this, so we use negative squared error
    balance_penalty = (c0 - target) * (c0 - target) + \
                      (c1 - target) * (c1 - target) + \
                      (c2 - target) * (c2 - target)
    
    # Roughness: sum of squared differences between adjacent elements
    # This penalizes "jagged" vectors
    roughness = 0
    prev = vector[0] if n > 0 else 0
    for i in range(1, n):
        curr = vector[i]
        diff = curr - prev
        roughness += diff * diff
        prev = curr
        
    # Combine scores
    # Balance is the primary driver.
    # Roughness is a secondary tie-breaker/smoothness filter.
    # Scale roughness down so it doesn't dominate balance completely
    return -balance_penalty - 0.01 * roughness
# EVOLVE-BLOCK-END
