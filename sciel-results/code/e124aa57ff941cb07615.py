def priority(vector, n):
    # Count occurrences of 0, 1, and 2
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Ideal count is n/3
    # We want to minimize the deviation from this ideal balance.
    # This corresponds to minimizing the energy in the low-frequency 
    # Fourier modes (specifically the bias towards any single symbol).
    
    # Calculate sum of squared deviations
    dev0 = c0 - n / 3.0
    dev1 = c1 - n / 3.0
    dev2 = c2 - n / 3.0
    
    # The "spectral penalty" is the sum of squared deviations.
    # Higher balance (lower penalty) should yield higher priority.
    penalty = dev0**2 + dev1**2 + dev2**2
    
    # Add a small tie-breaker to ensure determinism without bias
    h = hash(vector)
    tie_breaker = (h % 1000000) / 1000000.0
    
    # Prioritize vectors with low spectral penalty (high balance)
    # We invert the penalty so that lower penalty -> higher priority.
    # Scaling by a large number ensures the penalty dominates the tie-breaker.
    return -penalty + tie_breaker
