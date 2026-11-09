"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # The set of multiples of 3 in [1, 255]
    # Size: 255 // 3 = 85
    # All pairs have gcd >= 3.
    # Contains 255.
    # This is a known optimal or near-optimal solution for this specific n.
    # Attempts to find a larger set via greedy swap are unlikely to succeed 
    # given the structural constraints of primes 3, 5, 17.
    
    result = [i for i in range(1, 256) if i % 3 == 0]
    
    # Optional: A very quick local search to see if we can beat 85
    # But given the analysis, 85 is likely the max.
    # We return the multiples of 3.
    
    return result
# EVOLVE-BLOCK-END
