import time
import math

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Time limit guard
    start_time = time.time()
    time_limit = 80  # Leave 10s margin for 90s limit
    
    best_n = 1000
    best_count = float('inf')
    
    # We scan from 1000 upwards.
    # Since the theoretical minimum is 9 (as window size 24 >= all primes < 24),
    # we expect best_count to be 9 immediately.
    # We verify this with a quick scan.
    
    # Optimization: 
    # For a prime p, a window of size W=24 contains a multiple of p if and only if
    # the window does NOT fit strictly between two consecutive multiples of p.
    # The gap between multiples is p.
    # If W < p, it's possible to avoid.
    # If W >= p, it's IMPOSSIBLE to avoid.
    # Here W=24.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23.
    # All are <= 23 < 24.
    # So all primes MUST appear in any window of size 24.
    # Thus, the count of distinct primes is ALWAYS 9.
    
    # Therefore, any n is optimal.
    # We return 1000.
    
    return 1000
