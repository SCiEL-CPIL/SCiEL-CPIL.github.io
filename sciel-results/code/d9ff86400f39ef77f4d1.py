"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Approach:
    1. Iterate through primes P in increasing order.
    2. For each P, generate all P-smooth numbers in the range [10000, 1000001].
    3. Check if there are any consecutive integers in this set.
    4. If a pair (n, n+1) is found, n is a valid candidate with largest prime factor <= P.
    5. Since we iterate P in increasing order, the first valid n found minimizes the score.
    """
    # Primes to check. We can stop if we've checked enough.
    # The smallest prime factor of any n in [10000, 1000000] is at most sqrt(1000000)=1000 if n is prime,
    # but we are looking for smooth numbers. 
    # Let's generate primes up to a reasonable limit, e.g., 100.
    # Actually, we can just generate primes on the fly or use a static list.
    
    # Static list of small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    LOW = 10000
    HIGH = 1000001 # n+1 can be up to 1000001
    
    for idx, P in enumerate(primes):
        # Generate all P-smooth numbers in [LOW, HIGH]
        # A number is P-smooth if all its prime factors are <= P.
        # We can generate these by multiplying primes <= P.
        
        # To be efficient, we can use a recursive generator or a priority queue.
        # Given the small range of primes (<= 97 initially), the number of smooth numbers is manageable.
        
        smooth_numbers = set()
        
        # Recursive generation
        def gen_smooth(primes_list, max_val, current):
            if current > max_val:
                return
            if current >= LOW:
                smooth_numbers.add(current)
            for p in primes_list:
                if current * p > max_val:
                    break
                gen_smooth(primes_list, max_val, current * p)
                
        # We need primes up to P
        current_primes = [p for p in primes if p <= P]
        
        # Start from 1
        gen_smooth(current_primes, HIGH, 1)
        
        # Check for consecutive numbers
        # Convert to sorted list to check neighbors
        sorted_smooth = sorted(smooth_numbers)
        
        # We can just iterate and check if x+1 is in the set
        # Since the set size is small (fewer than a few thousand usually for small P), this is fast.
        for n in sorted_smooth:
            if n + 1 in smooth_numbers:
                if LOW <= n <= 1000000:
                    return n
                    
    # Fallback: If no pair found with primes up to 97, we should probably find one.
    # But to be safe, if we exhaust primes, we can fall back to a simple scan or return a known good value.
    # However, it is highly likely that a solution exists for small P.
    # For example, 10000 = 2^4 * 5^4, 10001 = 73 * 137. Not smooth for P=97? 137 > 97.
    # Let's trust the loop finds it. If not, return 10000 as a safe default (though not optimal).
    return 10000
# EVOLVE-BLOCK-END
