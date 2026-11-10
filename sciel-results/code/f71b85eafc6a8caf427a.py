import sys
import math

# Increase recursion limit just in case, though not strictly needed for this iterative approach
sys.setrecursionlimit(2000)

def construct():
    # The range is [10000, 1000000]
    # We need lpf for n and n+1, so we need lpf up to 1000001
    max_val = 1000001
    
    # Initialize largest prime factor array
    # lpf[i] will store the largest prime factor of i
    lpf = list(range(max_val + 1))
    
    # Sieve of Eratosthenes variant to find largest prime factor
    # We only need to iterate up to sqrt(max_val) to find primes, 
    # but we must update multiples.
    # Actually, standard sieve logic: 
    # for i from 2 to max_val:
    #   if lpf[i] == i: # i is prime
    #     for j in range(i, max_val+1, i):
    #       lpf[j] = i
    # This works because we iterate i in increasing order, so the last prime 
    # that divides j is the largest prime factor.
    
    for i in range(2, max_val + 1):
        if lpf[i] == i:  # i is a prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    min_lpf = float('inf')
    best_n = 10000
    
    # We need n in [10000, 1000000]
    # The product is n(n+1).
    # The largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
    # Note: n+1 can be up to 1000001, which is within our array bounds.
    
    # Optimization: Instead of scanning all n, we can use the fact that
    # we want to minimize max(lpf[n], lpf[n+1]).
    # The current approach is already O(N) for the scan after O(N log log N) sieve.
    # Let's keep the scan but ensure it's efficient.
    
    for n in range(10000, 1000001):
        # Calculate the largest prime factor of the product
        current_max_lpf = lpf[n] if lpf[n] > lpf[n + 1] else lpf[n + 1]
        
        if current_max_lpf < min_lpf:
            min_lpf = current_max_lpf
            best_n = n
            
    return best_n

# Keep the entry point exactly as requested
# The function name is construct
# It returns an integer

if __name__ == "__main__":
    # For local testing
    print(construct())
