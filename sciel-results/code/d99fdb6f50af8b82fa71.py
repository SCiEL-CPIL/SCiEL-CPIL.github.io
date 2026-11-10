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
    
    # Optimization: Only iterate up to sqrt(max_val) for the outer loop
    # but we still need to mark composites. 
    # However, the standard sieve for largest prime factor requires 
    # iterating through all potential primes. 
    # A more efficient way is to only iterate i up to sqrt(max_val) 
    # and then handle the remaining primes? No, that's not correct for 
    # finding the LARGEST prime factor directly in one pass.
    # 
    # Let's stick to the standard sieve but optimize the inner loop slightly.
    # Actually, the current sieve is O(N log log N) which is fast enough for N=1e6.
    # The main bottleneck is the Python loop overhead.
    
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
    
    for n in range(10000, 1000001):
        # Calculate the largest prime factor of the product
        current_max_lpf = max(lpf[n], lpf[n + 1])
        
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
