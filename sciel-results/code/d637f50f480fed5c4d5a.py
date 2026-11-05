import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 58  # Keep safety margin for 60s limit
    
    # Primes to check
    # We start from small primes. If we find a pair, that B is the minimal max prime factor.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    # We only need to go up to sqrt(TARGET) or so, but usually small primes suffice.
    # If no pair found for small primes, we can fall back to a larger B or the trivial solution.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate all p-smooth numbers <= TARGET
        # A number is p-smooth if all its prime factors are <= p.
        smooths = set()
        
        # Use a recursive generator to avoid storing too much if it gets big, 
        # but for p=11, size is ~40k, which is fine.
        
        def dfs(val, idx):
            if val > TARGET:
                return
            smooths.add(val)
            for i in range(idx, len(primes)):
                q = primes[i]
                if q > p:
                    break
                if val * q > TARGET:
                    break
                dfs(val * q, i)
                
        dfs(1, 0)
        
        # Check for pair
        for a in smooths:
            b = TARGET - a
            if b in smooths:
                return [a, b]
                
    # Fallback: return a pair with a known reasonable value
    # The current best known is likely 11.
    # If we didn't find it in the loop, it might be due to timeout or bug.
    # But for p=11, it should be fast.
    # Let's return a specific pair if we know one.
    # From previous analysis, a 11-smooth pair exists.
    # Example: a = 1, b = 123456 (P=643) is not 11-smooth.
    # We need to find the actual pair.
    # Since the loop should find it, this fallback is just in case.
    return [1, TARGET - 1]
