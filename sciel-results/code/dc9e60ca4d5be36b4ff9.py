import time
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # 5 seconds margin from 90s limit
    
    target = 123457
    
    # Primes to check, in increasing order of max prime factor
    # We know k=11 works (score 58.45), so we check smaller ones first
    prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13], # Fallback just in case
        [2, 3, 5, 7, 11, 13, 17]
    ]
    
    def generate_smooth_numbers(limit, primes):
        """
        Generate all numbers <= limit whose prime factors are in primes.
        Returns a set of such numbers.
        """
        smooth = set([1])
        # Recursive generation or iterative expansion
        # Iterative: start with 1, multiply by primes, add if <= limit
        # To avoid duplicates and ensure completeness, we can use a queue or recursion.
        # Given the small size, a simple recursive DFS is fine.
        
        def dfs(current, idx):
            if current > limit:
                return
            smooth.add(current)
            for i in range(idx, len(primes)):
                # Multiply by primes[i]
                # Check for overflow (not needed for small limits, but good practice)
                if current > limit // primes[i]:
                    break
                dfs(current * primes[i], i)
        
        dfs(1, 0)
        return smooth

    best_pair = None
    best_val = float('inf')
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        max_p = primes[-1]
        S = generate_smooth_numbers(target, primes)
        
        # Check if any s in S has target - s in S
        # And target - s > 0
        found = False
        for s in S:
            t = target - s
            if t > 0 and t in S:
                # Found a pair
                # We want to minimize max(P(a), P(b)).
                # Since all numbers in S have P(n) <= max_p, the max is <= max_p.
                # Is it exactly max_p?
                # We iterate max_p in increasing order, so the first found is minimal.
                best_pair = [s, t]
                best_val = max_p
                found = True
                break
        
        if found:
            break

    # If we didn't find a pair in the pre-defined sets (unlikely for k=11),
    # we should try to find a better pair than the seed [1, 123456] (val 643).
    # But the problem states k=11 is achievable.
    # If best_pair is None, we fall back to a simple search or the seed.
    if best_pair is None:
        # Fallback: try to find any pair with small max prime factor
        # Or just return the seed if time is up, but seed is bad.
        # Let's try a quick scan for small primes if we didn't find one?
        # Actually, if k=11 failed, it's a bug in my logic or the observation.
        # But I am confident in the logic.
        # Let's ensure we return something valid.
        best_pair = [1, target - 1]
        
    return best_pair
