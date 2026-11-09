import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    N = math.factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1

    # Generate all divisors of N using prime factorization
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    divisors = set()
    for a in range(11):  # 2^a
        for b in range(6):  # 3^b
            for c in range(3):  # 5^c
                for d in range(2):  # 7^d
                    for e in range(2):  # 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)
    
    # Filter: must be > 12, sort
    cands = sorted([d for d in divisors if d > 12])
    
    # Meet-in-the-middle: split cands into two halves
    # For each half, generate all subsets (with product tracking)
    # Then join subsets from both halves where product1 * product2 == N
    # and all factors are distinct and > 12
    
    # Actually, let's use a smarter approach:
    # Since we want to minimize the largest factor, we can iterate M from small to large
    # and check if a valid decomposition exists with all factors <= M.
    
    # Precompute for each candidate divisor, its index
    # Use DFS with pruning, but with better ordering and early termination
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    # For a given max_val M, try to find a decomposition of N into distinct factors
    # all in (12, M], with at most 12 factors
    def find_decomposition(max_val):
        if time.time() > deadline:
            return None
        
        # Filter candidates <= max_val
        # Find the rightmost index where cands[i] <= max_val
        idx = bisect.bisect_right(cands, max_val) - 1
        if idx < 0:
            return None
        
        # DFS: pick factors in increasing order to ensure distinctness
        # State: remaining product, start index, current factors list
        def dfs(remaining, start_idx, factors):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                if 1 <= len(factors) <= 12:
                    return factors[:]
                return None
            
            if len(factors) >= 12:
                return None
            
            # If remaining is prime and > 12 and <= max_val and not already used
            if is_prime(remaining):
                if remaining > 12 and remaining <= max_val:
                    # Check if remaining is not already in factors
                    if remaining not in factors:
                        factors.append(remaining)
                        if len(factors) <= 12:
                            return factors[:]
                        factors.pop()
                return None
            
            # Try each candidate from start_idx to idx
            for i in range(start_idx, idx + 1):
                d = cands[i]
                if d > remaining:
                    break
                if remaining % d != 0:
                    continue
                
                # Check distinctness: d not in factors
                if d in factors:
                    continue
                
                factors.append(d)
                result = dfs(remaining // d, i + 1, factors)
                if result is not None:
                    return result
                factors.pop()
            
            return None
        
        return dfs(N, 0, [])

    # Iterate M from smallest candidate to largest
    for i, M in enumerate(cands):
        if time.time() > deadline:
            break
        result = find_decomposition(M)
        if result is not None:
            return result
    
    # Fallback: return [N]
    return [N]
