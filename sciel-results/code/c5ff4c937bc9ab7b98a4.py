import math
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0

    # Generate primes up to N using a simple sieve
    # We need primes to define K-smooth sets
    # Since we expect small K, we don't need primes up to N immediately,
    # but for binary search upper bound, we might need to know primes exist.
    # However, we can just binary search on integer K, and generate primes <= K on the fly.
    
    def get_primes_up_to(limit):
        if limit < 2:
            return []
        is_prime = [True] * (limit + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, val in enumerate(is_prime) if val]

    def generate_smooth_numbers(primes, limit):
        """
        Generate all numbers <= limit whose prime factors are in 'primes'.
        Returns a set of such numbers.
        """
        if not primes:
            return {1}
        
        smooth = {1}
        # We process primes one by one, expanding the set
        for p in primes:
            new_smooth = set()
            # For each existing smooth number, multiply by p^k
            # We can do this by iterating through a copy of smooth
            # or using a queue.
            # Simple approach:
            current_smooth = list(smooth)
            for s in current_smooth:
                val = s
                while val * p <= limit:
                    val *= p
                    new_smooth.add(val)
            smooth.update(new_smooth)
            
        return smooth

    def exists_pair(K):
        """
        Check if there exist a, b such that a+b=N and both are K-smooth.
        Returns (True, a, b) if found, else (False, 0, 0).
        """
        primes = get_primes_up_to(K)
        if not primes:
            # K < 2, only 1 is smooth. 1+1=2 != N (unless N=2)
            if N == 2:
                return True, 1, 1
            return False, 0, 0
            
        smooth_set = generate_smooth_numbers(primes, N)
        
        # Check for pair
        for a in smooth_set:
            b = N - a
            if b in smooth_set:
                return True, a, b
                
        return False, 0, 0

    # Binary search for the smallest K
    # Lower bound: 1 (P(1)=1)
    # Upper bound: N (worst case)
    lo = 1
    hi = N
    
    best_a = 1
    best_b = N - 1
    found = False
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        exists, a, b = exists_pair(mid)
        
        if exists:
            best_a = a
            best_b = b
            found = True
            hi = mid - 1
        else:
            lo = mid + 1
            
    if not found:
        # Fallback to a valid pair, e.g., 1 and N-1
        # This case should not happen for N > 2 as K=N always works
        # (1 is smooth, N-1 is N-1-smooth? No, N-1 is (N-1)-smooth.
        # If K=N, primes <= N include all primes of N-1. So N-1 is K-smooth.
        # So a=1, b=N-1 is always a valid pair for K=N.
        best_a = 1
        best_b = N - 1

    return [best_a, best_b]
