"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    deadline = time.time() + 135  # 135s deadline, 15s safety margin
    
    # Start with a known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = len(best_k)
    
    def _sigma_sieve(n):
        """Compute sigma(k) for k=1..n using a sieve."""
        sigma = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(i, n + 1, i):
                sigma[j] += i
        return sigma
    
    # Try increasing ranges, but be smart about it
    # The key insight: M = k * sigma(k), and we want many k with the same M
    # For k up to N, M can be up to ~N * N * log(N) roughly
    # We need M <= 10^18, so N can be quite large
    
    # Strategy: Use the sieve approach but with early termination
    # and focus on finding collisions
    
    ranges = [10**5, 2*10**5, 5*10**5, 10**6, 2*10**6, 5*10**6, 10**7]
    
    for N in ranges:
        if time.time() > deadline - 30:
            break
        
        # Check if N is too large: max M would be around N * sigma(N)
        # sigma(N) can be up to ~N * log(log(N)) for highly composite numbers
        # So max M ~ N^2 * log(log(N))
        # For N = 10^7, max M ~ 10^14 * ~5 = 5*10^14, well within 10^18
        # For N = 10^8, max M ~ 10^16 * ~6 = 6*10^16, still within 10^18
        # But the sieve would take too long for N = 10^8
        
        try:
            sigma = _sigma_sieve(N)
        except MemoryError:
            break
        
        # Group by M value
        groups = defaultdict(list)
        for k in range(1, N + 1):
            M = k * sigma[k]
            if M <= 10**18:
                groups[M].append(k)
        
        # Find the best M with most solutions
        for M, ks in groups.items():
            count = len(ks)
            if count > best_count:
                best_M = M
                best_k = ks
                best_count = count
        
        # Early exit if we find a very good solution
        if best_count >= 30:
            break
    
    # If we still have time, try a more targeted approach
    # Look for M values that are products of many k*sigma(k)
    if time.time() < deadline - 40:
        # Try to extend the search with a different strategy
        # For highly composite M, many k might satisfy k*sigma(k)=M
        # Try M values that are products of small primes raised to high powers
        
        # Actually, let's try a different approach:
        # For each k, M = k * sigma(k). If we can find many k with the same M,
        # we win. Let's try to be smarter about which k to check.
        
        # Alternative: iterate over possible M values and find k such that k*sigma(k)=M
        # This is harder, so let's stick with the sieve approach but optimize it.
        
        # Try a larger N if we have time
        if time.time() < deadline - 50:
            try:
                N2 = 2 * 10**7
                sigma = _sigma_sieve(N2)
                groups = defaultdict(list)
                for k in range(1, N2 + 1):
                    M = k * sigma[k]
                    if M <= 10**18:
                        groups[M].append(k)
                
                for M, ks in groups.items():
                    count = len(ks)
                    if count > best_count:
                        best_M = M
                        best_k = ks
                        best_count = count
            except MemoryError:
                pass
    
    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Verify the solution
    valid = True
    for k in best_k:
        if k < 1 or k > math.isqrt(best_M):
            valid = False
            break
    if not valid:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
