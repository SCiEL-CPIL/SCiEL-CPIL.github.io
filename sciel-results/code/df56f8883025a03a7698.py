import time
from functools import lru_cache

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    primes = [2, 3, 5, 7, 11]
    target = (10, 5, 2, 1, 1)  # prime exponents of 12!
    
    def get_pf(n):
        """Return prime exponent tuple for n, or None if n has prime factor > 11."""
        pf = [0, 0, 0, 0, 0]
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                pf[i] += 1
                temp //= p
        if temp != 1:
            return None
        return tuple(pf)
    
    def check(M):
        """Check if 12! can be written as product of distinct integers in (12, M]."""
        # DP: set of reachable exponent vectors
        # State: frozenset of exponent tuples
        # Start: {zero vector}
        # For each candidate x in 13..M: can we add x to any existing state?
        
        # Use a set of tuples representing reachable exponent vectors
        reachable = {tuple(0 for _ in range(5))}
        
        for x in range(13, M + 1):
            if time.time() - start_time > 80:
                return False
            pf = get_pf(x)
            if pf is None:
                continue
            # Try adding x to each existing state
            new_states = set()
            for state in reachable:
                new_state = tuple(state[i] + pf[i] for i in range(5))
                # Check if new_state <= target component-wise
                valid = True
                for i in range(5):
                    if new_state[i] > target[i]:
                        valid = False
                        break
                if valid:
                    new_states.add(new_state)
            reachable = reachable | new_states
            
            if target in reachable:
                return True
        
        return target in reachable
    
    # Binary search on M
    # Lower bound: at least 13 (one factor of 13, rest must make up 12!/13)
    # Upper bound: N itself (single factor)
    lo = 13
    hi = N
    
    # First, find a feasible upper bound
    # We know M=N works (single factor)
    # Let's try to find the minimum M
    
    # Actually, let's think about lower bounds more carefully
    # If all factors are <= M, and we need product = 12!, 
    # and factors are distinct integers > 12
    # Minimum number of factors: 1 (just 12! itself)
    # Maximum: 12
    
    # For a given M, the maximum product of k distinct integers from {13,...,M} 
    # is M*(M-1)*...*(M-k+1)
    # We need at least one k where this >= 12!
    
    # Let's just binary search
    best_M = N
    best_factors = [N]
    
    # Try small values of M first to get a good upper bound quickly
    # Then binary search
    for M in range(20, 60):
        if time.time() - start_time > 70:
            break
        if check(M):
            best_M = M
            break
    
    # Binary search between lo and best_M
    if best_M < N:
        lo = 13
        hi = best_M
        while lo < hi:
            if time.time() - start_time > 75:
                break
            mid = (lo + hi) // 2
            if check(mid):
                hi = mid
            else:
                lo = mid + 1
        best_M = lo
    
    # Now we need to actually FIND the factors, not just verify existence
    # Use backtracking with the DP information
    def find_factors(M):
        """Find distinct factors in (12, M] whose product is 12!."""
        candidates = list(range(13, M + 1))
        # Filter to only those with valid prime factorizations
        valid_candidates = []
        for x in candidates:
            pf = get_pf(x)
            if pf is not None:
                valid_candidates.append((x, pf))
        
        # Backtracking search
        result = []
        
        def backtrack(start_idx, remaining_pf, used):
            if time.time() - start_time > 82:
                return False
            
            if remaining_pf == (0, 0, 0, 0, 0):
                return True
            
            # Check if remaining product is achievable
            rem_val = 1
            for i in range(5):
                rem_val *= primes[i] ** remaining_pf[i]
            
            # If rem_val == 1, we're done (handled above)
            # If rem_val > 1, we need at least one more factor
            
            for i in range(start_idx, len(valid_candidates)):
                x, pf = valid_candidates[i]
                # Check if pf <= remaining_pf
                can_use = True
                for j in range(5):
                    if pf[j] > remaining_pf[j]:
                        can_use = False
                        break
                if not can_use:
                    continue
                
                new_remaining = tuple(remaining_pf[j] - pf[j] for j in range(5))
                result.append(x)
                if backtrack(i + 1, new_remaining, used | (1 << i)):
                    return True
                result.pop()
            
            return False
        
        if backtrack(0, target, 0):
            return result
        return None
    
    factors = find_factors(best_M)
    if factors is None:
        # Fallback: try larger M
        for M in range(best_M + 1, min(best_M + 50, N + 1)):
            if time.time() - start_time > 83:
                break
            factors = find_factors(M)
            if factors is not None:
                best_M = M
                break
    
    if factors is None or not factors:
        # Last resort: return [N]
        return [N]
    
    # Verify
    prod = 1
    for f in factors:
        prod *= f
    assert prod == N, f"Product {prod} != {N}"
    assert all(f > 12 for f in factors)
    assert len(set(factors)) == len(factors), "Factors not distinct"
    assert 1 <= len(factors) <= 12
    
    return factors
