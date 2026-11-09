import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = factorial(12)
    
    # We want to find distinct factors > 12, product N, minimize max factor.
    # We iterate max_factor from 13 upwards.
    # For each max_factor, we check if a valid decomposition exists.
    
    def find_decomposition(remaining, min_factor, max_factor, current_factors):
        # If remaining is 1, we found a valid set
        if remaining == 1:
            return current_factors[:]
        
        # Pruning: if remaining < min_factor, we can't form it with factors >= min_factor
        # unless remaining is 1 (handled above). 
        # Actually, if remaining > 1, we must pick at least one factor f >= min_factor.
        # So remaining must be >= min_factor.
        if remaining < min_factor:
            return None
            
        # Try factors from min_factor to max_factor
        # Optimization: If remaining is prime and > max_factor, fail.
        # But checking primality is expensive. Just loop.
        
        for f in range(min_factor, max_factor + 1):
            # If f doesn't divide remaining, skip
            if remaining % f != 0:
                continue
                
            # Additional pruning: The product of the smallest possible remaining factors
            # must be able to reach 'remaining // f'.
            # Not strictly necessary for this small N, but good practice.
            
            current_factors.append(f)
            result = find_decomposition(remaining // f, f + 1, max_factor, current_factors)
            if result is not None:
                return result
            current_factors.pop()
            
        return None

    # Lower bound for max_factor:
    # We need at least one factor > 12.
    # The product of all available factors {13, ..., M} must be >= N?
    # No, we can use at most 12 factors.
    # Let's just start from 13.
    
    for M in range(13, 5000): # Upper bound guess
        if time.time() > deadline:
            break
        # Check if M is large enough to potentially hold the factors
        # Quick check: product of largest 12 factors <= M must be >= N?
        # If not, M is too small.
        # Largest 12 factors are M, M-1, ..., M-11.
        # Product P = M * (M-1) * ... * (M-11)
        # If P < N, then no subset of {13..M} can multiply to N.
        # This is a valid necessary condition.
        
        # Calculate max possible product with 12 factors <= M
        # Note: factors must be > 12.
        # If M < 13, impossible.
        # If M is small, the set {13..M} is small.
        
        # Let's compute max_prod
        max_prod = 1
        for i in range(12):
            val = M - i
            if val <= 12:
                break
            max_prod *= val
            if max_prod > N:
                break
        
        if max_prod < N:
            continue
            
        # Try to find decomposition
        factors = find_decomposition(N, 13, M, [])
        if factors is not None:
            # Verify
            if len(factors) > 12:
                continue
            if any(f <= 12 for f in factors):
                continue
            if len(set(factors)) != len(factors):
                continue
            prod = 1
            for f in factors:
                prod *= f
            if prod != N:
                continue
                
            return factors
            
    # Fallback: return the single factor 12!
    return [N]
