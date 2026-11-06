import math
import time

# Precompute 12!
T12 = math.factorial(12)

# Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
exponents = [10, 5, 2, 1, 1]

def generate_divisors():
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * p_power)
            p_power *= p
        divisors = new_divisors
    return divisors

ALL_DIVISORS = generate_divisors()
# Filter divisors > 12
VALID_DIVISORS = sorted([d for d in ALL_DIVISORS if d > 12])

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # We iterate M from smallest to largest
    # M must be a divisor of 12! and > 12
    # We want the smallest M such that 12!/M can be decomposed
    # into distinct factors d where 12 < d < M.
    
    for M in VALID_DIVISORS:
        if time.time() > deadline:
            return [T12]
            
        rem = T12 // M
        
        if rem == 1:
            return [M]
            
        # Candidates must be:
        # 1. Divisors of 12!
        # 2. > 12
        # 3. < M
        # 4. Divisors of rem
        
        candidates = []
        for d in VALID_DIVISORS:
            if d >= M:
                break
            if rem % d == 0:
                candidates.append(d)
                
        # If rem > 1, we need at least one candidate.
        if not candidates:
            continue
            
        # Sort candidates in descending order for backtracking
        candidates.sort(reverse=True)
        
        # Backtracking search
        # State: current_rem, max_allowed_factor
        # We want to find a sequence of factors f1 > f2 > ... > fk
        # such that f1 < M, f1 | rem, f2 | rem/f1, etc.
        # Product of all fi == rem.
        
        def can_decompose(current_rem, max_allowed):
            if current_rem == 1:
                return True
            
            # Pruning: 
            # If the largest candidate is greater than current_rem, we can skip it?
            # Actually, we iterate candidates.
            
            # Optimization: 
            # If current_rem is prime, it must be in candidates.
            # Check if current_rem is in candidates and < max_allowed.
            # But current_rem might be composite.
            
            # Lower bound check:
            # The minimum factor we can pick is > 12.
            # If current_rem < 13, and current_rem != 1, fail.
            if current_rem < 13:
                return False
                
            for c in candidates:
                if c >= max_allowed:
                    continue
                if c > current_rem:
                    continue
                if current_rem % c == 0:
                    if can_decompose(current_rem // c, c):
                        return True
            return False

        if can_decompose(rem, M):
            # We need to retrieve the actual factors, not just True/False
            # Let's modify to return the list
            
            def find_decomp(current_rem, max_allowed):
                if current_rem == 1:
                    return []
                if current_rem < 13:
                    return None
                    
                for c in candidates:
                    if c >= max_allowed:
                        continue
                    if c > current_rem:
                        continue
                    if current_rem % c == 0:
                        res = find_decomp(current_rem // c, c)
                        if res is not None:
                            return [c] + res
                return None

            decomp = find_decomp(rem, M)
            if decomp is not None:
                return decomp + [M]
                
    return [T12]
