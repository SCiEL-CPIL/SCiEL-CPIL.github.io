import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    target = math.factorial(12)
    
    # Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    # Precompute a set for quick lookup
    divisor_set = set(valid_divisors)
    
    # Prime factorization of target for better pruning
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    def prime_factorization(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    target_pf = prime_factorization(target)
    
    # Check if a number's prime factorization is a subset of the remaining factorization
    def can_factor(rem, remaining_pf):
        """Check if rem can be factored using only primes in remaining_pf with sufficient exponents."""
        temp = rem
        for p, exp in remaining_pf.items():
            while temp % p == 0:
                temp //= p
                if exp == 0:
                    return False
                exp -= 1
        return temp == 1
    
    # DFS with better pruning
    def dfs(rem, start_idx, max_limit, current_factors, remaining_pf):
        if time.time() > deadline:
            return False
            
        if rem == 1:
            return True
        if len(current_factors) >= 12:
            return False
            
        # Pruning: if rem itself is a valid divisor and we can use it
        if rem <= max_limit and rem in divisor_set and can_factor(rem, remaining_pf):
            # Check if rem is not already used
            if rem not in current_factors:
                current_factors.append(rem)
                # Update remaining_pf
                new_pf = dict(remaining_pf)
                for p, exp in prime_factorization(rem).items():
                    if p in new_pf:
                        new_pf[p] -= exp
                        if new_pf[p] < 0:
                            current_factors.pop()
                            return False
                if dfs(1, start_idx + 1, max_limit, current_factors, new_pf):
                    return True
                current_factors.pop()
        
        for i in range(start_idx, len(valid_divisors)):
            d = valid_divisors[i]
            if d > max_limit:
                break
            if d > rem:
                break
            if rem % d != 0:
                continue
            if d in current_factors:
                continue
                
            # Check if d can be factored from remaining_pf
            d_pf = prime_factorization(d)
            valid = True
            for p, exp in d_pf.items():
                if p not in remaining_pf or remaining_pf[p] < exp:
                    valid = False
                    break
            if not valid:
                continue
                
            # Pick d
            current_factors.append(d)
            new_pf = dict(remaining_pf)
            for p, exp in d_pf.items():
                new_pf[p] -= exp
            if dfs(rem // d, i + 1, max_limit, current_factors, new_pf):
                return True
            current_factors.pop()
                
        return False
    
    # Binary search for the minimum M
    # Lower bound: at least 13, and (12!)^(1/12) ≈ 40
    lo = 13
    hi = target
    
    # First, find a valid upper bound
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        factors = []
        if dfs(target, 0, mid, factors, target_pf):
            hi = mid
        else:
            lo = mid + 1
    
    # Return the solution found
    factors = []
    if dfs(target, 0, lo, factors, target_pf):
        return factors
    
    # Fallback: return a simple valid decomposition
    # 12! = 13 * 14 * 15 * 16 * 17 * 18 * 19 * 20 * 21 * 22 * 23 * 24 / (1*2*3*4*5*6*7*8*9*10*11*12)
    # Actually, let's just return a known valid set
    # 12! = 479001600
    # Let's try: 13*14*15*16*17*18*19*20*21*22*23*24 = ?
    # That's too large. Let's use a simpler approach.
    
    # Just return the target itself as a fallback (valid but not optimal)
    return [target]
