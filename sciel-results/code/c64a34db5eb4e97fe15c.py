"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    def factorize(n):
        """Return dict of prime->exponent for n's factorization within our primes."""
        exps = {}
        for p in primes:
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt > 0:
                exps[p] = cnt
        if n != 1:
            raise ValueError(f"n={n} not fully factorized")
        return exps
    
    target_exps = factorize(target)
    
    best_max = target  # trivial: [target]
    best_solution = [target]
    
    # We'll search for solutions with factors in increasing order
    # factors[0] < factors[1] < ... < factors[k-1], all > 12
    # Product = target
    # Minimize factors[k-1] (the largest)
    
    def can_complete(remaining_exps, remaining_product, min_factor, count_needed, current_max_limit):
        """Check if we can form count_needed distinct factors, all >= min_factor,
        with product = remaining_product, and all <= current_max_limit."""
        if count_needed == 0:
            return remaining_product == 1
        if count_needed == 1:
            return remaining_product >= min_factor and remaining_product <= current_max_limit
        
        # Lower bound: product of count_needed smallest distinct ints >= min_factor
        lower_prod = 1
        f = min_factor
        for i in range(count_needed):
            lower_prod *= f
            f += 1
            if lower_prod > remaining_product:
                return False
        
        # Upper bound: product of count_needed largest distinct ints <= current_max_limit
        upper_prod = 1
        f = current_max_limit
        for i in range(count_needed):
            upper_prod *= f
            f -= 1
            if f < min_factor:
                return False
            if upper_prod < remaining_product:
                return False
        
        return True
    
    def search(remaining_exps, remaining_product, min_factor, current_factors, count_used, max_count):
        nonlocal best_max, best_solution
        
        if time.time() > deadline:
            return
        
        if count_used == max_count:
            if remaining_product == 1:
                sol_max = max(current_factors) if current_factors else 0
                if sol_max < best_max:
                    best_max = sol_max
                    best_solution = list(current_factors)
            return
        
        # Remaining factors needed
        remaining_count = max_count - count_used
        
        if remaining_count == 0:
            if remaining_product == 1:
                sol_max = max(current_factors) if current_factors else 0
                if sol_max < best_max:
                    best_max = sol_max
                    best_solution = list(current_factors)
            return
        
        # The next factor must be >= min_factor and <= best_max (to improve)
        # Also, it must be a divisor of remaining_product
        
        # Generate candidate factors: divisors of remaining_product in range [min_factor, best_max]
        # But we need them to be >= min_factor and the product of remaining must work
        
        # Prune: can we complete?
        if not can_complete(remaining_exps, remaining_product, min_factor, remaining_count, best_max):
            return
        
        # Find divisors of remaining_product in [min_factor, best_max]
        # that are consistent with remaining_exps
        # A divisor d of remaining_product means for each prime p, exp_p(d) <= remaining_exps[p]
        
        # Enumerate possible values for the next factor
        # To keep it manageable, iterate through possible values
        # d must be in [min_factor, best_max] and divide remaining_product
        
        # Better: iterate through divisors
        # Get all divisors of remaining_product that are in range
        # remaining_product can be large, but we can generate divisors from prime exps
        
        def get_divisors_in_range(exps, lo, hi):
            """Generate all divisors of the number defined by exps that are in [lo, hi]."""
            divs = [1]
            for p, e in exps.items():
                new_divs = []
                for d in divs:
                    val = d
                    for i in range(e + 1):
                        if val >= lo and val <= hi:
                            new_divs.append(val)
                        val *= p
                        if val > hi:
                            break
                divs = new_divs
            return sorted(set(divs))
        
        candidates = get_divisors_in_range(remaining_exps, min_factor, best_max)
        
        for d in candidates:
            if time.time() > deadline:
                return
            
            # Check if d is valid: all prime exponents of d <= remaining_exps
            # Since d is a divisor generated from exps, this is guaranteed
            
            # New remaining
            new_exps = {}
            for p, e in remaining_exps.items():
                cnt = 0
                tmp = d
                while tmp % p == 0:
                    tmp //= p
                    cnt += 1
                new_exps[p] = e - cnt
            
            new_remaining = remaining_product // d
            current_factors.append(d)
            
            search(new_exps, new_remaining, d + 1, current_factors, count_used + 1, max_count)
            
            current_factors.pop()
    
    # Try different numbers of factors from 7 down to 1
    # More factors tend to allow smaller max
    for max_count in range(7, 0, -1):
        if time.time() > deadline:
            break
        
        # For max_count factors, the minimum possible max is at least the geometric mean
        # But let's just search
        
        # Initial min_factor is 13 (smallest > 12)
        search(dict(target_exps), target, 13, [], 0, max_count)
    
    return best_solution

# EVOLVE-BLOCK-END
