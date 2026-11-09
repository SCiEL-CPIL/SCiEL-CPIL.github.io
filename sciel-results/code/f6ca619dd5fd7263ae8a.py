"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 90.0
    target = Fraction(5, 31)
    
    best_max_denom = 10**7
    best_solution = []

    def find_2_term_completion(p, q, min_denom, max_denom_limit):
        """
        Find b, c such that 1/b + 1/c = p/q.
        Returns list of (b, c) or empty.
        Assumes p, q > 0 and gcd(p,q)=1? No, Fraction handles that.
        We use integer arithmetic.
        (pb - q)(pc - q) = q^2
        X = pb - q, Y = pc - q.
        X * Y = q^2.
        X = -q (mod p)
        Y = -q (mod p)
        b = (X + q) / p
        c = (Y + q) / p
        """
        if p <= 0 or q <= 0:
            return []
        
        q2 = q * q
        # Iterate divisors of q2
        # Optimization: only iterate up to sqrt(q2)
        limit = int(q2 ** 0.5)
        solutions = []
        
        # We need X such that X | q2 and X % p == (-q) % p
        target_mod = (-q) % p
        
        for x in range(1, limit + 1):
            if q2 % x != 0:
                continue
            if x % p != target_mod:
                continue
            y = q2 // x
            if y % p != target_mod:
                continue
            
            # Check if b and c are integers
            if (x + q) % p != 0:
                continue
            if (y + q) % p != 0:
                continue
            
            b = (x + q) // p
            c = (y + q) // p
            
            # Check bounds and distinctness
            if b < min_denom or c < min_denom:
                continue
            if b > max_denom_limit or c > max_denom_limit:
                continue
            if b == c:
                continue
            # We want distinct denominators in the final set.
            # The caller ensures a is distinct from b, c.
            # b and c are distinct here.
            solutions.append((b, c))
            
        return solutions

    # Search for 3-term solutions
    # a is the smallest denominator
    # 1/a < 5/31 => a > 31/5 = 6.2 => a >= 7
    for a in range(7, 1000):
        if time.time() > deadline - 5.0:
            break
            
        r = target - Fraction(1, a)
        if r <= 0:
            break
        
        p = r.numerator
        q = r.denominator
        
        # We want b, c > a to ensure a is the smallest? 
        # Not strictly required, but helps avoid permutations.
        # If we don't enforce b > a, we might find {b, a, c} which is same set.
        # Enforcing b > a and c > a ensures we find each set once with a as min.
        # However, we want to minimize MAX. 
        # If we find a set, we record max.
        
        # Limit for b, c: we want max < best_max_denom
        # So b < best_max_denom and c < best_max_denom
        max_limit = best_max_denom - 1
        
        completions = find_2_term_completion(p, q, a + 1, max_limit)
        
        for b, c in completions:
            current_max = max(a, b, c)
            if current_max < best_max_denom:
                best_max_denom = current_max
                best_solution = [a, b, c]
                
        # Also try 4-term? 
        # If 3-term doesn't improve, 4-term might.
        # But 3-term search is fast.
        # Let's also check if we can complete with 1 term (2-term total)?
        # No, 2-term total means a is part of 2-term.
        # 5/31 = 1/a + 1/b.
        # We checked this is impossible.
        
        # Check 4-term: 5/31 = 1/a + 1/b + 1/c + 1/d
        # This is more expensive. We'll skip unless 3-term max is large.
        # Given 434 is found, it's small.
        
    # If no 3-term solution found, fallback to greedy
    if not best_solution:
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values

    return best_solution

# EVOLVE-BLOCK-END
