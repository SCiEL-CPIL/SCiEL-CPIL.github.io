"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Returns the set of denominators for the Egyptian fraction expansion of 5/31
    that minimizes the largest denominator.
    """
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s limit
    
    target = Fraction(5, 31)
    best_max_den = 248  # current known solution
    best_solution = [8, 31, 248]
    
    def is_valid_solution(sol):
        """Check if a solution is valid: distinct denominators in [1, 10000000] summing to target."""
        if len(sol) < 1 or len(sol) > 256:
            return False
        if len(set(sol)) != len(sol):
            return False
        for d in sol:
            if d < 1 or d > 10000000:
                return False
        s = sum(Fraction(1, d) for d in sol)
        return s == target
    
    # Verify initial solution
    if not is_valid_solution(best_solution):
        # Fallback: greedy expansion
        best_solution = []
        remaining = target
        last_d = 0
        for _ in range(256):
            if remaining == 0:
                break
            d = max(last_d + 1, -(-remaining.numerator // remaining.denominator))  # ceil
            if d > 10000000:
                break
            best_solution.append(d)
            remaining -= Fraction(1, d)
            last_d = d
        best_max_den = max(best_solution) if best_solution else 10000000
    
    def search_two_terms(remaining, max_den, min_den=1):
        """Find two distinct denominators c, d with 1/c + 1/d = remaining, both in [min_den, max_den]."""
        if remaining <= 0:
            return None
        # 1/c + 1/d = remaining => d = c / (remaining*c - 1)
        # So remaining*c - 1 must divide c
        # Let remaining = p/q in lowest terms
        # p/q * c - 1 divides c
        # c = k * (remaining*c - 1) for some integer k
        # c = k*remaining*c - k
        # c*(1 - k*remaining) = -k
        # c = k / (k*remaining - 1)
        # For c to be positive integer, we need k*remaining - 1 > 0 => k > 1/remaining
        # And c = k / (k*remaining - 1)
        # Let's iterate over possible k values
        # k > 1/remaining => k >= floor(1/remaining) + 1
        # Also c >= min_den and c <= max_den
        # c = k / (k*remaining - 1)
        # For c >= min_den: k / (k*remaining - 1) >= min_den
        # k >= min_den * (k*remaining - 1)
        # k >= min_den*k*remaining - min_den
        # k*(1 - min_den*remaining) >= -min_den
        # If 1 - min_den*remaining > 0: k >= -min_den / (1 - min_den*remaining) = min_den / (min_den*remaining - 1)
        # If 1 - min_den*remaining <= 0: always true (k positive)
        
        # Alternative: iterate c from min_den to max_den
        # d = c / (remaining*c - 1)
        # remaining*c - 1 must divide c and be positive
        # remaining*c > 1 => c > 1/remaining
        # d = c / (remaining*c - 1)
        # d must be integer, distinct from c, and in [min_den, max_den]
        
        p = remaining.numerator
        q = remaining.denominator
        # 1/c + 1/d = p/q
        # q(c+d) = pc*d
        # (pc-q)(pd-q) = q^2
        # This is a standard transformation
        # Let A = pc - q, B = pd - q
        # A*B = q^2
        # A and B are divisors of q^2
        # c = (A + q) / p, d = (B + q) / p
        # c and d must be positive integers
        
        # So we need to find divisors A of q^2 such that:
        # 1. A + q is divisible by p
        # 2. c = (A + q) / p is in [min_den, max_den]
        # 3. B = q^2 / A, d = (B + q) / p is in [min_den, max_den]
        # 4. c != d
        # 5. c, d are distinct from any previously used denominators
        
        # q^2 can be large. Let's bound A.
        # c >= min_den => A + q >= p*min_den => A >= p*min_den - q
        # c <= max_den => A + q <= p*max_den => A <= p*max_den - q
        # Also A > 0 (since c > 0 and p, q > 0, and we need c > 1/remaining = q/p, so pc > q, A > 0)
        
        a_min = max(1, p * min_den - q)
        a_max = p * max_den - q
        if a_max < a_min:
            return None
        
        q2 = q * q
        
        # Find divisors of q^2 in [a_min, a_max]
        # This could be expensive if q is large. Let's try a different approach.
        # Iterate c from max(min_den, floor(q/p) + 1) to max_den
        # Check if (p*c - q) divides c
        
        c_start = max(min_den, q // p + 1)
        for c in range(c_start, max_den + 1):
            denom = p * c - q
            if denom <= 0:
                continue
            if c % denom != 0:
                continue
            d = c // denom
            if d < min_den or d > max_den:
                continue
            if d == c:
                continue
            return [c, d]
        
        return None
    
    def search(num_terms, max_den, min_den=1, used=None):
        nonlocal best_max_den, best_solution
        if used is None:
            used = set()
        
        # Pruning: if we've already exceeded our best, stop
        if max_den >= best_max_den:
            return
        
        # Check time
        if time.time() > deadline:
            return
        
        if num_terms == 0:
            return
        
        if num_terms == 1:
            # remaining must be 1/d for some d in used... wait, no
            # Actually, for num_terms == 1, we need remaining = 1/d where d is the last term
            # But we're building up the solution, so remaining should be 0 if we've used all terms
            # Let me restructure: the search builds a list of denominators
            # When num_terms == 1, we need to find d such that 1/d = remaining
            # d = 1/remaining must be integer
            if remaining > 0:
                d_num = remaining.denominator
                d_den = remaining.numerator
                if d_den == 1 and d_num >= min_den and d_num <= max_den and d_num not in used:
                    sol = sorted(list(used | {d_num}))
                    if max(sol) < best_max_den:
                        best_max_den = max(sol)
                        best_solution = sol
            return
        
        if num_terms == 2:
            result = search_two_terms(remaining, max_den, min_den)
            if result is not None:
                c, d = result
                if c not in used and d not in used:
                    sol = sorted(list(used | {c, d}))
                    if max(sol) < best_max_den:
                        best_max_den = max(sol)
                        best_solution = sol
            return
        
        # Try all possible values for the first denominator
        for d in range(min_den, max_den + 1):
            if d in used:
                continue
            if time.time() > deadline:
                return
            
            remaining = target - Fraction(1, d)
            if remaining <= 0:
                break  # No point trying larger d
            
            # Pruning: max sum of remaining (num_terms-1) terms with denominators >= d+1 and <= max_den
            # Max sum: (num_terms-1) * 1/(d+1)
            # Min sum: (num_terms-1) * 1/max_den
            max_possible = Fraction(num_terms - 1, d + 1)
            min_possible = Fraction(num_terms - 1, max_den)
            
            if remaining > max_possible:
                continue
            if remaining < min_possible:
                # Even the smallest possible sum is too large? No, min_possible is the smallest sum
                # If remaining < min_possible, we can't achieve it
                continue
            
            new_used = used | {d}
            search(num_terms - 1, max_den, d + 1, new_used)
    
    # Search for solutions with smaller max denominator
    # Try 2 terms first
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        search(2, max_d)
        if best_max_den <= max_d:
            break
    
    # Try 3 terms
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        search(3, max_d)
        if best_max_den <= max_d:
            break
    
    # Try 4 terms
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        search(4, max_d)
        if best_max_den <= max_d:
            break
    
    # Try 5 terms
    for max_d in range(best_max_den - 1, 50, -1):
        if time.time() > deadline:
            break
        search(5, max_d)
        if best_max_den <= max_d:
            break
    
    # Final verification
    if not is_valid_solution(best_solution):
        # This should not happen, but just in case
        best_solution = [8, 31, 248]
        best_max_den = 248
    
    return best_solution
# EVOLVE-BLOCK-END
