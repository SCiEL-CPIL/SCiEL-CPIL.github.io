import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def solve_k2(num, den, min_d):
        """
        Solve 1/d1 + 1/d2 = num/den with d1, d2 >= min_d, d1 < d2.
        Returns (d1, d2) or None.
        """
        # d1 must be > den/num
        d1_min = max(min_d, den // num + 1)
        # d1 < 2 * den / num because 1/d1 < num/den - 1/d1 => 2/d1 < num/den => d1 > 2*den/num is false?
        # 1/d1 + 1/d2 = num/den. Since d2 > d1, 1/d2 < 1/d1.
        # num/den < 2/d1 => d1 < 2*den/num.
        d1_max = (2 * den) // num
        
        for d1 in range(d1_min, d1_max + 1):
            # 1/d2 = num/den - 1/d1 = (num*d1 - den) / (den*d1)
            # d2 = den*d1 / (num*d1 - den)
            rem_num = num * d1 - den
            if rem_num <= 0:
                continue
            if (den * d1) % rem_num == 0:
                d2 = (den * d1) // rem_num
                if d2 > d1:
                    return (d1, d2)
        return None

    def solve_k3(num, den, min_d):
        """
        Solve 1/d1 + 1/d2 + 1/d3 = num/den with d1, d2, d3 >= min_d, d1 < d2 < d3.
        """
        # d1 must be such that 1/d1 < num/den
        d1_min = max(min_d, den // num + 1)
        # Also 1/d1 + 1/d2 + 1/d3 < 3/d1 => num/den < 3/d1 => d1 < 3*den/num
        d1_max = (3 * den) // num
        
        for d1 in range(d1_min, d1_max + 1):
            # Remainder R' = num/den - 1/d1
            # R' = (num*d1 - den) / (den*d1)
            rem_num = num * d1 - den
            if rem_num <= 0:
                continue
            rem_den = den * d1
            # Reduce fraction? Not strictly necessary for solve_k2 logic if we handle gcd, 
            # but solve_k2 uses integer division checks. 
            # Let's reduce to keep numbers small.
            g = rem_num.gcd(rem_den)
            rn = rem_num // g
            rd = rem_den // g
            
            res = solve_k2(rn, rd, d1 + 1)
            if res:
                return (d1, res[0], res[1])
        return None

    # Search for 8 terms
    # We expect denominators to be small, roughly 5 to 20.
    # We fix the first 5 and solve for the last 3.
    
    # Ranges for the first 5 terms
    # d1 is small
    for d1 in range(5, 11):
        if time.time() > deadline: return None
        # d2 > d1
        for d2 in range(d1 + 1, 16):
            if time.time() > deadline: return None
            # d3 > d2
            for d3 in range(d2 + 1, 21):
                if time.time() > deadline: return None
                # d4 > d3
                for d4 in range(d3 + 1, 26):
                    if time.time() > deadline: return None
                    # d5 > d4
                    for d5 in range(d4 + 1, 31):
                        if time.time() > deadline: return None
                        
                        # Sum of first 5
                        # S5 = 1/d1 + ... + 1/d5
                        # R5 = 1 - S5
                        # We need 1/d6 + 1/d7 + 1/d8 = R5
                        
                        # Calculate R5 as Fraction
                        # To avoid huge numbers, we can sum step by step or use Fraction
                        s5 = Fraction(1, d1) + Fraction(1, d2) + Fraction(1, d3) + Fraction(1, d4) + Fraction(1, d5)
                        r5 = 1 - s5
                        
                        if r5 <= 0:
                            continue
                        
                        # Solve for last 3
                        res = solve_k3(r5.numerator, r5.denominator, d5 + 1)
                        if res:
                            solution = [d1, d2, d3, d4, d5] + list(res)
                            # Verify
                            if len(set(solution)) == 8:
                                if all(5 <= d <= 10000000 for d in solution):
                                    s = sum(Fraction(1, d) for d in solution)
                                    if s == 1:
                                        return solution
    return None

# Fallback if the tight search fails (should not happen for k=8 min)
def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Try the optimized search first
    # Re-define inner functions or call them? 
    # It's cleaner to put the logic in one function.
    
    def solve_k2(num, den, min_d):
        d1_min = max(min_d, den // num + 1)
        d1_max = (2 * den) // num
        for d1 in range(d1_min, d1_max + 1):
            rem_num = num * d1 - den
            if rem_num <= 0:
                continue
            if (den * d1) % rem_num == 0:
                d2 = (den * d1) // rem_num
                if d2 > d1:
                    return (d1, d2)
        return None

    def solve_k3(num, den, min_d):
        d1_min = max(min_d, den // num + 1)
        d1_max = (3 * den) // num
        for d1 in range(d1_min, d1_max + 1):
            rem_num = num * d1 - den
            if rem_num <= 0:
                continue
            rem_den = den * d1
            g = rem_num.gcd(rem_den)
            rn = rem_num // g
            rd = rem_den // g
            res = solve_k2(rn, rd, d1 + 1)
            if res:
                return (d1, res[0], res[1])
        return None

    # Search
    for d1 in range(5, 12):
        if time.time() > deadline:
            break
        for d2 in range(d1 + 1, 17):
            if time.time() > deadline:
                break
            for d3 in range(d2 + 1, 22):
                if time.time() > deadline:
                    break
                for d4 in range(d3 + 1, 27):
                    if time.time() > deadline:
                        break
                    for d5 in range(d4 + 1, 32):
                        if time.time() > deadline:
                            break
                        s5 = Fraction(1, d1) + Fraction(1, d2) + Fraction(1, d3) + Fraction(1, d4) + Fraction(1, d5)
                        r5 = 1 - s5
                        if r5 <= 0:
                            continue
                        res = solve_k3(r5.numerator, r5.denominator, d5 + 1)
                        if res:
                            solution = [d1, d2, d3, d4, d5] + list(res)
                            if len(set(solution)) == 8 and all(5 <= d <= 10000000 for d in solution):
                                if sum(Fraction(1, d) for d in solution) == 1:
                                    return solution
    return None
