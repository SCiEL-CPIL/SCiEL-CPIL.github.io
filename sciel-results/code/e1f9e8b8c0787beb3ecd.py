import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin for 60s limit

    def count_nonzero_square(exps, coeffs):
        """
        Returns the number of nonzero terms in the exact integer square.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients (corresponding to exps)
        """
        if len(exps) != 12 or len(coeffs) != 12:
            return float('inf')
        
        # Check constraints
        if len(set(exps)) != 12:
            return float('inf')
        if max(exps) > 40:
            return float('inf')
        for c in coeffs:
            if c == 0 or not (-64 <= c <= 64):
                return float('inf')
        
        # Calculate square coefficients using a dictionary for sparse convolution
        square_dict = {}
        for i in range(12):
            ei = exps[i]
            ci = coeffs[i]
            for j in range(12):
                ej = exps[j]
                cj = coeffs[j]
                deg = ei + ej
                square_dict[deg] = square_dict.get(deg, 0) + ci * cj
        
        # Count nonzero
        return sum(1 for v in square_dict.values() if v != 0)

    def get_list_representation(exps, coeffs):
        """
        Converts sparse (exps, coeffs) to the required list format.
        """
        max_exp = max(exps)
        if max_exp > 40:
            return None
        res = [0] * (max_exp + 1)
        for e, c in zip(exps, coeffs):
            res[e] = c
        # Ensure length <= 41 and nonzero final coefficient
        if len(res) > 41:
            return None
        if res[-1] == 0:
            return None
        # Ensure exactly 12 nonzero terms
        if sum(1 for x in res if x != 0) != 12:
            return None
        return res

    best_score = float('inf')
    best_candidate = None

    def update_best(exps, coeffs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            best_score = score
            best_candidate = (exps, coeffs)

    # Strategy 1: Deterministic Search over Consecutive Exponents
    # Theoretical analysis suggests that for consecutive exponents 0..11,
    # specific sign patterns (like alternating + - + - ...) minimize the square's support.
    # We exhaustively check all 2^12 = 4096 sign patterns for exponents 0..11.
    # This is extremely fast and guarantees finding the best solution in this family.
    
    base_exps = list(range(12))
    
    # Pre-calculate pair sums and counts for speed? 
    # Actually, 4096 * 144 operations is ~600k ops, which is trivial.
    for mask in range(1 << 12):
        coeffs = []
        for i in range(12):
            if (mask >> i) & 1:
                coeffs.append(1)
            else:
                coeffs.append(-1)
        
        # Optimization: Skip if first or last coeff leads to obvious high support?
        # Not necessary, just compute.
        score = count_nonzero_square(base_exps, coeffs)
        update_best(base_exps, coeffs, score)
        
        if time.time() > deadline:
            break

    # If we found a score of 9, it's likely optimal for this structure.
    # Can we do better? Theoretical lower bound for square of 12-term poly is roughly 12.
    # Wait, (x^0 - x^1 + ...)^2 has cancellations.
    # For n=12, the minimal support of the square of a 12-term polynomial with +/-1 coeffs
    # on consecutive exponents is known to be 9 for some patterns?
    # Let's verify: (1-x)^2 = 1-2x+x^2 (3 terms). 
    # (1-x+x^2)^2 = 1 - 2x + 3x^2 - 2x^3 + x^4 (5 terms).
    # (1-x+x^2-x^3)^2 = 1 - 2x + 3x^2 - 4x^3 + 3x^4 - 2x^5 + x^6 (7 terms).
    # Pattern: 2n-1 terms for n terms in (1-x+...+(-x)^{n-1}).
    # For n=12, 2*12-1 = 23 terms. That's not good.
    
    # What about other patterns?
    # The search above covers all +/-1 patterns.
    
    # Strategy 2: Local Search on Coefficients for the Best Structure Found
    # If the best score is still high (e.g., > 15), we might try varying coefficients slightly.
    # But +/-1 is often optimal for cancellation.
    
    # Strategy 3: Random Search with Structured Initialization
    # If we have time left, do a few random restarts with local search.
    
    if time.time() < deadline - 5:
        # Try a few random sets of exponents to see if non-consecutive helps
        # Usually consecutive is best for maximizing collisions (densest packing).
        # But let's try a few just in case.
        
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            # Random exponents, but keep them somewhat dense
            # Sample 12 from 0..20
            exps = sorted(random.sample(range(0, 21), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            
            score = count_nonzero_square(exps, coeffs)
            update_best(exps, coeffs, score)
            
            # Local search: flip signs
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(12):
                    old_c = coeffs[i]
                    new_c = -old_c
                    coeffs[i] = new_c
                    new_score = count_nonzero_square(exps, coeffs)
                    if new_score < score:
                        score = new_score
                        improved = True
                        update_best(exps, coeffs, score)
                    else:
                        coeffs[i] = old_c
            
            if best_score <= 9:
                break

    # Final validation and formatting
    if best_candidate is not None:
        exps, coeffs = best_candidate
        res_list = get_list_representation(exps, coeffs)
        if res_list is not None:
            return res_list

    # Fallback: Return a valid, though not necessarily optimal, candidate
    # Use the seed-like structure: 12 ones
    return [1] * 12
