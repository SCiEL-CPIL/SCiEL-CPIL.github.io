"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final coefficient,
    entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    """
    
    def square_coeffs(p):
        """Compute P^2 coefficients from P's coefficient list."""
        n = len(p)
        if n == 0:
            return []
        result = [0] * (2*n - 1)
        for i in range(n):
            if p[i] == 0:
                continue
            for j in range(n):
                if p[j] == 0:
                    continue
                result[i+j] += p[i] * p[j]
        return result
    
    def count_nonzero(poly_coeffs):
        """Count nonzero terms in a polynomial given as coefficient list."""
        return sum(1 for c in poly_coeffs if c != 0)
    
    def is_valid(p):
        """Check if p satisfies all constraints."""
        if len(p) > 41:
            return False
        if len(p) == 0 or p[-1] == 0:
            return False
        if any(c < -64 or c > 64 for c in p):
            return False
        if sum(1 for c in p if c != 0) != 12:
            return False
        return True
        
    def trim_poly(p):
        """Remove trailing zeros."""
        while p and p[-1] == 0:
            p.pop()
        return p

    best_nz = float('inf')
    best_p = None
    
    # Strategy: Construct P as A * B where A and B are 6-term polynomials.
    # We search for A and B such that P has exactly 12 non-zero terms.
    # Then we check the sparsity of P^2.
    
    # Define a set of candidate exponent sets for 6-term polynomials.
    # We use small exponents to keep degree low.
    # Base sets: consecutive, gaps, etc.
    base_exponent_sets = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 6],
        [0, 1, 2, 3, 5, 6],
        [0, 1, 2, 4, 5, 6],
        [0, 1, 3, 4, 5, 6],
        [0, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 7],
        [0, 1, 2, 3, 5, 7],
        [0, 1, 2, 4, 6, 7],
        [0, 1, 2, 3, 4, 8],
        [0, 2, 4, 6, 8, 10],
        [0, 1, 3, 5, 7, 9],
        [0, 1, 2, 4, 5, 6],
        [0, 1, 2, 3, 4, 5], # Duplicate for clarity, will be handled by set logic below if needed
    ]
    
    # Unique sets
    unique_exp_sets = list(set(tuple(s) for s in base_exponent_sets))
    
    # Coefficient patterns for 6 terms: +/-1
    coeff_patterns = []
    for mask in range(64):
        coeffs = [1 if (mask >> i) & 1 else -1 for i in range(6)]
        coeff_patterns.append(coeffs)
    
    # Also try some patterns with 0s? No, A and B must have 6 non-zero terms.
    # But we can try coefficients in {-2, -1, 1, 2} later if needed.
    
    # Iterate over pairs of exponent sets for A and B
    # To limit search space, we only consider A and B from the same set of structures
    # or A from one set and B from another.
    
    # Optimization: We can fix A's exponents to [0..5] and vary B's exponents
    # and coefficients, then check P.
    
    fixed_a_exps = [0, 1, 2, 3, 4, 5]
    
    # Vary A's coefficients
    for a_coeffs in coeff_patterns:
        # Vary B's exponents
        for b_exps in unique_exp_sets:
            # Vary B's coefficients
            for b_coeffs in coeff_patterns:
                # Compute P = A * B
                # A has max exp 5, B has max exp max(b_exps)
                max_exp = max(fixed_a_exps) + max(b_exps)
                if max_exp >= 41:
                    continue
                
                p = [0] * (max_exp + 1)
                for i in range(6):
                    if a_coeffs[i] == 0: continue
                    for j in range(6):
                        if b_coeffs[j] == 0: continue
                        p[fixed_a_exps[i] + b_exps[j]] += a_coeffs[i] * b_coeffs[j]
                
                # Trim and check validity
                p = trim_poly(p)
                if is_valid(p):
                    sq = square_coeffs(p)
                    nz = count_nonzero(sq)
                    if nz < best_nz:
                        best_nz = nz
                        best_p = p[:]
    
    # Also try symmetric products where A and B are swapped or different base sets
    # To save time, we rely on the above loop covering many cases.
    # If best_nz is still high, we might need more complex searches.
    
    # Fallback: Direct search on P with 12 terms
    if best_nz > 20: # Threshold for "good"
        # Try consecutive 12 terms with optimized signs
        for mask in range(4096):
            coeffs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
            p = coeffs[:]
            if is_valid(p):
                sq = square_coeffs(p)
                nz = count_nonzero(sq)
                if nz < best_nz:
                    best_nz = nz
                    best_p = p[:]
    
    if best_p is None:
        return [1]*12
    
    return best_p
# EVOLVE-BLOCK-END
