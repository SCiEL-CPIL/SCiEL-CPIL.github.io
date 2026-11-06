"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def count_nonzero(coeffs):
        """Count nonzero terms in coefficient list."""
        # Trim trailing zeros
        while coeffs and coeffs[-1] == 0:
            coeffs = coeffs[:-1]
        return len(coeffs)
    
    def poly_square_nonzero(exponents, coefficients):
        """Compute number of nonzero terms in P^2 where P = sum(coeff[i] * x^exp[i])."""
        n = len(exponents)
        # Map exponent sum -> coefficient
        sq = defaultdict(int)
        for i in range(n):
            for j in range(n):
                e = exponents[i] + exponents[j]
                sq[e] += coefficients[i] * coefficients[j]
        # Count nonzero
        return sum(1 for v in sq.values() if v != 0)
    
    def make_poly(exponents, coefficients):
        """Build coefficient list in ascending degree."""
        max_deg = max(exponents)
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] += c
        # Trim trailing zeros
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    def is_valid(coeffs):
        """Check all constraints."""
        if len(coeffs) > 41:
            return False
        if not coeffs or coeffs[-1] == 0:
            return False
        nonzero = sum(1 for c in coeffs if c != 0)
        if nonzero != 12:
            return False
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        return True
    
    best_score = 23  # seed: [1]*12 has 23 nonzero in square
    best_poly = [1] * 12
    
    # Try various structured approaches
    # Approach 1: Consecutive exponents with sign patterns
    # P = sum_{i=0}^{11} s_i * x^i where s_i in {-1, 1}
    # Square has terms at exponents 0..22
    # Coefficient at exponent k = sum_{i+j=k} s_i * s_j
    
    # For consecutive exponents 0..11, the square coefficients are:
    # k=0: s0^2 = 1
    # k=1: 2*s0*s1
    # k=2: s1^2 + 2*s0*s2 = 1 + 2*s0*s2
    # etc.
    
    # Try all sign patterns (2^12 = 4096)
    for mask in range(4096):
        if time.time() > deadline:
            break
        signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
        exponents = list(range(12))
        nz = poly_square_nonzero(exponents, signs)
        if nz < best_score:
            best_score = nz
            best_poly = make_poly(exponents, signs)
    
    # Approach 2: Try exponents with gaps to create more collisions
    # Use exponents like {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} but permute signs
    # Already covered above
    
    # Approach 3: Try non-consecutive exponents
    # For example, exponents that create arithmetic progressions in sums
    # Try: 0,1,2,3,4,5,6,7,8,9,10,20 (gap at end)
    # Or: 0,1,2,3,4,5,6,7,8,9,18,19
    
    # Let's try small systematic search over exponent sets
    # Keep it bounded: max degree <= 40, 12 exponents
    
    # Try structured exponent sets
    exponent_sets = []
    
    # Consecutive
    for start in range(0, 10):
        exponent_sets.append(list(range(start, start + 12)))
    
    # With a gap
    for start in range(0, 10):
        for gap_pos in range(2, 12):
            exps = list(range(start, start + gap_pos)) + list(range(start + gap_pos + 1, start + gap_pos + 13))
            if len(exps) == 12:
                exponent_sets.append(exps)
    
    # Arithmetic progression with different step
    for step in [2, 3]:
        for start in range(0, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                exponent_sets.append(exps)
    
    # Mixed: some consecutive, some spaced
    # 0,1,2,3,4,5,6,7,10,11,12,13
    exponent_sets.append([0,1,2,3,4,5,6,7,10,11,12,13])
    exponent_sets.append([0,1,2,3,4,5,6,7,9,10,11,12])
    exponent_sets.append([0,1,2,3,4,5,6,8,9,10,11,12])
    
    for exps in exponent_sets:
        if time.time() > deadline:
            break
        if len(exps) != 12 or max(exps) > 40:
            continue
        if len(set(exps)) != 12:
            continue
        # Try sign patterns - but 2^12 is too many for many sets
        # Try a few key patterns
        for mask in range(0, 4096, 1):  # all patterns, but break early if too slow
            if time.time() > deadline:
                break
            signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
            nz = poly_square_nonzero(exps, signs)
            if nz < best_score:
                best_score = nz
                best_poly = make_poly(exps, signs)
    
    # Approach 4: Use coefficients other than ±1 to create more cancellations
    # P = a + b*x + c*x^2 + ... with small integer coefficients
    # This is a much larger search space. Let's try a few structured ones.
    
    # Try: coefficients in {-2, -1, 0, 1, 2} but exactly 12 nonzero
    # For consecutive exponents 0..11, try small coefficient patterns
    
    # Key insight: if P = Q^2 for some 6-term Q, then P^2 = Q^4
    # But that's probably not helpful for sparsity.
    
    # Try: P with coefficients that make P^2 have many zeros
    # For example, if P = (1 - x)(1 + x)(1 + x^2)(1 + x^3)(1 + x^4)(1 + x^5)
    # This has 2^6 = 64 terms... too many.
    
    # Try P = product of (1 ± x^k) for carefully chosen k
    # P = (1+x)(1+x^2)(1+x^3)(1+x^4)(1+x^5)(1+x^6) has 64 terms
    
    # What about P = (1-x)(1+x^2)(1+x^3)(1+x^4)(1+x^5)(1+x^6)?
    # Still 64 terms.
    
    # Let me try a different approach: use the fact that if we can make
    # many coefficients of P^2 zero, we win.
    
    # For P with 12 terms, P^2 has at most 144 "raw" terms (with multiplicity).
    # We want as many as possible to cancel or coincide.
    
    # Try: P = sum_{i=0}^{5} x^i - sum_{i=6}^{11} x^i
    # = (1-x^6)/(1-x) - x^6*(1-x^6)/(1-x) = (1-x^6)(1-x^6)/(1-x) ... no
    # = (1 - x^6) - x^6(1 - x^6) / (1-x) ... let me just compute
    
    # P = 1 + x + x^2 + x^3 + x^4 + x^5 - x^6 - x^7 - x^8 - x^9 - x^10 - x^11
    # P = (1-x^6)(1+x+...+x^5) - x^6(1+x+...+x^5) = (1-x^6)(1-x^6)/(1-x) = (1-x^6)^2/(1-x)
    # P = (1 - 2x^6 + x^12)/(1-x)
    # P^2 = (1 - 2x^6 + x^12)^2 / (1-x)^2
    # = (1 - 4x^6 + 6x^12 - 4x^18 + x^24) / (1-x)^2
    # (1-x)^{-2} = sum_{k>=0} (k+1)x^k
    # So P^2 = (1 - 4x^6 + 6x^12 - 4x^18 + x^24) * sum_{k>=0} (k+1)x^k
    # This is a finite polynomial since P is a polynomial.
    # Let me compute it directly.
    
    exps = list(range(12))
    coeffs = [1]*6 + [-1]*6
    nz = poly_square_nonzero(exps, coeffs)
    if nz < best_score:
        best_score = nz
        best_poly = make_poly(exps, coeffs)
    
    # Try: 1,1,1,1,1,1,-1,-1,-1,-1,-1,-1 (already done)
    # Try: 1,-1,1,-1,1,-1,1,-1,1,-1,1,-1
    exps = list(range(12))
    coeffs = [1 if i%2==0 else -1 for i in range(12)]
    nz = poly_square_nonzero(exps, coeffs)
    if nz < best_score:
        best_score = nz
        best_poly = make_poly(exps, coeffs)
    
    # Try: 1,0,1,0,1,0,1,0,1,0,1,0 - but that's only 6 nonzero, invalid
    # Need exactly 12 nonzero.
    
    # Try coefficients: [1,1,-1,1,-1,1,1,-1,1,-1,1,1]
    for test_coeffs in [
        [1,1,-1,1,-1,1,1,-1,1,-1,1,1],
        [1,-1,-1,1,1,-1,-1,1,1,-1,-1,1],
        [2,1,-1,0,0,0,0,0,0,0,0,0],  # only 3 nonzero, invalid
    ]:
        if len([c for c in test_coeffs if c != 0]) != 12:
            continue
        exps = list(range(12))
        nz = poly_square_nonzero(exps, test_coeffs)
        if nz < best_score:
            best_score = nz
            best_poly = make_poly(exps, test_coeffs)
    
    # Verify and return
    if is_valid(best_poly):
        return best_poly
    
    # Fallback: return seed
    return [1] * 12
# EVOLVE-BLOCK-END
