"""Structural approach: model P as product of three 4-term factors with disjoint
exponent supports. P^2 = (A*B*C)^2 = A^2 * B^2 * C^2. Each factor has 4 terms,
so each square has at most 10 nonzero terms. The product structure allows
systematic search over factor structures."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 48  # safety margin before 90s kill

    N = 41  # max length
    NZ_TARGET = 12

    def square_nonzero_count(poly):
        """Compute the number of nonzero coefficients in poly^2 (exact integer)."""
        nz = [(i, poly[i]) for i in range(len(poly)) if poly[i] != 0]
        n = len(nz)
        if n == 0:
            return 0
        sq = [0] * (2 * len(poly) - 1)
        for ii in range(n):
            i, ai = nz[ii]
            for jj in range(ii, n):
                j, aj = nz[jj]
                prod = ai * aj
                if ii == jj:
                    sq[i + j] += prod
                else:
                    sq[i + j] += 2 * prod
        return sum(1 for c in sq if c != 0)

    def poly_mul(p1, p2):
        """Multiply two polynomials (coefficient lists, ascending degree)."""
        r1 = [i for i in range(len(p1)) if p1[i] != 0]
        r2 = [i for i in range(len(p2)) if p2[i] != 0]
        if not r1 or not r2:
            return [0]
        result_len = len(p1) + len(p2) - 1
        result = [0] * result_len
        for i in r1:
            for j in r2:
                result[i + j] += p1[i] * p2[j]
        # Trim trailing zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    def factor_square_nonzero(factor):
        """Number of nonzero terms in factor^2."""
        return square_nonzero_count(factor)

    def product_square_nonzero(factors):
        """
        Given list of factor polynomials, compute P = prod(factors),
        then return nonzero count of P^2.
        """
        P = [1]
        for f in factors:
            P = poly_mul(P, f)
        return square_nonzero_count(P)

    def build_poly_from_factors(factors):
        """Build the full polynomial from factors, return (poly, nz_count)."""
        P = [1]
        for f in factors:
            P = poly_mul(P, f)
        nz = sum(1 for c in P if c != 0)
        return P, nz

    # Strategy: search over structures of three 4-term factors
    # Each factor has 4 nonzero terms at specific positions
    # We want the product to have exactly 12 nonzero terms (best case: no internal cancellation)
    
    best_poly = None
    best_score = float('inf')

    def try_candidate(poly):
        nonlocal best_poly, best_score
        # Trim trailing zeros
        last = -1
        for i in range(len(poly) - 1, -1, -1):
            if poly[i] != 0:
                last = i
                break
        if last == -1:
            return
        trimmed = poly[:last + 1]
        nz_count = sum(1 for c in trimmed if c != 0)
        if nz_count != NZ_TARGET:
            return
        if len(trimmed) > N:
            return
        if any(abs(c) > 64 for c in trimmed):
            return
        score = square_nonzero_count(trimmed)
        if score < best_score:
            best_score = score
            best_poly = trimmed[:]

    # Phase 1: Systematic search over factor structures
    # Factor positions: each factor has 4 terms
    # Try geometric progressions for each factor
    
    # Common base offsets to separate factor supports
    base_offsets = [0, 5, 10, 15, 20, 25, 30, 35]
    
    # Ratio patterns for 4-term geometric sequences
    ratios = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # Coefficient patterns for 4 terms (normalized to small integers)
    coeff_patterns = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
        [1, 2, 3, 4],
        [1, -2, 3, -4],
        [1, 1, 2, 2],
        [1, -1, 2, -2],
        [1, 2, 1, 2],
        [1, -2, 1, -2],
        [1, 1, 1, -1],
        [1, -1, 1, 1],
        [2, 1, 1, 2],
        [1, 2, 2, 1],
        [1, 1, 2, 1],
        [1, 2, 1, 1],
    ]
    
    # Sign patterns for overall factor scaling
    sign_patterns = [1, -1]
    
    # Try combinations of factor structures
    attempts = 0
    max_attempts = 50000
    
    while time.time() < deadline - 8 and attempts < max_attempts:
        attempts += 1
        
        # Choose base offset for each factor
        if random.random() < 0.3:
            # Structured: evenly spaced
            base0 = random.randint(0, 5)
            gap = random.randint(5, 12)
            bases = [base0, base0 + gap, base0 + 2 * gap]
        else:
            # Random bases
            bases = sorted(random.sample(range(0, 30), 3))
        
        # Choose ratio for each factor
        ratios_choice = [random.choice(ratios) for _ in range(3)]
        
        # Choose coefficient pattern for each factor
        coeffs_choice = [random.choice(coeff_patterns) for _ in range(3)]
        
        # Build factors
        factors = []
        valid = True
        for b, r, cp in zip(bases, ratios_choice, coeffs_choice):
            factor = [0] * N
            for k in range(4):
                pos = b + k * r
                if pos >= N:
                    valid = False
                    break
                coeff = cp[k]
                if random.random() < 0.3:
                    coeff = -coeff  # random sign flip
                factor[pos] = coeff
            if not valid:
                break
            factors.append(factor)
        
        if not valid or len(factors) != 3:
            continue
        
        # Compute product
        P = [1]
        for f in factors:
            P = poly_mul(P, f)
        
        # Check constraints
        if len(P) > N:
            continue
        if any(abs(c) > 64 for c in P):
            continue
        nz = sum(1 for c in P if c != 0)
        if nz != NZ_TARGET:
            continue
        
        try_candidate(P)
        
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Local search refinement on best found
    if best_poly is not None and time.time() < deadline - 3:
        current = best_poly[:]
        current_score = best_score
        best_local = best_poly[:]
        best_local_score = best_score
        
        max_iter = 50000
        for it in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            nz_positions = [i for i in range(len(current)) if current[i] != 0]
            if len(nz_positions) != NZ_TARGET:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Perturb coefficient
                idx = random.choice(nz_positions)
                old_val = current[idx]
                r = random.random()
                if r < 0.4:
                    delta = random.randint(-3, 3)
                    new_val = old_val + delta
                elif r < 0.7:
                    new_val = random.randint(1, 64)
                else:
                    new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                if new_val == 0:
                    new_val = 1
                if abs(new_val) > 64:
                    new_val = 64 if new_val > 0 else -64
                if new_val == old_val:
                    continue
                current[idx] = new_val
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = current[:]
                else:
                    current[idx] = old_val
                    
            elif move_type < 0.75:
                # Move term to new position
                idx = random.choice(nz_positions)
                coeff = current[idx]
                empty = [i for i in range(len(current)) if current[i] == 0]
                if not empty:
                    continue
                new_pos = random.choice(empty)
                current[idx] = 0
                current[new_pos] = coeff
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = current[:]
                else:
                    current[new_pos] = 0
                    current[idx] = coeff
                    
            else:
                # Swap coefficients
                i1, i2 = random.sample(nz_positions, 2)
                v1, v2 = current[i1], current[i2]
                current[i1], current[i2] = v2, v1
                new_score = square_nonzero_count(current)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local = current[:]
                else:
                    current[i1], current[i2] = v1, v2
        
        if best_local_score < best_score:
            best_poly = best_local
            best_score = best_local_score
    
    # Phase 3: Try some hand-crafted known-good structures
    # x^0 + x^5 + x^10 + x^15 + ... pattern with careful signs
    # Try: P = (1+x^a+x^{2a}+x^{3a})(1+x^b+x^{2b}+x^{3b})(1+x^c+x^{2c}+x^{3c})
    # where a,b,c are chosen to minimize collisions
    
    if time.time() < deadline - 5:
        for a in range(3, 15):
            for b in range(a + 1, a + 15):
                for c in range(b + 1, b + 15):
                    if time.time() > deadline - 5:
                        break
                    max_pos = 3 * (a + b + c)
                    if max_pos >= N:
                        continue
                    
                    # Try different sign patterns
                    for sa in [1, -1]:
                        for sb in [1, -1]:
                            for sc in [1, -1]:
                                f1 = [0] * N
                                f2 = [0] * N
                                f3 = [0] * N
                                for k in range(4):
                                    f1[k * a] = sa
                                    f2[k * b] = sb
                                    f3[k * c] = sc
                                
                                P = poly_mul(poly_mul(f1, f2), f3)
                                if len(P) > N:
                                    continue
                                nz = sum(1 for c in P if c != 0)
                                if nz == NZ_TARGET and all(abs(c) <= 64 for c in P):
                                    try_candidate(P)
    
    # Fallback
    if best_poly is None:
        # Use a simple valid polynomial
        best_poly = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        # Need exactly 12 nonzero terms
        best_poly = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        # Count: positions 0,5,10,15,20,25,30 = 7 terms, need 5 more
        best_poly = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        # That's 9 terms. Let me just make a simple one.
        best_poly = [1] * 12 + [0] * 29
    
    # Final validation
    last = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last = i
            break
    if last == -1:
        return [1] + [0] * 10 + [1]

    result = best_poly[:last + 1]
    nz_count = sum(1 for c in result if c != 0)
    if nz_count != NZ_TARGET:
        # Force fix
        nz_positions = [i for i in range(len(result)) if result[i] != 0]
        if len(nz_positions) >= NZ_TARGET:
            fixed = [0] * len(result)
            for i in nz_positions[:NZ_TARGET]:
                fixed[i] = result[i]
            result = fixed
        else:
            result = [1] * NZ_TARGET

    return result

# EVOLVE-BLOCK-END
