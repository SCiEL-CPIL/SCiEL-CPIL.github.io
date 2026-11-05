import time
import itertools

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    def count_square_terms(d):
        """Count nonzero terms in the exact integer square of sparse poly d={exp:coeff}."""
        if not d:
            return 0
        items = list(d.items())
        n = len(items)
        sq = {}
        for i in range(n):
            ei, ci = items[i]
            for j in range(i, n):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                sq[key] = sq.get(key, 0) + val
        return sum(1 for v in sq.values() if v != 0)

    def is_valid(d):
        """Check if sparse dict represents a valid polynomial per task constraints."""
        if not d:
            return False
        if len(d) != 12:
            return False
        max_exp = max(d.keys())
        if max_exp > 40:
            return False
        for c in d.values():
            if c < -64 or c > 64:
                return False
        return True

    def to_list(d):
        """Convert sparse dict to dense coefficient list (ascending degree)."""
        if not d:
            return [0]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d:
            lst[e] = c
        return lst

    best_score = float('inf')
    best_poly = None

    # ====================================================================
    # PHASE 1: Exhaustive search over consecutive exponents 0..11
    # Coefficients from {-2,-1,1,2}: 4^12 = 16,777,216 combinations
    # This is the most structured case where cancellation is most likely.
    # ====================================================================
    
    # Pre-compute all 4^12 coefficient tuples
    coeff_options = [-2, -1, 1, 2]
    
    # For speed, use a nested loop structure with early termination
    # 4^12 = 16.7M, each eval takes ~2us with optimized code
    # Total: ~34 seconds. Should fit in budget.
    
    # Use itertools.product for clean iteration
    count = 0
    for coeffs in itertools.product(coeff_options, repeat=12):
        if time.time() > deadline:
            break
        count += 1
        
        # Build sparse dict
        d = {i: coeffs[i] for i in range(12)}
        
        # Fast check: compute square term count
        score = count_square_terms(d)
        
        if score < best_score:
            best_score = score
            best_poly = to_list(d)
        
        # Early exit: if we found 2 terms (theoretical minimum for 12-term poly
        # where P^2 has only P^2's leading and trailing terms surviving), stop.
        # Actually minimum is likely higher. Let's stop if we hit 4 or less.
        if best_score <= 4:
            break

    # ====================================================================
    # PHASE 2: If time remains, search consecutive exponents with wider
    # coefficient range {-3,-2,-1,1,2,3} but only for the most promising
    # exponent patterns (still 0..11).
    # 6^12 = 2.17B - too many. Instead, do a targeted search.
    # ====================================================================
    
    if time.time() < deadline - 5.0:
        # Try coefficients from {-3,-2,-1,1,2,3} but only for specific
        # structured patterns that are likely to give good cancellation.
        
        # Pattern A: Alternating signs with varying magnitudes
        # c_i = (-1)^i * a_i where a_i in {1, 2, 3}
        # 3^12 = 531,441 - very fast
        for mags in itertools.product([1, 2, 3], repeat=12):
            if time.time() > deadline:
                break
            d = {}
            for i in range(12):
                d[i] = mags[i] if i % 2 == 0 else -mags[i]
            score = count_square_terms(d)
            if score < best_score:
                best_score = score
                best_poly = to_list(d)
            if best_score <= 4:
                break
        
        # Pattern B: c_i = a_i for i<6, c_i = -a_i for i>=6
        # 3^12 = 531,441
        for mags in itertools.product([1, 2, 3], repeat=12):
            if time.time() > deadline:
                break
            d = {}
            for i in range(12):
                d[i] = mags[i] if i < 6 else -mags[i]
            score = count_square_terms(d)
            if score < best_score:
                best_score = score
                best_poly = to_list(d)
            if best_score <= 4:
                break

    # ====================================================================
    # PHASE 3: Search over non-consecutive exponent sets with small
    # coefficient values. Focus on exponent sets that create structured
    # overlap in the square.
    # ====================================================================
    
    if time.time() < deadline - 3.0:
        # Try exponents with step 2: 0,2,4,...,22 (12 terms)
        exps_step2 = list(range(0, 24, 2))
        for coeffs in itertools.product([-2, -1, 1, 2], repeat=12):
            if time.time() > deadline:
                break
            d = {exps_step2[i]: coeffs[i] for i in range(12)}
            score = count_square_terms(d)
            if score < best_score:
                best_score = score
                best_poly = to_list(d)
            if best_score <= 4:
                break

        # Try exponents with step 3: 0,3,6,...,33 (12 terms)
        exps_step3 = list(range(0, 36, 3))
        for coeffs in itertools.product([-2, -1, 1, 2], repeat=12):
            if time.time() > deadline:
                break
            d = {exps_step3[i]: coeffs[i] for i in range(12)}
            score = count_square_terms(d)
            if score < best_score:
                best_score = score
                best_poly = to_list(d)
            if best_score <= 4:
                break

    # ====================================================================
    # PHASE 4: Two-cluster exhaustive search
    # P = A(x) + x^M * B(x) where A has 6 terms at 0..5, B has 6 terms at 0..5
    # M chosen to create overlap. Coeffs from {-2,-1,1,2}.
    # 4^6 * 4^6 = 4^12 = 16.7M per M value.
    # ====================================================================
    
    if time.time() < deadline - 3.0:
        a_coeffs = list(itertools.product([-2, -1, 1, 2], repeat=6))
        b_coeffs = list(itertools.product([-2, -1, 1, 2], repeat=6))
        
        for M in range(10, 18):
            if time.time() > deadline:
                break
            # A: exponents 0..5, B: exponents M..M+5
            # Total max exponent: M+5, must be <= 40
            if M + 5 > 40:
                continue
            
            for ac in a_coeffs:
                if time.time() > deadline:
                    break
                for bc in b_coeffs:
                    d = {}
                    for i in range(6):
                        d[i] = ac[i]
                    for j in range(6):
                        e = M + j
                        d[e] = d.get(e, 0) + bc[j]
                        if d[e] == 0:
                            del d[e]
                    
                    if len(d) != 12:
                        continue
                    
                    score = count_square_terms(d)
                    if score < best_score:
                        best_score = score
                        best_poly = to_list(d)
                    if best_score <= 4:
                        break
                if best_score <= 4:
                    break
            if best_score <= 4:
                break

    # ====================================================================
    # FINAL: Return best found, or fallback
    # ====================================================================
    
    if best_poly is None:
        # Fallback: 12 consecutive terms with coefficient 1
        best_poly = [1] * 12
    else:
        # Ensure final coefficient is nonzero
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
        # Ensure length <= 41
        if len(best_poly) > 41:
            best_poly = [1] * 12
        # Ensure exactly 12 nonzero terms
        nz = sum(1 for c in best_poly if c != 0)
        if nz != 12:
            best_poly = [1] * 12

    return best_poly
