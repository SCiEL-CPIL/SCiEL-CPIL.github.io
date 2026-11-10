def construct():
    """
    Find a 12-term integer polynomial (coeffs in [-64,64], degree <= 40)
    whose exact square has the minimum number of nonzero terms.
    
    Uses a constraint satisfaction approach: systematically search over
    well-structured 12-term polynomials, computing the exact square and
    counting nonzero terms.
    """
    
    best_score = 21
    best_poly = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    
    def compute_square(coeffs):
        """Compute the exact square of a polynomial given as coefficient list."""
        n = len(coeffs)
        square = [0] * (2 * n - 1)
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(i, n):
                if coeffs[j] == 0:
                    continue
                val = coeffs[i] * coeffs[j]
                if i == j:
                    square[i + j] += val
                else:
                    square[i + j] += 2 * val
        return square
    
    def count_nonzeros(poly):
        return sum(1 for c in poly if c != 0)
    
    def check_valid(poly):
        if len(poly) > 41:
            return False
        if poly[-1] == 0:
            return False
        if sum(1 for c in poly if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in poly):
            return False
        return True
    
    # Search strategy: try various structured families of 12-term polynomials
    
    # Family 1: Antisymmetric P(x) = A(x) - x^k A(x) where A has 6 terms
    # This naturally creates cancellations
    def search_antisymmetric():
        best = (best_score, best_poly[:])
        
        for half_len in range(4, 10):
            for shift in range(half_len, min(30, 42 - half_len)):
                # A(x) has exactly 6 nonzero terms in positions 0..half_len-1
                # P(x) = A(x) - x^shift * A(x)
                # Total nonzero terms = 12 (assuming no overlap)
                
                # Try small coefficient values for A
                coeff_vals = [-2, -1, 1, 2]
                
                # Choose 6 positions for A's terms from 0..half_len-1
                import itertools
                positions = list(range(half_len))
                if len(positions) < 6:
                    continue
                
                for pos_combo in itertools.combinations(positions, 6):
                    for coeffs in itertools.product(coeff_vals, repeat=6):
                        # Build A(x)
                        a_coeffs = [0] * half_len
                        for p, c in zip(pos_combo, coeffs):
                            a_coeffs[p] = c
                        
                        # Build P(x) = A(x) - x^shift * A(x)
                        total_len = shift + half_len
                        p_coeffs = [0] * total_len
                        for i, c in enumerate(a_coeffs):
                            p_coeffs[i] += c
                            p_coeffs[shift + i] += -c
                        
                        if not check_valid(p_coeffs):
                            continue
                        
                        sq = compute_square(p_coeffs)
                        score = count_nonzeros(sq)
                        
                        if score < best[0]:
                            best = (score, p_coeffs[:])
                            if score <= 16:
                                return best
        
        return best
    
    # Family 2: Polynomials with specific exponent patterns
    # Try exponents that create arithmetic progressions or other structured sets
    def search_structured_exponents():
        best = (best_score, best_poly[:])
        
        import itertools
        
        # Try various exponent sets
        exponent_patterns = []
        
        # Consecutive blocks
        for start in range(0, 20):
            for length in range(12, 20):
                if start + length <= 41:
                    exponent_patterns.append(list(range(start, start + length)))
        
        # Arithmetic progressions
        for start in range(0, 10):
            for d in range(1, 5):
                exponents = [start + i * d for i in range(12)]
                if max(exponents) < 41:
                    exponent_patterns.append(exponents)
        
        # Symmetric around center
        for center in range(6, 20):
            for spread in range(1, 10):
                exponents = [center - spread + i for i in range(12)]
                if min(exponents) >= 0 and max(exponents) < 41:
                    exponent_patterns.append(sorted(set(exponents)))
        
        # Remove duplicates
        seen = set()
        unique_patterns = []
        for pat in exponent_patterns:
            key = tuple(pat)
            if key not in seen:
                seen.add(key)
                unique_patterns.append(pat)
        
        coeff_vals = [-2, -1, 1, 2]
        
        for exponents in unique_patterns[:200]:  # Limit to avoid timeout
            if len(exponents) != 12:
                continue
            if min(exponents) < 0 or max(exponents) >= 41:
                continue
            
            # Try coefficient assignments with symmetry breaking
            for coeffs in itertools.product(coeff_vals, repeat=12):
                # Symmetry: fix first nonzero coefficient to be positive
                if coeffs[0] < 0:
                    continue
                
                poly = [0] * (max(exponents) + 1)
                for e, c in zip(exponents, coeffs):
                    poly[e] = c
                
                if not check_valid(poly):
                    continue
                
                sq = compute_square(poly)
                score = count_nonzeros(sq)
                
                if score < best[0]:
                    best = (score, poly[:])
                    if score <= 16:
                        return best
        
        return best
    
    # Family 3: Randomized search with local improvement
    def search_randomized():
        best = (best_score, best_poly[:])
        import random
        random.seed(42)
        
        for _ in range(5000):
            # Generate random 12-term polynomial
            positions = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            
            # Ensure last position has nonzero coefficient
            if positions[-1] == 40:
                coeffs[-1] = random.choice([1, 2])
            
            poly = [0] * 41
            for p, c in zip(positions, coeffs):
                poly[p] = c
            
            if not check_valid(poly):
                continue
            
            sq = compute_square(poly)
            score = count_nonzeros(sq)
            
            if score < best[0]:
                best = (score, poly[:])
                if score <= 16:
                    return best
        
        return best
    
    # Run searches
    s1, p1 = search_antisymmetric()
    if s1 < best_score:
        best_score = s1
        best_poly = p1
    
    if best_score > 18:
        s2, p2 = search_structured_exponents()
        if s2 < best_score:
            best_score = s2
            best_poly = p2
    
    if best_score > 17:
        s3, p3 = search_randomized()
        if s3 < best_score:
            best_score = s3
            best_poly = p3
    
    # Final verification
    assert sum(1 for c in best_poly if c != 0) == 12
    assert all(-64 <= c <= 64 for c in best_poly)
    assert len(best_poly) <= 41
    assert best_poly[-1] != 0
    
    return best_poly
