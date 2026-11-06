"""Sparse squares of twelve-term integer polynomials - search for sparse square."""
import time
import random

_DEADLINE = time.time() + 80.0  # safety margin before 90s kill

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _square_nonzeros(poly):
    """Compute square of poly and count nonzero terms."""
    n = len(poly)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        pi = poly[i]
        if pi == 0:
            continue
        for j in range(n):
            pj = poly[j]
            if pj != 0:
                sq[i + j] += pi * pj
    return _count_nonzero(sq)

def _valid(poly):
    """Check all constraints."""
    # length <= 41
    if len(poly) > 41:
        return False
    # nonzero final coefficient
    if poly[-1] == 0:
        return False
    # entries in [-64, 64]
    if any(c < -64 or c > 64 for c in poly):
        return False
    # exactly 12 nonzero terms
    if _count_nonzero(poly) != 12:
        return False
    return True

def _try_local_opt(poly, budget):
    """Local search: flip signs, swap positions, change values to reduce square nonzeros."""
    n = len(poly)
    best = list(poly)
    best_score = _square_nonzeros(best)
    
    while time.time() < budget:
        improved = False
        # Try flipping signs
        for i in range(n):
            if best[i] != 0:
                trial = best[:]
                trial[i] = -trial[i]
                s = _square_nonzeros(trial)
                if s < best_score:
                    best = trial
                    best_score = s
                    improved = True
        # Try moving a term to another position (swap with zero)
        for i in range(n):
            if best[i] != 0:
                for j in range(n):
                    if best[j] == 0:
                        trial = best[:]
                        trial[j] = trial[i]
                        trial[i] = 0
                        # Ensure last coefficient is nonzero
                        if trial[-1] == 0:
                            continue
                        s = _square_nonzeros(trial)
                        if s < best_score:
                            best = trial
                            best_score = s
                            improved = True
        # Try changing a coefficient value
        for i in range(n):
            if best[i] != 0:
                for val in [-1, 1, -2, 2, -3, 3, -6, 6, -12, 12, -24, 24]:
                    if val == best[i]:
                        continue
                    trial = best[:]
                    trial[i] = val
                    s = _square_nonzeros(trial)
                    if s < best_score:
                        best = trial
                        best_score = s
                        improved = True
        if not improved:
            break
    return best, best_score

def _random_poly(rng):
    """Generate a random valid 12-term polynomial."""
    # Choose 12 distinct positions from 0..40
    positions = rng.sample(range(41), 12)
    positions.sort()
    # Ensure last position has nonzero (it will since it's in positions)
    # But we need the polynomial length to have nonzero final coeff
    # Let's make sure the max position is used
    max_pos = max(positions)
    poly = [0] * (max_pos + 1)
    for p in positions:
        # Random value in [-64, 64], nonzero
        val = rng.choice(range(1, 65))
        if rng.random() < 0.5:
            val = -val
        poly[p] = val
    return poly

def _random_sign_poly(rng, positions):
    """Given fixed positions, random signs and magnitudes."""
    n = max(positions) + 1
    poly = [0] * n
    for p in positions:
        val = rng.choice([1, -1, 2, -2, 3, -3, 6, -6])
        poly[p] = val
    return poly

def construct():
    rng = random.Random(42)
    
    # Baseline: [1]*12 has 23 nonzero terms in square
    best_poly = [1] * 12
    best_score = _square_nonzeros(best_poly)  # 23
    
    budget = _DEADLINE - 5.0  # leave 5s margin
    
    # Phase 1: Random search with local optimization
    attempts = 0
    while time.time() < budget:
        poly = _random_poly(rng)
        s = _square_nonzeros(poly)
        if s < best_score:
            best_poly = poly
            best_score = s
            # Local optimize
            best_poly, best_score = _try_local_opt(best_poly, min(budget, time.time() + 2.0))
        attempts += 1
        if attempts % 1000 == 0 and time.time() > budget:
            break
    
    # Phase 2: Systematic search over structured patterns
    # Try polynomials with symmetric patterns
    while time.time() < budget:
        # Pattern: terms at positions forming arithmetic progression or clustered
        # Try: a0 + a1*x^d + a2*x^(2d) + ... 
        d = rng.randint(1, 5)
        positions = list(range(0, 12 * d, d))[:12]
        if len(positions) < 12:
            positions = rng.sample(range(41), 12)
            positions.sort()
        poly = _random_sign_poly(rng, positions)
        s = _square_nonzeros(poly)
        if s < best_score:
            best_poly, best_score = _try_local_opt(poly, min(budget, time.time() + 1.0))
    
    # Phase 3: Try specific known sparse-square constructions
    # Sum of squares of binomials: (a+b)^2 = a^2+2ab+b^2 (3 terms)
    # Product of sparse polynomials can give sparse squares
    # Try: p = a0 + a1*x^k (2 terms) -> p^2 = a0^2 + 2*a0*a1*x^k + a1^2*x^(2k) (3 terms)
    # But we need exactly 12 terms in p.
    # Try: p = sum of 12 terms arranged so cross terms cancel
    # E.g., p = x^0 + x^a + x^(2a) + ... + x^(11a) -> square has many terms
    # Try: p = (x^0+x^a)(x^0+x^b)(x^0+x^c)... arranged to be sparse
    
    # Try: p with terms that make p^2 a sum of few squares
    # p = a*x^i + b*x^j where a^2+b^2 and 2ab terms...
    # For 12 terms: try p = sum_{k=0}^{11} c_k * x^{m_k}
    # where the squares c_k^2 land on few distinct degrees
    # and cross terms 2*c_i*c_j*x^(m_i+m_j) cancel or land on same degrees
    
    # Strategy: make all c_k^2 land on same set of degrees
    # e.g., m_k such that 2*m_k are all equal mod something... 
    # Or: make cross terms cancel: c_i*c_j = -c_k*c_l for i+j = k+l
    
    # Try: p = x^0 - x^1 + x^2 - x^3 + ... (alternating)
    # p = (1-x^12)/(1-x) if geometric... not quite
    # p = 1 - x + x^2 - x^3 + ... + x^11 (12 terms, alternating)
    # p^2 = (1-x)^2 * (1+x^2+...+x^20)/(1-x^2) ... complex
    
    # Let's try: p = sum of pairs that cancel
    # p = (a0 + a1*x^d) + (a2 + a3*x^d) + ... where within each pair cross terms cancel
    # Actually: p = sum_{k=0}^{5} (c_k * x^{2kd} + c_k * x^{(2k+1)d})
    # = sum_{k=0}^{5} c_k * x^{kd} * (1 + x^d)
    # p^2 = (1+x^d)^2 * (sum c_k x^{kd})^2
    # (1+x^d)^2 = 1 + 2x^d + x^(2d)
    # This gives 3 * (number of terms in (sum c_k x^{kd})^2) terms
    # (sum c_k x^{kd})^2 has at most 11*12/2+1 = 67 terms... too many
    
    # Better: make (sum c_k x^{kd})^2 sparse
    # If c_k are all 1: (1+x^d+...+x^{5d})^2 has 11 nonzero terms (degrees 0 to 10d)
    # Then p^2 has 3*11 - overlaps... roughly 20-30 terms
    
    # Try: p = (1+x^d) * q where q has 6 terms
    # p has 12 terms if q has 6 terms with no overlaps
    # p^2 = (1+x^d)^2 * q^2 = (1+2x^d+x^{2d}) * q^2
    # If q^2 is sparse, so is p^2
    
    # Let q = sum_{k=0}^{5} x^{k*e} (6 terms, geometric)
    # q^2 = sum_{j=0}^{10} (j+1) * x^{j*e} (11 terms)
    # p^2 = (1+2x^d+x^{2d}) * sum_{j=0}^{10}(j+1)x^{je}
    # = sum_{j=0}^{10}(j+1)x^{je} + 2*sum_{j=0}^{10}(j+1)x^{(j+1)d+je} + sum_{j=0}^{10}(j+1)x^{(j+2)d+je}
    # If d and e are coprime and chosen so no overlaps: 33 terms. Not great.
    
    # Alternative: try to find p where p^2 has exactly 12 or fewer terms
    # This is very hard. Let's just do extensive random + local search.
    
    # Phase 4: More aggressive random with different structure
    while time.time() < budget:
        # Random degree, random 12 positions, random values from small set
        degree = rng.randint(11, 40)
        positions = rng.sample(range(degree + 1), 12)
        positions.sort()
        # Ensure degree is used
        if positions[-1] != degree:
            positions[-1] = degree
            positions.sort()
        poly = [0] * (degree + 1)
        for p in positions:
            poly[p] = rng.choice([1, -1, 2, -2])
        s = _square_nonzeros(poly)
        if s < best_score:
            best_poly, best_score = _try_local_opt(poly, min(budget, time.time() + 1.0))
    
    # Phase 5: Try to force cancellation with specific structures
    # p = a + b*x^m + c*x^n where a^2, b^2, c^2, 2ab, 2ac, 2bc all land on few degrees
    # For 12 terms: p = sum of 12 monomials where pairwise products land on few degrees
    
    # Try: all terms have the same coefficient magnitude
    # p = sum_{k} s_k * x^{m_k} where s_k in {1,-1}
    # p^2 = 12 + 2*sum_{i<j} s_i*s_j * x^{m_i+m_j}
    # Nonzero terms: 1 (constant) + number of distinct sums m_i+m_j where s_i*s_j != 0 (always true)
    # minus cancellations where s_i*s_j*x^{m_i+m_j} = -s_k*s_l*x^{m_k+m_l}
    # i.e., m_i+m_j = m_k+m_l and s_i*s_j = -s_k*s_l
    
    # To minimize: maximize number of (i,j) pairs that share the same sum AND have opposite sign products
    # This is like: partition pairs into groups with same sum, and within each group, 
    # maximize cancellation (equal + and - sign products)
    
    while time.time() < budget:
        # Choose 12 positions to maximize sum collisions
        # Use positions from a small set of "sum classes"
        # E.g., positions: 0,1,2,3,4,5, 10,11,12,13,14,15
        # Sums from first group: 0-9 (10 values)
        # Sums from second group: 20-29 (10 values)  
        # Cross sums: 10-19 (10 values)
        # Total distinct sums: 30, so 66 cross pairs + 12 self = 78 terms before cancellation
        # With signs: can cancel some
        
        base1 = rng.randint(0, 5)
        base2 = rng.randint(10, 20)
        positions = list(range(base1, base1+6)) + list(range(base2, base2+6))
        if max(positions) > 40:
            continue
        n = max(positions) + 1
        poly = [0] * n
        for p in positions:
            poly[p] = rng.choice([1, -1])
        s = _square_nonzeros(poly)
        if s < best_score:
            best_poly, best_score = _try_local_opt(poly, min(budget, time.time() + 1.0))
    
    # Phase 6: Try structured cancellation patterns
    # p = (x^a - x^b) + (x^c - x^d) + ... pairs that give cancellation in square
    # (x^a - x^b)^2 = x^{2a} - 2x^{a+b} + x^{2b}
    # Sum of 6 such pairs: 6*3 = 18 terms if no overlaps
    # Can we make overlaps to reduce?
    while time.time() < budget:
        # 6 pairs
        pairs = []
        used = set()
        for _ in range(6):
            a = rng.randint(0, 35)
            b = rng.randint(a+1, 40)
            if a in used or b in used:
                continue
            pairs.append((a, b))
            used.add(a)
            used.add(b)
        if len(pairs) < 6:
            continue
        n = max(max(p[0], p[1]) for p in pairs) + 1
        if n > 41:
            continue
        poly = [0] * n
        for a, b in pairs:
            sign = rng.choice([1, -1])
            poly[a] += sign
            poly[b] -= sign
        # Check exactly 12 nonzero
        if _count_nonzero(poly) != 12:
            continue
        if poly[-1] == 0:
            continue
        s = _square_nonzeros(poly)
        if s < best_score:
            best_poly, best_score = _try_local_opt(poly, min(budget, time.time() + 1.0))
    
    # Ensure final result is valid
    if not _valid(best_poly):
        best_poly = [1] * 12
    
    return best_poly
