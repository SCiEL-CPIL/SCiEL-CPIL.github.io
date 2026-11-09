import time
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = 80.0  # safety margin before 90s kill

def _sq_nonzero_count(coefs):
    """Count nonzero coefficients in the exact square of a polynomial."""
    n = len(coefs)
    # Build square coefficients
    sq = [0] * (2 * n - 1)
    for i in range(n):
        a = coefs[i]
        if a == 0:
            continue
        for j in range(i, n):
            v = a * a if i == j else 2 * a * coefs[j]
            if v != 0:
                sq[i + j] += v
    cnt = 0
    for c in sq:
        if c != 0:
            cnt += 1
    return cnt

def _valid(coefs):
    if len(coefs) > 41:
        return False
    nz = 0
    for c in coefs:
        if not isinstance(c, int):
            return False
        if c < -64 or c > 64:
            return False
        if c != 0:
            nz += 1
    if nz != 12:
        return False
    if coefs[-1] == 0:
        return False
    return True

def _count_nz(coefs):
    """Count nonzero entries in a coefficient list."""
    return sum(1 for c in coefs if c != 0)

def _polynomial_product(p1, p2):
    """Multiply two polynomials represented as dicts {degree: coeff}."""
    result = defaultdict(int)
    for d1, c1 in p1.items():
        if c1 == 0:
            continue
        for d2, c2 in p2.items():
            if c2 == 0:
                continue
            result[d1 + d2] += c1 * c2
    return dict(result)

def _poly_from_dict(d, length=41):
    """Convert dict {degree: coeff} to coefficient list of given length."""
    coefs = [0] * length
    for deg, c in d.items():
        if 0 <= deg < length:
            coefs[deg] = c
    return coefs

def _factor_search_2(binomial_pairs, rng, best):
    """
    Search over products of two binomials: P = (a1*x^p1 + a2*x^p2) * (b1*x^q1 + b2*x^q2)
    This gives P with up to 4 terms. P^2 has structure from A^2 * B^2.
    We extend by adding more binomial factors or using trinomials.
    """
    # Try P = product of k binomials, where k is chosen so P has exactly 12 nonzero terms
    # A product of k binomials has at most 2^k terms.
    # For 12 terms, we need combinations that give exactly 12.
    # Alternatively, use a mix: some binomials, some trinomials.
    
    # Strategy: P = A * B where A has 6 terms, B has 2 terms (binomial)
    # Or A has 4 terms, B has 3 terms
    # Or A has 3 terms, B has 4 terms
    # Or A has 2 terms, B has 6 terms
    
    # Let's focus on A = product of 3 binomials (up to 8 terms) and B = binomial (2 terms)
    # P = A * B has up to 16 terms, we need exactly 12 nonzero.
    
    # Simpler: directly construct P with 12 terms as a product of factors
    # P = (x^a1 + x^a2 + x^a3) * (x^b1 + x^b2 + x^b3) * (x^c1 + x^c2 + x^c3)
    # This can give up to 27 terms, but with cancellation we can get 12.
    
    # Even simpler and more targeted: 
    # P = (x^a + x^b) * (x^c + x^d + x^e + x^f + x^g + x^h)
    # First factor: 2 terms, second: 6 terms -> P has up to 12 terms
    # If all products are distinct, P has exactly 12 terms!
    
    # Let's search over such structures
    for _ in range(5000):
        if time.time() > _DEADLINE:
            break
        
        # Choose binomial part: 2 exponents
        a = rng.randrange(0, 10)
        b = a + rng.randrange(1, 15)
        
        # Choose 6 exponents for the other factor, all >= 0, max such that 2*max <= 40
        max_deg = 40 // 2
        exponents = set()
        while len(exponents) < 6:
            exponents.add(rng.randrange(0, max_deg + 1))
        exponents = sorted(exponents)
        
        # Build P = (x^a + s1*x^b) * sum(s2_i * x^e_i)
        # Try different sign patterns
        s1 = rng.choice([-1, 1])
        s2s = [rng.choice([-1, 1]) for _ in range(6)]
        
        # Compute P coefficients
        P = defaultdict(int)
        for e, s2 in zip(exponents, s2s):
            P[a + e] += s1 * s2
            P[b + e] += s2
        
        # Check if P has exactly 12 nonzero terms
        nz = sum(1 for v in P.values() if v != 0)
        if nz != 12:
            continue
        
        # Check degree constraint
        max_p = max(P.keys())
        if max_p > 40:
            continue
        
        # Check that the highest degree term is nonzero
        if P.get(max_p, 0) == 0:
            continue
        
        # Convert to coefficient list
        coefs = _poly_from_dict(P, 41)
        
        # Check coefficient bounds
        if any(abs(c) > 64 for c in coefs):
            continue
        
        # Check validity
        if not _valid(coefs):
            continue
        
        # Count nonzero in square
        sc = _sq_nonzero_count(coefs)
        if sc < best[0]:
            best[0] = sc
            best[1] = coefs[:]
    
    return best

def _factor_search_3(rng, best):
    """
    Search over P = A * B * C where A, B, C are binomials or trinomials.
    Focus on structures that naturally produce 12 terms.
    """
    for _ in range(3000):
        if time.time() > _DEADLINE:
            break
        
        # Choose three factors, each with 2 or 3 terms
        n_terms = [rng.choice([2, 3]) for _ in range(3)]
        
        # Generate exponents for each factor
        factors = []
        valid = True
        for nt in n_terms:
            exps = set()
            while len(exps) < nt:
                exps.add(rng.randrange(0, 15))
            exps = sorted(exps)
            signs = [rng.choice([-1, 1]) for _ in range(nt)]
            factor = {e: s for e, s in zip(exps, signs)}
            factors.append(factor)
        
        # Multiply factors
        P = {0: 1}
        for f in factors:
            P = _polynomial_product(P, f)
        
        # Check if P has exactly 12 nonzero terms
        nz = sum(1 for v in P.values() if v != 0)
        if nz != 12:
            continue
        
        max_p = max(P.keys())
        if max_p > 40 or P.get(max_p, 0) == 0:
            continue
        
        coefs = _poly_from_dict(P, 41)
        
        if any(abs(c) > 64 for c in coefs):
            continue
        
        if not _valid(coefs):
            continue
        
        sc = _sq_nonzero_count(coefs)
        if sc < best[0]:
            best[0] = sc
            best[1] = coefs[:]
    
    return best

def _cancellation_target_search(rng, best):
    """
    Targeted search: for a given support, find coefficient values that cause
    specific cancellations in P^2.
    
    Key idea: if we want P^2[d] = 0 for some degree d, we need the sum of
    products a_i * a_j where i+j=d to be zero.
    
    We can set up this as a constraint satisfaction problem.
    """
    # Generate random supports with structure that allows cancellation
    for _ in range(2000):
        if time.time() > _DEADLINE:
            break
        
        # Generate a support with some symmetry to encourage cancellation
        # Use a base set and add symmetric counterparts
        base = set()
        while len(base) < 7:
            base.add(rng.randrange(0, 25))
        base = sorted(base)
        
        # Create symmetric support: for each d in base, also consider 40-d
        support_set = set(base)
        for d in base:
            support_set.add(40 - d)
        
        # Trim to 12 elements, ensuring 40 is included
        support_list = sorted(support_set)
        if 40 not in support_list:
            support_list = support_list[:11] + [40]
            support_list = sorted(set(support_list))
        
        if len(support_list) != 12:
            continue
        
        support = support_list
        hi = max(support)
        
        # Assign values with structure to encourage cancellation
        # Use small integers with some negative values
        val_pool = [-4, -3, -2, -1, 1, 2, 3, 4]
        
        # Try several random assignments
        for _attempt in range(20):
            if time.time() > _DEADLINE:
                break
            
            coefs = [0] * 41
            for d in support:
                coefs[d] = rng.choice(val_pool)
            
            if coefs[hi] == 0:
                coefs[hi] = 1
            
            if not _valid(coefs):
                continue
            
            sc = _sq_nonzero_count(coefs)
            if sc < best[0]:
                best[0] = sc
                best[1] = coefs[:]
            
            # Local polish: try to force one more cancellation
            # Find a degree where P^2 is nonzero and try to adjust coefficients
            if sc > 12:
                sq = [0] * (2 * 41 - 1)
                for i in range(41):
                    a = coefs[i]
                    if a == 0:
                        continue
                    for j in range(i, 41):
                        v = a * a if i == j else 2 * a * coefs[j]
                        if v != 0:
                            sq[i + j] += v
                
                # Find a degree with nonzero square coefficient
                nonzero_degs = [d for d in range(len(sq)) if sq[d] != 0]
                if nonzero_degs:
                    target_d = rng.choice(nonzero_degs)
                    
                    # Try to adjust one coefficient to reduce |sq[target_d]|
                    for d in support:
                        if time.time() > _DEADLINE:
                            break
                        old_val = coefs[d]
                        for delta in [-2, -1, 1, 2]:
                            new_val = old_val + delta
                            if new_val == 0 or abs(new_val) > 64:
                                continue
                            coefs[d] = new_val
                            if not _valid(coefs):
                                coefs[d] = old_val
                                continue
                            sc2 = _sq_nonzero_count(coefs)
                            if sc2 < best[0]:
                                best[0] = sc2
                                best[1] = coefs[:]
                                coefs[d] = old_val
                                break
                            coefs[d] = old_val
    
    return best

def _polish(best, support, rng):
    """Local hill-climbing polish: try changing one coefficient at a time."""
    n = len(support)
    hi = max(support)
    best_local = best[1][:] if best[1] else None
    if best_local is None:
        return best
    best_score = best[0]
    
    # Try multiple polish passes
    for _pass in range(5):
        if time.time() > _DEADLINE:
            break
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for idx, d in enumerate(support):
                if time.time() > _DEADLINE:
                    break
                old_val = best_local[d]
                # Try various new values
                candidates = []
                for v in [-64,-32,-16,-8,-4,-3,-2,-1,1,2,3,4,8,16,32,64]:
                    if v != old_val:
                        candidates.append(v)
                # Also try small perturbations
                for delta in [-4,-3,-2,-1,1,2,3,4]:
                    v = old_val + delta
                    if -64 <= v <= 64 and v != 0 and v != old_val:
                        candidates.append(v)
                rng.shuffle(candidates)
                for new_val in candidates[:10]:
                    if time.time() > _DEADLINE:
                        break
                    best_local[d] = new_val
                    if not _valid(best_local):
                        best_local[d] = old_val
                        continue
                    sc = _sq_nonzero_count(best_local)
                    if sc < best_score:
                        best_score = sc
                        best[0] = sc
                        best[1] = best_local[:]
                        improved = True
                    else:
                        best_local[d] = old_val
    
    return best

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 80.0
    rng = random.Random(12345)
    best = [10**9, None]
    
    # Phase 1: Factorization-based search (2-factor: binomial * 6-term)
    best = _factor_search_2(None, rng, best)
    
    # Phase 2: Factorization-based search (3-factor)
    best = _factor_search_3(rng, best)
    
    # Phase 3: Targeted cancellation search
    best = _cancellation_target_search(rng, best)
    
    # Phase 4: Final polish
    if best[1] is not None:
        support = [i for i in range(41) if best[1][i] != 0]
        _polish(best, support, rng)
    
    if best[1] is None:
        coefs = [0] * 41
        for i in range(12):
            coefs[i] = 1
        return coefs
    
    return best[1]
