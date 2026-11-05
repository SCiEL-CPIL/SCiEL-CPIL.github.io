import time
import random
import math
import itertools

def _count_sq_nonzero(coeff_list):
    """Count nonzero terms in the square of polynomial given as coefficient list."""
    n = len(coeff_list)
    # Determine max degree
    max_deg = 0
    for i, c in enumerate(coeff_list):
        if c != 0:
            max_deg = i
    # Square has degree 2 * max_deg
    sq = [0] * (2 * max_deg + 1)
    
    # Extract nonzero terms (exponent, coeff)
    terms = [(i, c) for i, c in enumerate(coeff_list) if c != 0]
    
    # Convolve
    for i, ci in terms:
        for j, cj in terms:
            sq[i + j] += ci * cj
            
    return sum(1 for c in sq if c != 0)

def _validate(coeff_list):
    """Check if coefficient list is valid. Returns True if valid."""
    if len(coeff_list) > 41:
        return False
    if len(coeff_list) < 1:
        return False
    if coeff_list[-1] == 0:
        return False
    nz = sum(1 for c in coeff_list if c != 0)
    if nz != 12:
        return False
    for c in coeff_list:
        if c < -64 or c > 64:
            return False
    return True

def _build_poly(exps, coefs):
    """Build dense coefficient list from sparse representation."""
    if not exps:
        return [0]
    max_deg = max(exps)
    poly = [0] * (max_deg + 1)
    for e, c in zip(exps, coefs):
        poly[e] = c
    return poly

def _hill_climb(exps, init_coefs, max_iter=200, deadline=None):
    """Perform local search on coefficients for a fixed set of exponents."""
    current_coefs = list(init_coefs)
    poly = _build_poly(exps, current_coefs)
    current_score = _count_sq_nonzero(poly)
    
    best_coefs = list(current_coefs)
    best_score = current_score
    
    # Small neighborhood for coefficients: change one coeff to a nearby small integer
    # We restrict values to [-5, 5] mostly, as large values rarely help cancellation in sparse squares
    candidates = [-3, -2, -1, 0, 1, 2, 3] # 0 is allowed temporarily during search, but final must be nonzero?
    # Actually, if a coeff becomes 0, the term count drops below 12.
    # The problem states "exactly 12 nonzero terms".
    # So we must ensure we don't land on a solution with < 12 terms.
    # However, during intermediate steps, we might allow 0s if we plan to fix them?
    # Better: Only mutate to non-zero values, or if we allow 0, we must check validity at the end.
    # Since the score is based on the square, and validity is checked at the end,
    # we can allow 0s in intermediate states as long as we return a valid state.
    # But _count_sq_nonzero works fine with 0s.
    # We will just track the best VALID state.
    
    T = 1.0
    no_improve = 0
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Pick a random index to mutate
        idx = random.randint(0, 11)
        old_val = current_coefs[idx]
        
        # Generate new value
        # Bias towards small integers
        new_val = random.choice(candidates)
        if new_val == old_val:
            continue
            
        current_coefs[idx] = new_val
        
        poly = _build_poly(exps, current_coefs)
        new_score = _count_sq_nonzero(poly)
        
        # Check validity of current state for "best" tracking
        # Validity: len<=41, last!=0, exactly 12 non-zeros, range [-64,64]
        is_valid = True
        if new_val < -64 or new_val > 64:
            is_valid = False
        # Check nonzero count
        nz_count = sum(1 for c in current_coefs if c != 0)
        if nz_count != 12:
            is_valid = False
        # Check last coeff of poly (which is at max(exps))
        if poly[-1] == 0:
            is_valid = False
            
        if is_valid and new_score < best_score:
            best_score = new_score
            best_coefs = list(current_coefs)
            
        delta = new_score - current_score
        
        if delta < 0:
            # Accept
            current_score = new_score
            no_improve = 0
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                current_score = new_score
                no_improve += 1
            else:
                # Revert
                current_coefs[idx] = old_val
                no_improve += 1
                
        if no_improve > 50:
            T *= 0.8
            if T < 0.1:
                break
            no_improve = 0

    return best_coefs, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_sq = 999
    best_coeffs = None
    
    def _time_check():
        return time.time() > deadline

    # Strategy:
    # 1. Generate structured exponent sets (APs, unions of APs)
    # 2. For each, try structured coefficient patterns (alternating signs, small ints)
    # 3. Run hill climbing from these starts
    
    exp_sets = []
    
    # 1. Arithmetic Progressions
    # a + i*d for i in 0..11
    # max degree <= 40 => a + 11d <= 40
    for d in range(1, 4):
        for a in range(0, 41 - 11 * d + 1):
            exps = tuple(a + i * d for i in range(12))
            exp_sets.append(exps)
            
    # 2. Union of two APs? e.g., 6 terms from AP1 and 6 from AP2
    # This might create more complex collision structures
    # Example: {0, 2, 4, 6, 8, 10} U {12, 14, 16, 18, 20, 22}
    for d1 in range(1, 3):
        for d2 in range(1, 3):
            for a1 in range(0, 20):
                for a2 in range(20, 41):
                    exps1 = tuple(a1 + i * d1 for i in range(6))
                    exps2 = tuple(a2 + i * d2 for i in range(6))
                    exps = tuple(sorted(exps1 + exps2))
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        exp_sets.append(exps)
                        
    # 3. Random sets (fewer, to save time)
    random.seed(42)
    for _ in range(20):
        exps = tuple(sorted(random.sample(range(0, 41), 12)))
        exp_sets.append(exps)
        
    exp_sets = list(set(exp_sets))
    
    # Coefficient Patterns to try as seeds
    # Pattern 1: All 1s
    # Pattern 2: Alternating 1, -1
    # Pattern 3: Alternating 1, -2
    # Pattern 4: Alternating 2, -1
    # Pattern 5: Random small ints
    def gen_patterns():
        pats = []
        # All 1s
        pats.append([1]*12)
        # Alternating 1, -1
        pats.append([1 if i%2==0 else -1 for i in range(12)])
        # Alternating -1, 1
        pats.append([-1 if i%2==0 else 1 for i in range(12)])
        # Alternating 1, -2
        pats.append([1 if i%2==0 else -2 for i in range(12)])
        # Alternating 2, -1
        pats.append([2 if i%2==0 else -1 for i in range(12)])
        # Alternating 1, -3
        pats.append([1 if i%2==0 else -3 for i in range(12)])
        # Alternating 3, -1
        pats.append([3 if i%2==0 else -1 for i in range(12)])
        
        # Random small ints
        for _ in range(5):
            pats.append([random.choice([-2,-1,1,2]) for _ in range(12)])
            
        return pats

    patterns = gen_patterns()
    
    # Shuffle exp_sets to ensure we don't get stuck on bad ones if time is short
    random.shuffle(exp_sets)
    
    for exps in exp_sets:
        if _time_check():
            break
            
        for pat in patterns:
            if _time_check():
                break
                
            # Ensure pattern is valid for this exp set (no zeros in pattern)
            if 0 in pat:
                continue
                
            # Run hill climb
            coefs, score = _hill_climb(exps, pat, max_iter=50, deadline=deadline)
            
            # Validate the result
            poly = _build_poly(exps, coefs)
            if _validate(poly):
                if score < best_sq:
                    best_sq = score
                    best_coeffs = list(poly)
                    
    # If we still have time, do some random restarts with hill climbing on random exps
    if not _time_check():
        for _ in range(50):
            if _time_check():
                break
            exps = tuple(sorted(random.sample(range(0, 41), 12)))
            pat = [random.choice([-2,-1,1,2]) for _ in range(12)]
            coefs, score = _hill_climb(exps, pat, max_iter=50, deadline=deadline)
            poly = _build_poly(exps, coefs)
            if _validate(poly):
                if score < best_sq:
                    best_sq = score
                    best_coeffs = list(poly)

    if best_coeffs is not None:
        return best_coeffs
    
    # Fallback: A valid polynomial
    # 12 terms, e.g., 0..11
    return [1] * 12
