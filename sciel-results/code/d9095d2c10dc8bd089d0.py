import time
import random
import itertools

def _square(coeffs):
    """Compute exact integer square of polynomial given as coefficient list (ascending degree)."""
    n = len(coeffs)
    result = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0:
                continue
            result[i + j] += ci * cj
    # Trim trailing zeros but keep at least length 1
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def _nonzero_count(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _is_valid(coeffs):
    """Check all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    nz = _nonzero_count(coeffs)
    if nz != 12:
        return False
    return True

def _score(coeffs):
    """Lower is better: number of nonzero terms in the square."""
    sq = _square(coeffs)
    return _nonzero_count(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    best_poly = None
    best_score = float('inf')
    
    def update_best(poly):
        nonlocal best_poly, best_score
        if _is_valid(poly):
            s = _score(poly)
            if s < best_score:
                best_score = s
                best_poly = poly[:]
    
    # Try structured candidates first
    
    # Family 1: Geometric progression P = sum_{i=0}^{11} a * x^(i*d)
    # P^2 = a^2 * sum_{i,j} x^(i*d + j*d) = a^2 * sum_{k} c_k x^(k*d)
    # where c_k = number of pairs (i,j) with i+j=k, for k=0..22
    # This gives 23 terms in the square (for d>=1). Not great.
    
    # Family 2: P = x^m * (sum of 12 terms with specific structure)
    # Try: P = a*(x^0 + x^2 + x^4 + ... + x^22) - 12 terms, d=2
    # P^2 has terms at even degrees 0..44, with binomial-like coefficients -> 23 terms
    
    # Family 3: Use cancellation. P = A + B where A and B are 6-term each,
    # arranged so cross terms cancel.
    # E.g., P = (x^0 + x^2 + x^4 + x^6 + x^8 + x^10) - (x^1 + x^3 + x^5 + x^7 + x^9 + x^11)
    # This is 12 terms. Let's compute its square.
    
    # Let's just try many structured and random candidates.
    
    random.seed(42)
    
    # Structured: consecutive terms with alternating signs
    for base in range(0, 30):
        for d in [1, 2, 3, 4, 5]:
            for sign_pattern in [[1]*12, [1,-1]*6, [1,1,-1,-1]*3, [1,-1,1,-1]*6]:
                poly = [0] * 41
                for i in range(12):
                    deg = base + i * d
                    if deg < 41:
                        poly[deg] = sign_pattern[i]
                if poly[-1] != 0:
                    update_best(poly)
    
    # Structured: symmetric patterns
    for center in range(5, 35):
        for d in [1, 2, 3]:
            # 12 terms centered around 'center'
            poly = [0] * 41
            for i in range(12):
                deg = center - 5 + i * d
                if 0 <= deg < 41:
                    poly[deg] = 1
            if poly[-1] != 0 and poly[0] is not None:
                # Ensure last nonzero
                nz_positions = [i for i, c in enumerate(poly) if c != 0]
                if nz_positions and poly[nz_positions[-1]] != 0:
                    # Trim to last nonzero
                    last_nz = max(i for i, c in enumerate(poly) if c != 0)
                    poly_trimmed = poly[:last_nz+1]
                    update_best(poly_trimmed)
    
    # Random search with local improvement
    def random_poly():
        # Choose 12 distinct degrees from 0..40
        degrees = random.sample(range(41), 12)
        degrees.sort()
        poly = [0] * 41
        for d in degrees:
            poly[d] = random.choice([-1, 1, -2, 2, -3, 3, -4, 4])
        # Ensure last coefficient nonzero
        if poly[40] == 0:
            poly[degrees[-1]] = random.choice([-1, 1, 2, 3])
        # Trim
        last_nz = max(i for i in range(40, -1, -1) if poly[i] != 0)
        return poly[:last_nz+1]
    
    def local_improve(poly, max_iters=200):
        """Try flipping signs, changing coefficients, moving terms."""
        poly = poly[:]
        best_local = _score(poly)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            nz_indices = [i for i, c in enumerate(poly) if c != 0]
            
            # 1. Flip sign
            for idx in nz_indices:
                old = poly[idx]
                poly[idx] = -old
                s = _score(poly)
                if s < best_local:
                    best_local = s
                    improved = True
                else:
                    poly[idx] = old

            # 2. Change coefficient magnitude
            for idx in nz_indices:
                old = poly[idx]
                for new_val in [-64, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 64]:
                    if new_val == old:
                        continue
                    poly[idx] = new_val
                    s = _score(poly)
                    if s < best_local:
                        best_local = s
                        improved = True
                    else:
                        poly[idx] = old

            # 3. Move term to nearby degree
            for idx in nz_indices:
                old_val = poly[idx]
                for delta in [-2, -1, 1, 2]:
                    new_deg = idx + delta
                    if new_deg < 0 or new_deg >= 41:
                        continue
                    if new_deg == idx:
                        continue
                    
                    # Check if target is occupied in current poly
                    if new_deg < len(poly) and poly[new_deg] != 0:
                        continue
                    
                    # Create new poly
                    new_poly = poly[:]
                    # Extend if necessary
                    while len(new_poly) <= new_deg:
                        new_poly.append(0)
                    
                    new_poly[idx] = 0
                    new_poly[new_deg] = old_val
                    
                    # Trim trailing zeros
                    while len(new_poly) > 1 and new_poly[-1] == 0:
                        new_poly.pop()
                    
                    if not _is_valid(new_poly):
                        continue
                        
                    s = _score(new_poly)
                    if s < best_local:
                        best_local = s
                        poly = new_poly
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return poly, best_local
    
    # Random search loop
    n_tries = 0
    while time.time() < deadline:
        n_tries += 1
        poly = random_poly()
        if not _is_valid(poly):
            continue
        s = _score(poly)
        if s < best_score:
            best_score = s
            best_poly = poly[:]
        
        # Local improvement
        if s <= best_score + 5:  # Only improve promising candidates
            improved_poly, improved_score = local_improve(poly)
            if _is_valid(improved_poly) and improved_score < best_score:
                best_score = improved_score
                best_poly = improved_poly[:]
        
        if time.time() > deadline:
            break
    
    # If we found something good, return it
    if best_poly is not None:
        return best_poly
    
    # Fallback: return a valid 12-term polynomial
    fallback = [0] * 41
    for i in range(12):
        fallback[i] = 1
    return fallback
