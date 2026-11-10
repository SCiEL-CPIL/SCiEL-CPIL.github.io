import time
import random
import numpy as np

def _square_coefficients(poly):
    """Compute the coefficients of poly^2 as a list in ascending degree."""
    n = len(poly)
    result = [0] * (2 * n - 1)
    for i in range(n):
        if poly[i] == 0:
            continue
        for j in range(n):
            if poly[j] != 0:
                result[i + j] += poly[i] * poly[j]
    return result

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # We want exactly 12 nonzero terms, coefficients in [-64, 64], degree <= 40
    # Minimize number of nonzero terms in the square
    
    best_poly = None
    best_sq_nonzero = float('inf')
    
    # Strategy: Try structured patterns and local search
    # Key insight: if P(x) = sum a_i x^{e_i}, then P(x)^2 has terms at e_i + e_j
    # If we can arrange that many pairs (i,j) produce the same sum, cancellations can occur
    
    # Try several families of exponents
    def try_poly(poly):
        nonlocal best_poly, best_sq_nonzero
        # Check constraints
        if len(poly) > 41:
            return
        if poly[-1] == 0:
            # Trim trailing zeros
            while len(poly) > 1 and poly[-1] == 0:
                poly.pop()
        if len(poly) > 41:
            return
        nz = [c for c in poly if c != 0]
        if len(nz) != 12:
            return
        if any(abs(c) > 64 for c in nz):
            return
        sq = _square_coefficients(poly)
        cnt = _count_nonzero(sq)
        if cnt < best_sq_nonzero:
            best_sq_nonzero = cnt
            best_poly = list(poly)
    
    # Family 1: Symmetric patterns - P(x) = x^k * Q(x) where Q is palindromic
    # If Q is palindromic, P^2 might have structure
    
    # Try geometric progression of exponents
    # P = 1 + x^d + x^{2d} + ... + x^{11d} -> square has many terms
    # Better: P = 1 - x^d + x^{2d} - ... (alternating signs)
    
    # Let's try: P = 1 + x^a + x^{2a} + x^{3a} (4 terms) times something
    # Actually, let's think about what gives sparse squares.
    
    # If P(x) = A(x) * B(x) where A and B have few terms and their product square is sparse...
    # Actually P^2 = (A*B)^2 = A^2 * B^2
    # If A^2 and B^2 are both sparse, the product might be sparse.
    
    # Try: P = (1 + x^a)(1 + x^b)(1 + x^c)(1 + x^d)(1 + x^e)(1 + x^f)
    # That's 2^6 = 64 terms in the expansion - too many nonzero terms in P.
    
    # Better approach: P with terms that cancel in the square.
    # If P = 1 + x^a, then P^2 = 1 + 2x^a + x^{2a} (3 terms)
    # If P = 1 + x^a - x^{2a}, P^2 = 1 + 2x^a - x^{2a} - 2x^{3a} + x^{4a} (5 terms)
    
    # Key: if we can make P such that P^2 = Q where Q is very sparse
    # e.g., P^2 = 1 + x^{2k} (2 terms) -> P = sqrt(1+x^{2k}) - not polynomial
    
    # Let's try random search with local improvement
    random.seed(42)
    
    # Initialize with some structured candidates
    # Candidate: exponents in arithmetic progression with alternating signs
    for d in range(1, 5):
        for start in range(0, 41 - 11*d):
            poly = [0] * (start + 11*d + 1)
            for i in range(12):
                sign = 1 if i % 2 == 0 else -1
                poly[start + i*d] = sign
            try_poly(poly)
    
    # Try: all same sign, arithmetic progression
    for d in range(1, 5):
        for start in range(0, 41 - 11*d):
            poly = [0] * (start + 11*d + 1)
            for i in range(12):
                poly[start + i*d] = 1
            try_poly(poly)
    
    # Now do local search from best found
    if best_poly is not None:
        current = list(best_poly)
        current_score = best_sq_nonzero
        no_improve_count = 0
        max_no_improve = 50000
        
        while time.time() < deadline - 2.0 and no_improve_count < max_no_improve:
            # Pick a random modification
            mod_type = random.choice(['coeff', 'exponent', 'add_term', 'remove_term'])
            
            trial = list(current)
            
            if mod_type == 'coeff':
                # Change a coefficient
                nz_indices = [i for i, c in enumerate(trial) if c != 0]
                if not nz_indices:
                    continue
                idx = random.choice(nz_indices)
                old_val = trial[idx]
                new_val = random.randint(-64, 64)
                if new_val == old_val:
                    continue
                trial[idx] = new_val
                if new_val == 0:
                    # This would remove a term, check if still 12
                    pass
            
            elif mod_type == 'exponent':
                # Move a term to a nearby position
                nz_indices = [i for i, c in enumerate(trial) if c != 0]
                if len(nz_indices) < 2:
                    continue
                idx = random.choice(nz_indices)
                val = trial[idx]
                trial[idx] = 0
                # Try nearby positions
                for offset in random.sample(range(-3, 4), min(3, 6)):
                    new_idx = idx + offset
                    if 0 <= new_idx <= 40 and trial[new_idx] == 0:
                        trial[new_idx] = val
                        break
                else:
                    trial[idx] = val  # revert
                    continue
            
            elif mod_type == 'add_term':
                # This would make 13 terms, so we need to also remove one
                pass  # Skip for now
            
            else:  # remove_term
                # Remove a term and add it elsewhere
                nz_indices = [i for i, c in enumerate(trial) if c != 0]
                if len(nz_indices) < 12:
                    continue
                idx = random.choice(nz_indices)
                val = trial[idx]
                trial[idx] = 0
                # Add to a new position
                possible = [i for i in range(len(trial)) if trial[i] == 0]
                if possible:
                    new_idx = random.choice(possible)
                    trial[new_idx] = val
                else:
                    trial[idx] = val
                    continue
            
            # Validate trial
            if len(trial) > 41:
                continue
            while len(trial) > 1 and trial[-1] == 0:
                trial.pop()
            if len(trial) > 41:
                continue
            nz_count = sum(1 for c in trial if c != 0)
            if nz_count != 12:
                continue
            if any(abs(c) > 64 for c in trial):
                continue
            
            sq = _square_coefficients(trial)
            cnt = _count_nonzero(sq)
            
            if cnt < current_score:
                current = trial
                current_score = cnt
                best_sq_nonzero = cnt
                best_poly = list(trial)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Occasionally do a larger jump
            if no_improve_count % 1000 == 0 and time.time() < deadline - 5.0:
                # Random restart with bias toward current structure
                trial = [0] * random.randint(10, 41)
                positions = sorted(random.sample(range(len(trial)), 12))
                for pos in positions:
                    trial[pos] = random.choice([-1, 1, 2, -2, 3, -3, 5, -5, 8, -8])
                # Ensure last is nonzero
                if trial[-1] == 0:
                    trial[positions[-1]] = 1
                
                nz_count = sum(1 for c in trial if c != 0)
                if nz_count == 12:
                    sq = _square_coefficients(trial)
                    cnt = _count_nonzero(sq)
                    if cnt < current_score:
                        current = list(trial)
                        current_score = cnt
                        best_sq_nonzero = cnt
                        best_poly = list(trial)
    
    # Final validation
    if best_poly is not None:
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
        if len(best_poly) <= 41 and best_poly[-1] != 0:
            nz = [c for c in best_poly if c != 0]
            if len(nz) == 12 and all(abs(c) <= 64 for c in nz):
                return best_poly
    
    # Fallback: return a valid 12-term polynomial
    return [1] * 12
