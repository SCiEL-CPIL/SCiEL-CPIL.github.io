"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from collections import defaultdict
import itertools

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    def square_nonzero_count_from_terms(terms):
        """
        Compute number of nonzero terms in the square of a polynomial
        given as a list of (exponent, coefficient) pairs.
        """
        sq = defaultdict(int)
        n = len(terms)
        for i in range(n):
            ei, ci = terms[i]
            for j in range(i, n):
                ej, cj = terms[j]
                if i == j:
                    sq[2 * ei] += ci * ci
                else:
                    sq[ei + ej] += 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)
    
    def terms_to_poly(terms):
        """Convert list of (exponent, coefficient) to coefficient list."""
        if not terms:
            return [1]
        max_deg = max(e for e, c in terms)
        poly = [0] * (max_deg + 1)
        for e, c in terms:
            poly[e] = c
        return poly
    
    def is_valid_poly(poly):
        """Check if polynomial meets all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        if any(c < -64 or c > 64 for c in poly):
            return False
        return True

    best_count = 23
    best_terms = [(i, 1) for i in range(12)]
    
    # Structural Search: Superposition of 3 shifted 4-term polynomials
    # P(x) = c1 * Q1(x^s1) + c2 * Q2(x^s2) + c3 * Q3(x^s3)
    # We restrict Qs to be small patterns to keep degree low.
    
    # Base patterns for 4-term polynomials (exponents relative to shift)
    base_patterns = [
        [0, 1, 2, 3],       # Consecutive
        [0, 1, 2, 4],
        [0, 1, 3, 4],
        [0, 2, 4, 6],       # Even
        [0, 1, 3, 5],
        [0, 1, 4, 5],
        [0, 2, 3, 5],
        [0, 1, 2, 5],
        [0, 1, 3, 6],
        [0, 2, 4, 8],       # Powers of 2
        [0, 1, 2, 6],
        [0, 1, 4, 6]
    ]
    
    # Coefficients for the 4 terms in the base pattern
    # We will search over sign patterns for the terms in Q
    # And overall scale c_i
    
    def generate_candidate(shifts, pattern_idx, sign_mask, scale):
        """
        Generate a candidate polynomial from structural parameters.
        shifts: list of 3 shifts
        pattern_idx: list of 3 indices into base_patterns
        sign_mask: list of 3 masks (4 bits each) for signs of the 4 terms
        scale: list of 3 scales (integers)
        """
        terms_dict = defaultdict(int)
        valid = True
        
        for i in range(3):
            s = shifts[i]
            pat = base_patterns[pattern_idx[i]]
            mask = sign_mask[i]
            sc = scale[i]
            
            for j in range(4):
                exp = s + pat[j]
                if exp > 40:
                    valid = False
                    break
                sign = 1 if (mask >> j) & 1 else -1
                coeff = sign * sc
                if coeff == 0:
                    valid = False
                    break
                terms_dict[exp] += coeff
        
        if not valid:
            return None
            
        # Check if exactly 12 nonzero terms
        # Note: If terms overlap and cancel, count might drop.
        # If terms overlap and add, count stays same but coeff changes.
        # We need exactly 12 nonzero entries in the final dict.
        
        final_terms = []
        for e, c in terms_dict.items():
            if c != 0:
                final_terms.append((e, c))
                
        if len(final_terms) != 12:
            return None
            
        # Check coeff bounds
        for e, c in final_terms:
            if c < -64 or c > 64:
                return None
                
        return final_terms

    # Search Space:
    # shifts: 3 integers in [0, 15] (to keep degree < 40 with max pattern span ~8)
    # pattern_idx: 3 integers in [0, 11]
    # sign_mask: 3 integers in [0, 15]
    # scale: 3 integers in [1, 2, 3, 4]
    
    # This space is 16^3 * 12^3 * 16^3 * 4^3 = 4096 * 1728 * 4096 * 64 ~ 1.8e10. Too large for full grid.
    # We will use random sampling with bias towards structured configurations.
    
    rng = random.Random(42)
    
    # Phase 1: Random sampling of the structural space
    samples = 20000
    for _ in range(samples):
        if time.time() > deadline:
            break
            
        # Choose shifts
        # To avoid overlap, shifts should be somewhat separated, but we allow overlap for cancellation
        shifts = [rng.randint(0, 15) for _ in range(3)]
        # Sort shifts to canonicalize? No, order matters for assignment to patterns
        # But we can sort the resulting terms later.
        
        # Choose patterns
        pat_idxs = [rng.randint(0, len(base_patterns)-1) for _ in range(3)]
        
        # Choose signs
        masks = [rng.randint(0, 15) for _ in range(3)]
        
        # Choose scales
        scales = [rng.choice([1, 1, 2, 3]) for _ in range(3)]
        
        terms = generate_candidate(shifts, pat_idxs, masks, scales)
        if terms:
            cnt = square_nonzero_count_from_terms(terms)
            if cnt < best_count:
                best_count = cnt
                best_terms = terms[:]
                
    # Phase 2: Local search on the best structural candidate found
    # If we found a good one, we can refine it by small perturbations of shifts/scales
    
    if best_terms:
        # Extract parameters? Hard to invert. 
        # Instead, do a generic local search on the 12 terms found, 
        # but restricted to moves that preserve the "structure" loosely?
        # Actually, just do a standard local search on the terms, 
        # but starting from the structural seed.
        
        current = best_terms[:]
        current_count = best_count
        
        # Standard local search
        no_improve = 0
        max_no_improve = 100
        rng2 = random.Random(99)
        
        while no_improve < max_no_improve and time.time() < deadline:
            best_neighbor = None
            best_neighbor_count = current_count
            
            # Try changing one exponent
            for idx in range(12):
                old_e = current[idx][0]
                # Try nearby exponents
                for delta in [-2, -1, 1, 2]:
                    new_e = old_e + delta
                    if new_e < 0 or new_e > 40:
                        continue
                    exps = set(e for e, c in current)
                    exps.discard(old_e)
                    if new_e in exps:
                        continue
                    trial = current[:]
                    trial[idx] = (new_e, current[idx][1])
                    cnt = square_nonzero_count_from_terms(trial)
                    if cnt < best_neighbor_count:
                        best_neighbor_count = cnt
                        best_neighbor = trial[:]
                        
            # Try changing one coefficient
            for idx in range(12):
                old_c = current[idx][1]
                for new_c in [-3, -2, -1, 1, 2, 3]:
                    if new_c == old_c:
                        continue
                    trial = current[:]
                    trial[idx] = (current[idx][0], new_c)
                    cnt = square_nonzero_count_from_terms(trial)
                    if cnt < best_neighbor_count:
                        best_neighbor_count = cnt
                        best_neighbor = trial[:]
                        
            if best_neighbor:
                current = best_neighbor
                current_count = best_neighbor_count
                no_improve = 0
                if current_count < best_count:
                    best_count = current_count
                    best_terms = current[:]
            else:
                no_improve += 1
                # Random perturbation
                if no_improve % 10 == 0:
                    idx = rng2.randint(0, 11)
                    new_e = rng2.randint(0, 40)
                    exps = set(e for e, c in current)
                    if new_e not in exps:
                        trial = current[:]
                        trial[idx] = (new_e, current[idx][1])
                        cnt = square_nonzero_count_from_terms(trial)
                        if cnt < current_count:
                            current = trial
                            current_count = cnt
                            no_improve = 0
                            if current_count < best_count:
                                best_count = current_count
                                best_terms = current[:]

    # Fallback: If structural search didn't find anything better than seed, 
    # run a quick generic SA as in original, but shorter.
    if best_count >= 23:
         # Re-run a very short version of the original logic or just return seed
         # To save time, we'll just return the best found (which is seed if no improvement)
         pass

    best_poly = terms_to_poly(best_terms)
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        
    nonzero_count = sum(1 for c in best_poly if c != 0)
    if nonzero_count != 12:
        return [1] * 12
        
    for i, c in enumerate(best_poly):
        best_poly[i] = int(c)
        if best_poly[i] < -64:
            best_poly[i] = -64
        elif best_poly[i] > 64:
            best_poly[i] = 64
            
    if not is_valid_poly(best_poly):
        return [1] * 12
        
    return best_poly
# EVOLVE-BLOCK-END
