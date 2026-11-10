import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def count_nonzero_square(coeffs, exponents):
        n = len(coeffs)
        max_deg = 2 * max(exponents)
        square = [0] * (max_deg + 1)
        for i in range(n):
            ci = coeffs[i]
            ei = exponents[i]
            square[ei + ei] += ci * ci
            for j in range(i + 1, n):
                deg = ei + exponents[j]
                square[deg] += 2 * ci * coeffs[j]
        return sum(1 for v in square if v != 0)

    def make_result(coeffs, exponents):
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for c, e in zip(coeffs, exponents):
            result[e] = c
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result

    def validate(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(abs(c) > 64 for c in coeffs):
            return False
        if max(exponents) * 2 + 1 > 41:
            return False
        return True

    best_nonzero = 78
    best_result = None

    # Strategy 1: Arithmetic progression exponents with structured coefficient search
    # For exponents 0,1,...,11, square degree k gets pairs (i,j) with i+j=k
    # We want to maximize zeros in the square
    
    # Precompute pair structures for AP exponents
    def get_pairs_for_degree(k, n=12):
        """Get all pairs (i,j) with i<=j and i+j=k for n-term AP"""
        pairs = []
        for i in range(max(0, k - (n-1)), min(n-1, k) + 1):
            j = k - i
            if j >= i and j < n:
                pairs.append((i, j))
        return pairs

    # For AP exponents 0..11, try to find coefficient patterns with max cancellation
    # Focus on small coefficient sets: {-2,-1,1,2} and {-3,-2,-1,1,2,3}
    
    # Phase 1: Exhaustive search over small coefficient sets for AP exponents
    for step in [1, 2]:
        exponents = [i * step for i in range(12)]
        if max(exponents) * 2 + 1 > 41:
            continue
        
        # For step=1: degrees 0..22, 23 possible degrees
        # For step=2: degrees 0..44 but max is 22*2=44 > 40, skip
        # Actually step=2: max exponent = 22, 2*22+1 = 45 > 41, skip
        if step > 1:
            continue
        
        # Try coefficient patterns from small sets
        # Use symmetry: fix c[0]=1 (sign symmetry), search remaining
        coeff_values = [-2, -1, 1, 2]
        
        # Systematic search with pruning
        # For each degree k in the square, the coefficient is:
        # sum over pairs (i,j) with i+j=k of (ci*cj if i==j else 2*ci*cj)
        # We want this to be 0 for as many k as possible
        
        # Try all combinations from {-2,-1,1,2}^12 with c[0]=1
        # That's 4^11 = 4M, too many. Use smarter approach.
        
        # Phase 1a: Try periodic patterns
        for period in [2, 3, 4, 6]:
            for base in product(coeff_values, repeat=period):
                if 0 in base:
                    continue
                coeffs = [base[i % period] for i in range(12)]
                if coeffs[0] != 1:
                    coeffs = [c * (1 // coeffs[0]) if coeffs[0] != 0 else 1 for c in coeffs]
                    # This normalization is wrong, just skip
                    continue
                nz = count_nonzero_square(coeffs, exponents)
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_result = make_result(coeffs, exponents)
        
        # Phase 1b: Try symmetric patterns c[i] = s * c[11-i]
        for sign in [1, -1]:
            for half in product(coeff_values, repeat=6):
                coeffs = list(half) + [sign * half[i] for i in range(5, -1, -1)]
                if 0 in coeffs or any(abs(c) > 64 for c in coeffs):
                    continue
                nz = count_nonzero_square(coeffs, exponents)
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_result = make_result(coeffs, exponents)
        
        # Phase 1c: Try antisymmetric patterns c[i] = -c[11-i]
        for half in product(coeff_values, repeat=6):
            coeffs = list(half) + [-half[i] for i in range(5, -1, -1)]
            if 0 in coeffs or any(abs(c) > 64 for c in coeffs):
                continue
            nz = count_nonzero_square(coeffs, exponents)
            if nz < best_nonzero:
                best_nonzero = nz
                best_result = make_result(coeffs, exponents)

    # Phase 2: Backtracking with constraint propagation for AP exponents
    # Directly enforce zeros in the square
    
    exponents = list(range(12))
    
    # Precompute which pairs contribute to each degree
    degree_pairs = {}
    for k in range(23):
        degree_pairs[k] = get_pairs_for_degree(k)
    
    # Backtracking: assign coefficients one by one, check partial constraints
    # When we've assigned enough coefficients to determine a degree's value,
    # check if it can still be zero
    
    def backtrack(idx, coeffs, assigned_count):
        nonlocal best_nonzero, best_result
        if time.time() > deadline:
            return
        
        if idx == 12:
            # All assigned, count nonzero
            nz = count_nonzero_square(coeffs, exponents)
            if nz < best_nonzero:
                best_nonzero = nz
                best_result = make_result(coeffs, exponents)
            return
        
        # Try each possible coefficient value
        for val in [-3, -2, -1, 1, 2, 3]:
            coeffs[idx] = val
            # Check if any fully-determined degree is now nonzero
            # A degree k is fully determined when all pairs (i,j) with i+j=k
            # have both i and j < idx (i.e., both assigned)
            valid = True
            for k in range(23):
                pairs = degree_pairs[k]
                # Check if all pairs are assigned
                all_assigned = True
                for (i, j) in pairs:
                    if j >= idx:  # j >= idx means not yet assigned (since we assign in order)
                        all_assigned = False
                        break
                if all_assigned and pairs:
                    # Compute the degree coefficient
                    val_k = 0
                    for (i, j) in pairs:
                        if i == j:
                            val_k += coeffs[i] * coeffs[j]
                        else:
                            val_k += 2 * coeffs[i] * coeffs[j]
                    # If this degree is "expected" to be zero and isn't, prune
                    # We don't know which should be zero, so just track
                    pass
            
            backtrack(idx + 1, coeffs, assigned_count + 1)
        
        coeffs[idx] = 0

    # Run backtracking with time limit
    coeffs = [0] * 12
    backtrack(0, coeffs, 0)

    # Phase 3: Local search refinement
    if best_result is not None and time.time() < deadline:
        current_coeffs = []
        current_exps = []
        for i, c in enumerate(best_result):
            if c != 0:
                current_coeffs.append(c)
                current_exps.append(i)
        
        # Simulated annealing style local search
        for _ in range(20000):
            if time.time() > deadline:
                break
            new_coeffs = current_coeffs[:]
            idx = np.random.randint(0, 12)
            delta = np.random.choice([-3, -2, -1, 1, 2, 3])
            new_val = new_coeffs[idx] + delta
            if new_val == 0 or abs(new_val) > 64:
                continue
            new_coeffs[idx] = new_val
            nz = count_nonzero_square(new_coeffs, current_exps)
            if nz < best_nonzero:
                best_nonzero = nz
                best_result = make_result(new_coeffs, current_exps)
                current_coeffs = new_coeffs[:]

    # Phase 4: Try other exponent patterns with best coefficient structure
    if best_result is not None and time.time() < deadline:
        # Extract the coefficient pattern from best result
        base_coeffs = []
        for i, c in enumerate(best_result):
            if c != 0:
                base_coeffs.append(c)
        
        if len(base_coeffs) == 12:
            # Try shifted/scaled versions
            for shift in range(0, 5):
                exponents = [i + shift for i in range(12)]
                if max(exponents) * 2 + 1 > 41:
                    continue
                nz = count_nonzero_square(base_coeffs, exponents)
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_result = make_result(base_coeffs, exponents)

    # Fallback
    if best_result is None:
        best_result = [1] * 12
    
    # Final validation
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    if sum(1 for c in best_result if c != 0) != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
