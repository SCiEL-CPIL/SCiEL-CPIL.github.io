import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin before 90s kill
    
    best_result = [1] * 12  # fallback
    best_nonzero = 144  # worst case
    
    def count_nonzero_square(coeffs, exponents):
        """Count nonzero terms in the square of sum(c_i * x^e_i) using numpy"""
        n = len(coeffs)
        max_deg = max(exponents)
        poly = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for c, e in zip(coeffs, exponents):
            poly[e] = c
        square = np.convolve(poly, poly)
        return int(np.count_nonzero(square))
    
    def count_nonzero_square_fast(coeffs, exponents):
        """Faster count using direct computation for small polynomials"""
        n = len(coeffs)
        max_deg = max(exponents)
        size = 2 * max_deg + 1
        square = np.zeros(size, dtype=np.int64)
        for i in range(n):
            for j in range(i, n):
                d = exponents[i] + exponents[j]
                val = coeffs[i] * coeffs[j]
                if i == j:
                    square[d] += val
                else:
                    square[d] += 2 * val
        return int(np.count_nonzero(square))
    
    def is_valid_result(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if any(e < 0 for e in exponents):
            return False
        max_deg = max(exponents)
        if 2 * max_deg + 1 > 41:
            return False
        return True
    
    def coeffs_from_state(state):
        """Convert state to coefficient list in ascending degree, length <= 41"""
        coeffs = [0] * (2 * max(state[1]) + 1)
        for c, e in zip(state[0], state[1]):
            coeffs[e] = c
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    def update_best(coeffs, exponents, nz):
        nonlocal best_nonzero, best_result
        if nz < best_nonzero:
            best_nonzero = nz
            result = coeffs_from_state((coeffs, exponents))
            if len(result) <= 41 and result[-1] != 0:
                best_result = result
    
    # Expanded exponent patterns
    exponent_patterns = [
        list(range(12)),
        list(range(0, 24, 2)),
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
        [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17],
        [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20],
        [0, 2, 4, 6, 14, 16, 18, 20, 22, 24, 26, 28],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],
        [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25],
        [0, 1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18],
        [0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
    ]
    
    # Coefficient sets
    coeff_sets = [
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
        [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [3, -2, 1, -3, 3, -2, 1, -3, 3, -2, 1, -3],
        [1, 1, -2, 2, -1, -1, 2, -2, 1, 1, -2, 2],
        [2, 1, -1, -2, -1, 2, 1, -1, -2, -1, 2, 1],
        [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12],
        [1, 2, 3, 4, -5, -6, -7, -8, 9, 10, 11, 12],
        [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6],
        [1, 3, -2, 4, -3, 5, -4, 6, -5, 7, -6, 8],
        [1, -3, 2, -4, 3, -5, 4, -6, 5, -7, 6, -8],
        [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2],
        [1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12],
    ]
    
    # First pass: quick evaluation of structured patterns
    for exponents in exponent_patterns:
        if time.time() > deadline:
            break
        for coeffs in coeff_sets:
            if not is_valid_result(coeffs, exponents):
                continue
            nz = count_nonzero_square(coeffs, exponents)
            update_best(coeffs, exponents, nz)
    
    # GRAPH-BASED STRATEGY: Build collision graph and use branch-and-bound
    # over sign assignments to enforce cancellations
    if time.time() < deadline:
        for exponents in exponent_patterns:
            if time.time() > deadline:
                break
            
            # Build collision structure: for each degree d in P^2, which pairs (i,j) contribute
            n = 12
            max_deg = max(exponents)
            # Map degree -> list of (i, j) pairs with i <= j
            degree_pairs = defaultdict(list)
            for i in range(n):
                for j in range(i, n):
                    d = exponents[i] + exponents[j]
                    degree_pairs[d].append((i, j))
            
            # For each degree, we want sum_{(i,j)} weight(i,j) * c_i * c_j = 0
            # where weight is 1 if i==j, 2 otherwise
            # This is a system of quadratic equations in the coefficients
            
            # Strategy: Fix magnitudes, search over signs
            # For a given set of magnitudes m_i, the equation becomes:
            # sum_{(i,j)} w(i,j) * m_i * m_j * s_i * s_j = 0 for each degree
            # where s_i in {+1, -1}
            
            # Try a few magnitude patterns
            mag_patterns = [
                [1]*12,
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,2,2,2,2,2,2,2],
                [1,1,1,2,2,2,2,2,2,2,2,2],
                [1,1,2,2,2,2,2,2,2,2,2,2],
                [1,2,2,2,2,2,2,2,2,2,2,2],
                [1,1,1,1,2,2,2,2,2,2,2,2],
                [1,1,2,1,2,1,2,1,2,1,2,1],
                [1,2,1,2,1,2,1,2,1,2,1,2],
            ]
            
            for mags in mag_patterns:
                if time.time() > deadline:
                    break
                
                # Pre-compute the constraint matrix for signs
                # For each degree d with pairs, we have:
                # sum over pairs (i,j) of w(i,j)*m_i*m_j*s_i*s_j = 0
                
                # Collect all constraints
                constraints = []  # list of lists of (i, j, weight)
                for d in sorted(degree_pairs.keys()):
                    pairs = degree_pairs[d]
                    if len(pairs) <= 1:
                        # Single pair: w*m_i*m_j*s_i*s_j must be 0, impossible if m>0
                        # Actually if it's a diagonal term (i==j), it's m_i^2 which is always nonzero
                        # So we can't have a degree with only one contributing pair unless it's diagonal
                        # For diagonal: c_i^2 is always positive, so that degree is always nonzero
                        # For off-diagonal single pair: 2*c_i*c_j, can be zero only if one is zero
                        # Since all coeffs nonzero, this degree is always nonzero
                        continue
                    constraint = []
                    for (i, j) in pairs:
                        w = 1 if i == j else 2
                        weight = w * mags[i] * mags[j]
                        constraint.append((i, j, weight))
                    constraints.append(constraint)
                
                if not constraints:
                    continue
                
                # Branch and bound over sign assignments
                # We'll use a greedy + local search approach
                # Start with random sign assignment, then flip signs to reduce nonzero count
                
                rng = random.Random(hash(tuple(mags)) % 10000)
                for trial in range(20):
                    if time.time() > deadline:
                        break
                    
                    # Random initial sign assignment
                    signs = [1 if rng.random() < 0.5 else -1 for _ in range(12)]
                    coeffs = [mags[i] * signs[i] for i in range(12)]
                    
                    # Local search: flip individual signs
                    improved = True
                    max_iters = 50
                    for it in range(max_iters):
                        if time.time() > deadline:
                            break
                        if not improved:
                            break
                        improved = False
                        
                        best_flip = None
                        best_nz = count_nonzero_square(coeffs, exponents)
                        
                        for flip_idx in range(12):
                            new_coeffs = coeffs[:]
                            new_coeffs[flip_idx] = -new_coeffs[flip_idx]
                            if new_coeffs[flip_idx] == 0:
                                continue
                            nz = count_nonzero_square(new_coeffs, exponents)
                            if nz < best_nz:
                                best_nz = nz
                                best_flip = flip_idx
                        
                        if best_flip is not None:
                            coeffs[best_flip] = -coeffs[best_flip]
                            improved = True
                    
                    if is_valid_result(coeffs, exponents):
                        nz = count_nonzero_square(coeffs, exponents)
                        update_best(coeffs, exponents, nz)
    
    # Strategy: Arithmetic progression with optimized coefficients
    if time.time() < deadline:
        for step in [1, 2, 3]:
            exponents = [i * step for i in range(12)]
            max_deg = 2 * exponents[-1]
            if max_deg + 1 > 41:
                continue
            
            # For AP exponents, degree d in P^2 corresponds to i+j = d/step
            # So we need: for each s, sum_{i+j=s} w(i,j)*c_i*c_j = 0
            # This is a system of linear equations in the products c_i*c_j
            
            # Try systematic coefficient patterns
            for pattern_type in range(8):
                for base in range(1, 5):
                    coeffs = []
                    for i in range(12):
                        if pattern_type == 0:
                            c = base * ((-1) ** i)
                        elif pattern_type == 1:
                            c = base * ((-1) ** (i % 2))
                        elif pattern_type == 2:
                            c = base * ((-1) ** (i // 2))
                        elif pattern_type == 3:
                            c = base * (1 + (i % 3)) * ((-1) ** i)
                        elif pattern_type == 4:
                            c = base * ((i % 4) - 1) * ((-1) ** i)
                        elif pattern_type == 5:
                            c = base * ((i % 3) - 1) * ((-1) ** (i // 3))
                        elif pattern_type == 6:
                            c = base * ((-1) ** (i % 3))
                        else:
                            c = base * ((-1) ** (i % 4))
                        coeffs.append(c)
                    
                    if any(c == 0 for c in coeffs):
                        continue
                    if not is_valid_result(coeffs, exponents):
                        continue
                    nz = count_nonzero_square(coeffs, exponents)
                    update_best(coeffs, exponents, nz)
    
    # Strategy: Randomized local search with graph-based initialization
    if time.time() < deadline:
        rng = random.Random(42)
        for exponents in exponent_patterns[:10]:
            if time.time() > deadline:
                break
            max_deg = max(exponents)
            
            # Build collision graph for this exponent set
            degree_pairs = defaultdict(list)
            for i in range(12):
                for j in range(i, 12):
                    d = exponents[i] + exponents[j]
                    degree_pairs[d].append((i, j))
            
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                # Generate random coefficients
                coeffs = []
                for i in range(12):
                    mag = rng.choice([1, 1, 1, 2, 2, 3, 4, 5])
                    sign = 1 if rng.random() < 0.5 else -1
                    coeffs.append(sign * mag)
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, exponents):
                    continue
                
                # Local search with sign flips
                current_nz = count_nonzero_square(coeffs, exponents)
                improved = True
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    if not improved:
                        break
                    improved = False
                    
                    best_flip = None
                    best_nz = current_nz
                    for flip_idx in range(12):
                        new_coeffs = coeffs[:]
                        new_coeffs[flip_idx] = -new_coeffs[flip_idx]
                        if new_coeffs[flip_idx] == 0:
                            continue
                        nz = count_nonzero_square(new_coeffs, exponents)
                        if nz < best_nz:
                            best_nz = nz
                            best_flip = flip_idx
                    
                    if best_flip is not None:
                        coeffs[best_flip] = -coeffs[best_flip]
                        current_nz = best_nz
                        improved = True
                
                if is_valid_result(coeffs, exponents):
                    nz = count_nonzero_square(coeffs, exponents)
                    update_best(coeffs, exponents, nz)
    
    # Strategy: P = A(x) + x^k * A(x) where A has 6 terms - EXPANDED
    if time.time() < deadline:
        rng_ss = random.Random(999)
        for k in range(1, 12):
            if k > 19:  # need 2*max_deg+1 <= 41, max_deg = k+5, so k+5 <= 20
                break
            max_deg = k + 5
            if 2 * max_deg + 1 > 41:
                continue
            
            # Systematic search over small coefficient sets for A
            mag_options = [1, 2, 3, 4, 5]
            for _ in range(200):
                if time.time() > deadline:
                    break
                A = []
                for i in range(6):
                    mag = rng_ss.choice(mag_options)
                    sign = 1 if rng_ss.random() < 0.5 else -1
                    A.append(sign * mag)
                
                if k <= 5:
                    full_coeffs = [0] * (k + 6)
                    for i in range(6):
                        full_coeffs[i] += A[i]
                        full_coeffs[k + i] += A[i]
                    terms = [(i, c) for i, c in enumerate(full_coeffs) if c != 0]
                    if len(terms) != 12:
                        continue
                    coeffs = [c for _, c in terms]
                    exponents = [e for e, _ in terms]
                else:
                    coeffs = A + A
                    exponents = list(range(6)) + list(range(k, k + 6))
                
                if any(abs(c) > 64 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, exponents):
                    continue
                nz = count_nonzero_square_fast(coeffs, exponents)
                update_best(coeffs, exponents, nz)
                
                # Local search on this candidate
                if nz < best_nonzero:
                    improved = True
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        if not improved:
                            break
                        improved = False
                        cur_nz = nz
                        for flip_idx in range(len(coeffs)):
                            new_coeffs = coeffs[:]
                            new_coeffs[flip_idx] = -new_coeffs[flip_idx]
                            if new_coeffs[flip_idx] == 0:
                                continue
                            if not is_valid_result(new_coeffs, exponents):
                                continue
                            new_nz = count_nonzero_square_fast(new_coeffs, exponents)
                            if new_nz < cur_nz:
                                cur_nz = new_nz
                                coeffs = new_coeffs
                                improved = True
                        if improved:
                            nz = cur_nz
                            if is_valid_result(coeffs, exponents):
                                update_best(coeffs, exponents, nz)
    
    # Strategy: Local search refinement on best found - ENHANCED
    if time.time() < deadline:
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_exponents = nz_indices
            best_coeffs = [best_result[i] for i in nz_indices]
            
            for _ in range(300):
                if time.time() > deadline:
                    break
                coeffs = best_coeffs[:]
                num_perturb = random.randint(1, 3)
                for _ in range(num_perturb):
                    idx = random.randint(0, 11)
                    old_c = coeffs[idx]
                    new_c = old_c + random.choice([-3, -2, -1, 1, 2, 3])
                    if new_c != 0 and -64 <= new_c <= 64:
                        coeffs[idx] = new_c
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, best_exponents):
                    continue
                nz = count_nonzero_square_fast(coeffs, best_exponents)
                update_best(coeffs, best_exponents, nz)
                if nz <= best_nonzero:
                    best_coeffs = coeffs[:]
                
                # Also try sign flips on the current best
                for flip_idx in range(12):
                    if time.time() > deadline:
                        break
                    new_coeffs = best_coeffs[:]
                    new_coeffs[flip_idx] = -new_coeffs[flip_idx]
                    if new_coeffs[flip_idx] == 0:
                        continue
                    if not is_valid_result(new_coeffs, best_exponents):
                        continue
                    nz = count_nonzero_square_fast(new_coeffs, best_exponents)
                    if nz < best_nonzero:
                        best_coeffs = new_coeffs
                        update_best(new_coeffs, best_exponents, nz)
    
    # Strategy: More random exponent patterns with local search
    if time.time() < deadline:
        rng = random.Random(123)
        for _ in range(80):
            if time.time() > deadline:
                break
            exponents = sorted(rng.sample(range(0, 21), 12))
            if max(exponents) * 2 + 1 > 41:
                continue
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                coeffs = []
                for i in range(12):
                    mag = rng.choice([1, 1, 2, 2, 3, 4])
                    sign = 1 if rng.random() < 0.5 else -1
                    coeffs.append(sign * mag)
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, exponents):
                    continue
                nz = count_nonzero_square_fast(coeffs, exponents)
                update_best(coeffs, exponents, nz)
                
                # Quick local search
                if nz < best_nonzero + 5:
                    improved = True
                    for _ in range(15):
                        if time.time() > deadline:
                            break
                        if not improved:
                            break
                        improved = False
                        cur_nz = nz
                        for flip_idx in range(12):
                            new_coeffs = coeffs[:]
                            new_coeffs[flip_idx] = -new_coeffs[flip_idx]
                            if new_coeffs[flip_idx] == 0:
                                continue
                            if not is_valid_result(new_coeffs, exponents):
                                continue
                            new_nz = count_nonzero_square_fast(new_coeffs, exponents)
                            if new_nz < cur_nz:
                                cur_nz = new_nz
                                coeffs = new_coeffs
                                improved = True
                        if improved:
                            nz = cur_nz
                            if is_valid_result(coeffs, exponents):
                                update_best(coeffs, exponents, nz)
    
    # Strategy: Targeted search for P = A + x^k*A with specific structures
    if time.time() < deadline:
        rng_t = random.Random(777)
        for k in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]:
            if time.time() > deadline:
                break
            max_deg = k + 5
            if 2 * max_deg + 1 > 41:
                continue
            
            # Try patterns where A has symmetric structure
            for pattern in range(30):
                if time.time() > deadline:
                    break
                A = []
                for i in range(6):
                    if pattern % 2 == 0:
                        mag = (i % 3) + 1
                    else:
                        mag = rng_t.choice([1, 2, 3, 4])
                    sign = 1 if (i + pattern) % 2 == 0 else -1
                    A.append(sign * mag)
                
                if k > 5:
                    coeffs = A + A
                    exponents = list(range(6)) + list(range(k, k + 6))
                else:
                    full_coeffs = [0] * (k + 6)
                    for i in range(6):
                        full_coeffs[i] += A[i]
                        full_coeffs[k + i] += A[i]
                    terms = [(i, c) for i, c in enumerate(full_coeffs) if c != 0]
                    if len(terms) != 12:
                        continue
                    coeffs = [c for _, c in terms]
                    exponents = [e for e, _ in terms]
                
                if any(abs(c) > 64 for c in coeffs) or any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, exponents):
                    continue
                nz = count_nonzero_square_fast(coeffs, exponents)
                update_best(coeffs, exponents, nz)
                
                # Local search
                if nz < best_nonzero:
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        improved = False
                        cur_nz = nz
                        for flip_idx in range(len(coeffs)):
                            new_coeffs = coeffs[:]
                            new_coeffs[flip_idx] = -new_coeffs[flip_idx]
                            if new_coeffs[flip_idx] == 0:
                                continue
                            if not is_valid_result(new_coeffs, exponents):
                                continue
                            new_nz = count_nonzero_square_fast(new_coeffs, exponents)
                            if new_nz < cur_nz:
                                cur_nz = new_nz
                                coeffs = new_coeffs
                                improved = True
                        if improved:
                            nz = cur_nz
                            if is_valid_result(coeffs, exponents):
                                update_best(coeffs, exponents, nz)
                        else:
                            break
    
    # Ensure result is valid
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
