import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
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
    
    def fast_square_nonzero(coeffs, exponents, degree_map):
        """Fast count using precomputed degree_map: dict degree -> list of (i,j,weight)"""
        sq = {}
        for d, pairs in degree_map.items():
            val = 0
            for (i, j, w) in pairs:
                val += w * coeffs[i] * coeffs[j]
            sq[d] = val
        return sum(1 for v in sq.values() if v != 0)
    
    def fast_square_values(coeffs, exponents, degree_map):
        """Compute all square values using precomputed degree_map"""
        sq = {}
        for d, pairs in degree_map.items():
            val = 0
            for (i, j, w) in pairs:
                val += w * coeffs[i] * coeffs[j]
            sq[d] = val
        return sq
    
    def build_degree_map(exponents):
        """Precompute degree -> list of (i, j, weight) for square terms"""
        n = len(exponents)
        dm = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                d = exponents[i] + exponents[j]
                w = 1 if i == j else 2
                dm[d].append((i, j, w))
        return dict(dm)
    
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
    
    # First pass: quick evaluation of structured patterns with fast square
    for exponents in exponent_patterns:
        if time.time() > deadline:
            break
        dm = build_degree_map(exponents)
        for coeffs in coeff_sets:
            if not is_valid_result(coeffs, exponents):
                continue
            nz = fast_square_nonzero(coeffs, exponents, dm)
            update_best(coeffs, exponents, nz)
    
    # SPECTRAL-PRUNED SEARCH: Use autocorrelation to find exponent supports
    # with high collision density, then targeted sign search
    if time.time() < deadline:
        # For each exponent pattern, compute the "collision density" - how many
        # degrees have multiple contributing pairs (more opportunities for cancellation)
        collision_scores = []
        for exponents in exponent_patterns:
            dm = build_degree_map(exponents)
            # Count degrees with >= 2 pairs (cancellable)
            cancellable = sum(1 for d, pairs in dm.items() if len(pairs) >= 2)
            total_degrees = len(dm)
            # Also count total pairs involved in cancellable degrees
            total_pairs_in_cancellable = sum(len(pairs) for d, pairs in dm.items() if len(pairs) >= 2)
            score = cancellable * 10 + total_pairs_in_cancellable
            collision_scores.append((score, exponents, dm, cancellable, total_degrees))
        
        # Sort by collision density (highest first)
        collision_scores.sort(key=lambda x: -x[0])
        
        # Focus search on top patterns with high collision density
        for score, exponents, dm, cancellable, total_degrees in collision_scores[:10]:
            if time.time() > deadline:
                break
            
            # For high-collision patterns, do intensive sign search
            # Key insight: if many pairs collide at same degree, we can satisfy
            # more cancellation constraints with sign flips
            
            # Build the constraint structure
            # For each cancellable degree, we have: sum w_ij * m_i * m_j * s_i * s_j = 0
            # With all m_i = 1: sum w_ij * s_i * s_j = 0
            
            # Precompute for each degree the list of (i, j, weight)
            degree_constraints = {}
            for d, pairs in dm.items():
                if len(pairs) >= 2:
                    degree_constraints[d] = [(i, j, w) for (i, j, w) in pairs]
            
            # Intensive local search with multiple restarts
            rng = random.Random(hash(tuple(exponents)) % 100000)
            num_restarts = 50 if cancellable > 5 else 30
            max_iters = 120
            
            for restart in range(num_restarts):
                if time.time() > deadline:
                    break
                
                # Random initial assignment
                coeffs = [1 if rng.random() < 0.5 else -1 for _ in range(12)]
                current_nz = fast_square_nonzero(coeffs, exponents, dm)
                
                # Multi-phase local search
                improved = True
                for it in range(max_iters):
                    if time.time() > deadline:
                        break
                    if not improved:
                        break
                    improved = False
                    
                    # Try all single flips
                    best_flip = None
                    best_nz = current_nz
                    for flip_idx in range(12):
                        new_coeff = -coeffs[flip_idx]
                        nz = 0
                        for d, pairs in dm.items():
                            val = 0
                            for (i, j, w) in pairs:
                                ci = new_coeff if i == flip_idx else coeffs[i]
                                cj = new_coeff if j == flip_idx else coeffs[j]
                                val += w * ci * cj
                            if val != 0:
                                nz += 1
                        if nz < best_nz:
                            best_nz = nz
                            best_flip = flip_idx
                    
                    if best_flip is not None:
                        coeffs[best_flip] = -coeffs[best_flip]
                        current_nz = best_nz
                        improved = True
                    else:
                        # Try double flips (2-opt)
                        best_pair = None
                        for i in range(12):
                            for j in range(i+1, 12):
                                old_i, old_j = coeffs[i], coeffs[j]
                                coeffs[i], coeffs[j] = -old_i, -old_j
                                nz = fast_square_nonzero(coeffs, exponents, dm)
                                coeffs[i], coeffs[j] = old_i, old_j
                                if nz < best_nz:
                                    best_nz = nz
                                    best_pair = (i, j)
                        
                        if best_pair is not None:
                            i, j = best_pair
                            coeffs[i] = -coeffs[i]
                            coeffs[j] = -coeffs[j]
                            current_nz = best_nz
                            improved = True
                
                if is_valid_result(coeffs, exponents):
                    nz = fast_square_nonzero(coeffs, exponents, dm)
                    update_best(coeffs, exponents, nz)
    
    # GRAPH-BASED STRATEGY: Build collision graph and use branch-and-bound
    # over sign assignments to enforce cancellations
    if time.time() < deadline:
        for exponents in exponent_patterns:
            if time.time() > deadline:
                break
            
            # Build collision structure with weights
            n = 12
            dm = build_degree_map(exponents)
            
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
                [2,1,1,2,1,1,2,1,1,2,1,1],
                [1,2,2,1,2,2,1,2,2,1,2,2],
            ]
            
            for mags in mag_patterns:
                if time.time() > deadline:
                    break
                
                # Identify "forced" degrees: those with only one pair (always nonzero)
                forced_nonzero = set()
                for d, pairs in dm.items():
                    if len(pairs) == 1:
                        forced_nonzero.add(d)
                
                # Count how many degrees have multiple pairs (can potentially cancel)
                cancellable_degrees = [d for d in dm if d not in forced_nonzero]
                
                # Branch and bound over sign assignments with fast evaluation
                rng = random.Random(hash(tuple(mags)) % 10000)
                num_trials = 30 if len(cancellable_degrees) > 5 else 15
                
                for trial in range(num_trials):
                    if time.time() > deadline:
                        break
                    
                    # Random initial sign assignment
                    signs = [1 if rng.random() < 0.5 else -1 for _ in range(12)]
                    coeffs = [mags[i] * signs[i] for i in range(12)]
                    
                    # Fast local search using incremental updates
                    current_nz = fast_square_nonzero(coeffs, exponents, dm)
                    improved = True
                    max_iters = 80
                    for it in range(max_iters):
                        if time.time() > deadline:
                            break
                        if not improved:
                            break
                        improved = False
                        
                        best_flip = None
                        best_nz = current_nz
                        
                        for flip_idx in range(12):
                            new_coeff = -coeffs[flip_idx]
                            nz = 0
                            for d, pairs in dm.items():
                                val = 0
                                for (i, j, w) in pairs:
                                    ci = new_coeff if i == flip_idx else coeffs[i]
                                    cj = new_coeff if j == flip_idx else coeffs[j]
                                    val += w * ci * cj
                                if val != 0:
                                    nz += 1
                            if nz < best_nz:
                                best_nz = nz
                                best_flip = flip_idx
                        
                        if best_flip is not None:
                            coeffs[best_flip] = -coeffs[best_flip]
                            current_nz = best_nz
                            improved = True
                    
                    if is_valid_result(coeffs, exponents):
                        nz = fast_square_nonzero(coeffs, exponents, dm)
                        update_best(coeffs, exponents, nz)
    
    # Strategy: Arithmetic progression with optimized coefficients
    if time.time() < deadline:
        for step in [1, 2, 3]:
            exponents = [i * step for i in range(12)]
            max_deg = 2 * exponents[-1]
            if max_deg + 1 > 41:
                continue
            
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
        for exponents in exponent_patterns[:12]:
            if time.time() > deadline:
                break
            dm = build_degree_map(exponents)
            
            for _ in range(120):
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
                
                # Local search with sign flips using fast evaluation
                current_nz = fast_square_nonzero(coeffs, exponents, dm)
                improved = True
                for _ in range(40):
                    if time.time() > deadline:
                        break
                    if not improved:
                        break
                    improved = False
                    
                    best_flip = None
                    best_nz = current_nz
                    for flip_idx in range(12):
                        new_coeff = -coeffs[flip_idx]
                        nz = 0
                        for d, pairs in dm.items():
                            val = 0
                            for (i, j, w) in pairs:
                                ci = new_coeff if i == flip_idx else coeffs[i]
                                cj = new_coeff if j == flip_idx else coeffs[j]
                                val += w * ci * cj
                            if val != 0:
                                nz += 1
                        if nz < best_nz:
                            best_nz = nz
                            best_flip = flip_idx
                    
                    if best_flip is not None:
                        coeffs[best_flip] = -coeffs[best_flip]
                        current_nz = best_nz
                        improved = True
                
                if is_valid_result(coeffs, exponents):
                    nz = fast_square_nonzero(coeffs, exponents, dm)
                    update_best(coeffs, exponents, nz)
    
    # Strategy: P = A(x) + x^k * A(x) where A has 6 terms
    if time.time() < deadline:
        for k in range(1, 12):
            for _ in range(80):
                if time.time() > deadline:
                    break
                A = []
                for i in range(6):
                    mag = random.choice([1, 1, 1, 2, 2, 3])
                    sign = 1 if random.random() < 0.5 else -1
                    A.append(sign * mag)
                
                if k <= 5:
                    full_coeffs = [0] * (max(6, k + 6))
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
                dm = build_degree_map(exponents)
                nz = fast_square_nonzero(coeffs, exponents, dm)
                update_best(coeffs, exponents, nz)
    
    # Strategy: Local search refinement on best found
    if time.time() < deadline:
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_exponents = nz_indices
            best_coeffs = [best_result[i] for i in nz_indices]
            dm = build_degree_map(best_exponents)
            
            # First, do sign-flip local search on the best found
            current_coeffs = best_coeffs[:]
            current_nz = fast_square_nonzero(current_coeffs, best_exponents, dm)
            improved = True
            for _ in range(100):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                best_flip = None
                best_nz = current_nz
                for flip_idx in range(12):
                    new_coeff = -current_coeffs[flip_idx]
                    nz = 0
                    for d, pairs in dm.items():
                        val = 0
                        for (i, j, w) in pairs:
                            ci = new_coeff if i == flip_idx else current_coeffs[i]
                            cj = new_coeff if j == flip_idx else current_coeffs[j]
                            val += w * ci * cj
                        if val != 0:
                            nz += 1
                    if nz < best_nz:
                        best_nz = nz
                        best_flip = flip_idx
                if best_flip is not None:
                    current_coeffs[best_flip] = -current_coeffs[best_flip]
                    current_nz = best_nz
                    improved = True
            
            if is_valid_result(current_coeffs, best_exponents):
                update_best(current_coeffs, best_exponents, current_nz)
            
            # Then, magnitude perturbation
            for _ in range(400):
                if time.time() > deadline:
                    break
                coeffs = best_coeffs[:]
                num_perturb = random.randint(1, 3)
                for _ in range(num_perturb):
                    idx = random.randint(0, 11)
                    old_c = coeffs[idx]
                    new_c = old_c + random.choice([-2, -1, 1, 2])
                    if new_c != 0 and -64 <= new_c <= 64:
                        coeffs[idx] = new_c
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, best_exponents):
                    continue
                nz = fast_square_nonzero(coeffs, best_exponents, dm)
                update_best(coeffs, best_exponents, nz)
                if nz == best_nonzero:
                    best_coeffs = coeffs[:]
    
    # Strategy: More random exponent patterns with fast evaluation
    if time.time() < deadline:
        rng = random.Random(123)
        for _ in range(150):
            if time.time() > deadline:
                break
            exponents = sorted(rng.sample(range(0, 21), 12))
            if max(exponents) * 2 + 1 > 41:
                continue
            dm = build_degree_map(exponents)
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                coeffs = []
                for i in range(12):
                    mag = rng.choice([1, 1, 1, 2, 2, 3])
                    sign = 1 if rng.random() < 0.5 else -1
                    coeffs.append(sign * mag)
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, exponents):
                    continue
                
                # Quick local search
                current_nz = fast_square_nonzero(coeffs, exponents, dm)
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    best_flip = None
                    best_nz = current_nz
                    for flip_idx in range(12):
                        new_coeff = -coeffs[flip_idx]
                        nz = 0
                        for d, pairs in dm.items():
                            val = 0
                            for (i, j, w) in pairs:
                                ci = new_coeff if i == flip_idx else coeffs[i]
                                cj = new_coeff if j == flip_idx else coeffs[j]
                                val += w * ci * cj
                            if val != 0:
                                nz += 1
                        if nz < best_nz:
                            best_nz = nz
                            best_flip = flip_idx
                    if best_flip is not None:
                        coeffs[best_flip] = -coeffs[best_flip]
                        current_nz = best_nz
                    else:
                        break
                
                if is_valid_result(coeffs, exponents):
                    update_best(coeffs, exponents, current_nz)
    
    # Ensure result is valid
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
