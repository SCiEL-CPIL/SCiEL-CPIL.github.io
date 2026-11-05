"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_square_count = 100
    best_poly = [0] * 41
    for i in range(12):
        best_poly[i] = 1
        
    def count_nonzero_square(poly):
        p = np.array(poly, dtype=np.int64)
        sq = np.convolve(p, p)
        return int(np.count_nonzero(sq))
    
    best_square_count = count_nonzero_square(best_poly)
    
    def evaluate(poly):
        if len(poly) != 41:
            return 1000, False
        if any(c < -64 or c > 64 for c in poly):
            return 1000, False
        nz_count = sum(1 for c in poly if c != 0)
        if nz_count != 12:
            return 1000, False
        sq_nz = count_nonzero_square(poly)
        return sq_nz, True

    # Generate structured exponent sets that maximize sum collisions
    # Key: for a 12-term poly, square has positions 0..80 (if exponents in 0..40)
    # We want many pairs (i,j) with e_i + e_j = same value
    
    def generate_exponent_sets():
        """Generate exponent sets with high sum-collision density."""
        sets = []
        
        # 1. Pure APs (step 1, 2, 3)
        for step in [1, 2, 3]:
            max_start = 40 - 11 * step
            for start in range(0, max_start + 1):
                sets.append([start + i * step for i in range(12)])
        
        # 2. Two interleaved APs (6 terms each, offset)
        for step_a in [1, 2, 3]:
            for step_b in [1, 2, 3]:
                for offset in range(0, 5):
                    try:
                        ap1 = [offset + i * step_a for i in range(6)]
                        ap2 = [offset + step_a * 3 + i * step_b for i in range(6)]
                        if all(0 <= e <= 40 for e in ap1 + ap2):
                            sets.append(sorted(ap1 + ap2))
                    except:
                        pass
        
        # 3. Symmetric sets: e and (max_e - e) pairs
        for max_e in range(12, 41):
            for k in range(12):
                try:
                    # Pick 6 values from 0..max_e//2, mirror them
                    half = max_e // 2
                    candidates = list(range(0, half + 1))
                    if len(candidates) >= 6:
                        for combo in random.sample(range(len(candidates)), 6):
                            vals = sorted([candidates[combo[i]] for i in range(6)])
                            mirrored = [max_e - v for v in vals]
                            full_set = sorted(vals + mirrored)
                            if len(set(full_set)) == 12 and all(0 <= e <= 40 for e in full_set):
                                sets.append(full_set)
                except:
                    pass
        
        # 4. Random structured sets
        for _ in range(200):
            base = random.randint(0, 30)
            step = random.randint(1, 4)
            exponents = []
            for i in range(12):
                e = base + i * step + random.choice([0, 0, 0, 1, -1])
                if 0 <= e <= 40:
                    exponents.append(e)
            if len(exponents) == 12:
                exponents = sorted(exponents)
                if len(set(exponents)) == 12:
                    sets.append(exponents)
        
        return sets
    
    def systematic_coeff_search(exponents, coeff_values):
        """
        Systematically try coefficient assignments from coeff_values.
        Uses pruning: evaluate partial assignments early.
        """
        n = len(exponents)  # 12
        best_local = (1000, None)
        
        # Precompute: for each pair (i,j), the sum position e_i + e_j
        sum_positions = {}
        for i in range(n):
            for j in range(i, n):
                s = exponents[i] + exponents[j]
                if s not in sum_positions:
                    sum_positions[s] = []
                sum_positions[s].append((i, j))
        
        # Count how many pairs hit each sum position (collision count)
        collision_counts = {s: len(pairs) for s, pairs in sum_positions.items()}
        
        # Sort exponents by "collision importance" - those involved in more collisions first
        # Actually, let's just do systematic search with pruning
        
        def partial_eval(partial_coeffs):
            """
            Evaluate the square with partial coefficients (rest assumed 0).
            Returns count of nonzero terms in the partial square.
            This is a lower bound on the full square's nonzero count.
            """
            poly = [0] * 41
            for i, c in enumerate(partial_coeffs):
                if c is not None:
                    poly[exponents[i]] = c
            p = np.array(poly, dtype=np.int64)
            sq = np.convolve(p, p)
            return int(np.count_nonzero(sq))
        
        # Recursive search with pruning
        def search(idx, current_coeffs, current_partial_count):
            nonlocal best_local
            
            if time.time() > deadline:
                return
            
            if idx == n:
                # Full assignment
                poly = [0] * 41
                for i, c in enumerate(current_coeffs):
                    poly[exponents[i]] = c
                score, valid = evaluate(poly)
                if valid and score < best_local[0]:
                    best_local = (score, poly[:])
                return
            
            # Pruning: if current partial count already exceeds best, prune
            if current_partial_count > best_local[0]:
                return
            
            for c in coeff_values:
                new_coeffs = current_coeffs + [c]
                new_partial = partial_eval(new_coeffs)
                search(idx + 1, new_coeffs, new_partial)
        
        search(0, [], 0)
        return best_local
    
    # Try small coefficient sets systematically
    # Start with the most structured exponent sets
    exponent_sets = generate_exponent_sets()
    
    # Sort by collision density (more collisions = more potential cancellation)
    def collision_score(exps):
        sums = [exps[i] + exps[j] for i in range(12) for j in range(i+1, 12)]
        from collections import Counter
        cnt = Counter(sums)
        # Sum of (count-1) for each sum = total excess collisions
        return sum(max(0, c-1) for c in cnt.values())
    
    exponent_sets.sort(key=collision_score, reverse=True)
    
    # Coefficient value sets to try (small integers for exact cancellation)
    coeff_sets = [
        [1, -1],           # Binary: max cancellation potential
        [1, -1, 2, -2],    # Slightly richer
        [1, -1, 2, -2, 3, -3],  # Even richer
    ]
    
    for exps in exponent_sets[:50]:  # Limit to top 50 most collision-dense
        if time.time() > deadline:
            break
        
        # For each coefficient set, do systematic search
        for cvals in coeff_sets:
            if time.time() > deadline:
                break
            
            # For binary {1,-1}, we can enumerate all 2^12 = 4096 assignments
            if len(cvals) == 2:
                # Fast path: enumerate all sign patterns
                for mask in range(1 << 12):
                    if time.time() > deadline:
                        break
                    coeffs = []
                    for i in range(12):
                        if (mask >> i) & 1:
                            coeffs.append(cvals[1])  # -1
                        else:
                            coeffs.append(cvals[0])  # 1
                    poly = [0] * 41
                    for i, c in enumerate(coeffs):
                        poly[exps[i]] = c
                    score, valid = evaluate(poly)
                    if valid and score < best_square_count:
                        best_square_count = score
                        best_poly = poly
            else:
                # Slower systematic search for larger coefficient sets
                # Use a limited random sampling approach
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    coeffs = [random.choice(cvals) for _ in range(12)]
                    poly = [0] * 41
                    for i, c in enumerate(coeffs):
                        poly[exps[i]] = c
                    score, valid = evaluate(poly)
                    if valid and score < best_square_count:
                        best_square_count = score
                        best_poly = poly
    
    # Final local refinement around best found
    if best_poly is not None and best_square_count < 100:
        # Extract current exponents and do a quick local search
        current_exps = [i for i in range(41) if best_poly[i] != 0]
        current_coeffs = [best_poly[i] for i in current_exps]
        
        # Try flipping individual signs
        for _ in range(500):
            if time.time() > deadline:
                break
            new_poly = best_poly[:]
            idx = random.choice(current_exps)
            new_poly[idx] = -new_poly[idx]
            score, valid = evaluate(new_poly)
            if valid and score < best_square_count:
                best_square_count = score
                best_poly = new_poly
                current_exps = [i for i in range(41) if best_poly[i] != 0]
                current_coeffs = [best_poly[i] for i in current_exps]
    
    # Final validation
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12 or any(c < -64 or c > 64 for c in best_poly):
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
        best_square_count = count_nonzero_square(best_poly)
    
    # Trim trailing zeros
    result = best_poly[:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
