import random
import itertools
import numpy as np

def construct():
    best_count = 23  # baseline for degree 11 contiguous
    best_coeffs = [1]*12  # placeholder
    
    def count_nonzero_from_poly(poly_list):
        # poly_list is a list of coefficients, length <= 41
        # Compute square coefficients
        n = len(poly_list)
        if n == 0:
            return 0, []
        # Pad to ensure square fits
        max_deg = n - 1
        square_len = 2 * max_deg + 1
        square = [0] * square_len
        # Optimization: only iterate non-zero terms
        nz_terms = [(i, c) for i, c in enumerate(poly_list) if c != 0]
        for i, ci in nz_terms:
            for j, cj in nz_terms:
                square[i+j] += ci * cj
        return sum(1 for x in square if x != 0), square

    random.seed(42)
    
    # Search over sparse polynomials of length up to 41
    # We represent a candidate as a list of length 41, with exactly 12 non-zeros.
    # To speed up, we can work with (indices, coeffs) pairs.
    
    max_degree = 40
    list_len = max_degree + 1
    
    # Initialize with a few structured guesses
    candidates = []
    
    # 1. Random sparse polynomials
    for _ in range(2000):
        # Choose 12 distinct indices
        indices = random.sample(range(list_len), 12)
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        # Create full list for evaluation
        poly = [0] * list_len
        for idx, c in zip(indices, coeffs):
            poly[idx] = c
            
        cnt, _ = count_nonzero_from_poly(poly)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = poly[:]
            
        # Local search on this candidate
        current_count = cnt
        improved = True
        # Limit local search depth/steps to avoid timeout
        steps = 0
        while improved and steps < 100:
            improved = False
            steps += 1
            # Pick a random term to modify
            k = random.randint(0, 11)
            old_idx = indices[k]
            old_coeff = coeffs[k]
            
            # Option 1: Change coefficient
            new_coeff = random.choice([-3, -2, -1, 1, 2, 3])
            if new_coeff != old_coeff:
                indices[k] = old_idx
                coeffs[k] = new_coeff
                poly = [0] * list_len
                for idx, c in zip(indices, coeffs):
                    poly[idx] = c
                new_count, _ = count_nonzero_from_poly(poly)
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_coeffs = poly[:]
                else:
                    coeffs[k] = old_coeff # revert
            
            # Option 2: Move term to new index
            if not improved:
                new_idx = random.randint(0, max_degree)
                # Ensure no collision with other indices
                if new_idx not in indices:
                    old_idx = indices[k]
                    indices[k] = new_idx
                    poly = [0] * list_len
                    for idx, c in zip(indices, coeffs):
                        poly[idx] = c
                    new_count, _ = count_nonzero_from_poly(poly)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_coeffs = poly[:]
                    else:
                        indices[k] = old_idx # revert

    # Final local search on the best found
    if best_count < 23:
        # Reconstruct indices and coeffs from best_coeffs
        nz = [(i, c) for i, c in enumerate(best_coeffs) if c != 0]
        indices = [x[0] for x in nz]
        coeffs = [x[1] for x in nz]
        
        current_count = best_count
        improved = True
        steps = 0
        while improved and steps < 500:
            improved = False
            steps += 1
            k = random.randint(0, 11)
            old_idx = indices[k]
            old_coeff = coeffs[k]
            
            # Try changing coefficient to any small integer
            for new_coeff in range(-10, 11):
                if new_coeff == 0 or new_coeff == old_coeff:
                    continue
                indices[k] = old_idx
                coeffs[k] = new_coeff
                poly = [0] * list_len
                for idx, c in zip(indices, coeffs):
                    poly[idx] = c
                new_count, _ = count_nonzero_from_poly(poly)
                if new_count < current_count:
                    current_count = new_count
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_coeffs = poly[:]
                    break # restart search with new best
                else:
                    coeffs[k] = old_coeff # revert
            
            # Try moving term
            if not improved:
                for new_idx in range(max_degree + 1):
                    if new_idx == old_idx or new_idx in indices:
                        continue
                    indices[k] = new_idx
                    poly = [0] * list_len
                    for idx, c in zip(indices, coeffs):
                        poly[idx] = c
                    new_count, _ = count_nonzero_from_poly(poly)
                    if new_count < current_count:
                        current_count = new_count
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_coeffs = poly[:]
                        break
                    else:
                        indices[k] = old_idx # revert

    return best_coeffs
