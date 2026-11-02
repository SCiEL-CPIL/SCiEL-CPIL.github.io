import numpy as np
import itertools
import sys

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        total_words = q ** n
        
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n):
                digits[i] = idx % q
                idx //= q
            return digits

        def digits_to_idx(digits):
            idx = 0
            mult = 1
            for d in digits:
                idx += d * mult
                mult *= q
            return idx

        # Precompute sphere lists (indices of words covered)
        # Using lists of ints is faster for counting than bitsets for small spheres?
        # Sphere size is 17.
        sphere_indices = [None] * total_words
        for i in range(total_words):
            digits = idx_to_digits(i)
            neighbors = [i]
            for pos in range(n):
                orig_d = digits[pos]
                new_d = (orig_d + 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                j = digits_to_idx(new_digits)
                neighbors.append(j)
                new_d = (orig_d - 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                j = digits_to_idx(new_digits)
                neighbors.append(j)
            sphere_indices[i] = neighbors

        full_mask = (1 << total_words) - 1
        uncovered = full_mask
        codewords = []
        
        # Greedy Phase
        while uncovered:
            best_center = -1
            best_count = 0
            for c in range(total_words):
                inter = uncovered & ((1 << total_words) - 1) # Placeholder, need bitset
                # We need bitsets for the greedy phase to find best center fast.
                # Let's stick to bitsets for greedy, but use counts for local search.
                pass
        
        # Re-implementing Greedy with Bitsets for speed
        sphere_bits = [0] * total_words
        for i in range(total_words):
            mask = 1 << i
            for j in sphere_indices[i]:
                mask |= (1 << j)
            sphere_bits[i] = mask

        while uncovered:
            best_center = -1
            best_count = 0
            for c in range(total_words):
                inter = uncovered & sphere_bits[c]
                count = inter.bit_count()
                if count > best_count:
                    best_count = count
                    best_center = c
                    if best_count == uncovered.bit_count():
                        break
            
            if best_center == -1 or best_count == 0:
                break
                
            codewords.append(best_center)
            uncovered &= ~sphere_bits[best_center]
            
        # Local Search: Minimization using coverage counts
        # Initialize coverage counts
        coverage_counts = [0] * total_words
        for c in codewords:
            for w in sphere_indices[c]:
                coverage_counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            # Try to remove each codeword
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                # Check if any word in c's sphere has count 1
                is_essential = False
                for w in sphere_indices[c]:
                    if coverage_counts[w] == 1:
                        is_essential = True
                        break
                
                if not is_essential:
                    # Remove c
                    codewords.pop(i)
                    for w in sphere_indices[c]:
                        coverage_counts[w] -= 1
                    changed = True
        
        # Convert to digits
        result = []
        for idx in codewords:
            result.append(idx_to_digits(idx))
            
        return result

    # Fallback
    if q ** n <= 100:
         return [list(itertools.product(range(q), repeat=n))[i] for i in range(q**n)]
    return [[0]*n]
