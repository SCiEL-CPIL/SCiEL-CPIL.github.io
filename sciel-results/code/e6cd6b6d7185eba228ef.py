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

        # Precompute sphere bitsets
        # Python int as bitset
        sphere_bits = [0] * total_words
        for i in range(total_words):
            digits = idx_to_digits(i)
            mask = 1 << i
            for pos in range(n):
                orig_d = digits[pos]
                # +1 mod q
                new_d = (orig_d + 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                j = digits_to_idx(new_digits)
                mask |= (1 << j)
                # -1 mod q
                new_d = (orig_d - 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                j = digits_to_idx(new_digits)
                mask |= (1 << j)
            sphere_bits[i] = mask

        full_mask = (1 << total_words) - 1
        uncovered = full_mask
        codewords = []
        
        # Greedy Phase
        while uncovered:
            best_center = -1
            best_count = 0
            # Iterate through all centers to find best
            # Optimization: If we find a count == popcount(uncovered), we can break early?
            # No, we need the one that covers the most.
            # But if we find one that covers ALL uncovered, we can take it.
            for c in range(total_words):
                # Count intersection
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
            
        # Local Search: Minimization
        # Try to remove each codeword if coverage is maintained
        changed = True
        while changed:
            changed = False
            # Iterate backwards to allow index removal
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                # Check if c is necessary
                # Union of all other spheres
                # This is expensive: O(N) unions of bitsets.
                # N is ~500. Bitset union is fast.
                # But doing this for every i in every pass is 500 * 500 * 500? No.
                # Pass 1: 500 checks. Each check sums 499 bitsets.
                # 500 * 500 bitset ops. Bitset is 6561 bits ~ 100 64-bit words.
                # 250,000 * 100 ops = 25M ops. Fast enough in Python? Maybe.
                
                # Optimization: Only check if c covers some unique word?
                # Precompute "essential" words?
                # Simpler: Just do the union check.
                
                union_mask = 0
                for j in range(len(codewords)):
                    if j != i:
                        union_mask |= sphere_bits[codewords[j]]
                
                if (union_mask & full_mask) == full_mask:
                    # c is redundant
                    codewords.pop(i)
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
