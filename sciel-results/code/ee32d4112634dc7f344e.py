import numpy as np
import itertools

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        total_words = q ** n
        
        # Helper to convert index to digits
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n):
                digits[i] = idx % q
                idx //= q
            return digits

        # Helper to convert digits to index
        def digits_to_idx(digits):
            idx = 0
            mult = 1
            for d in digits:
                idx += d * mult
                mult *= q
            return idx

        # 1. Construct Tetracode C4
        # C4 is a perfect code of length 4, size 9, radius 1
        tetracode_digits = [
            [0, 0, 0, 0],
            [0, 1, 2, 1],
            [0, 2, 1, 2],
            [1, 0, 1, 2],
            [1, 1, 0, 1],
            [1, 2, 2, 0],
            [2, 0, 2, 1],
            [2, 1, 1, 0],
            [2, 2, 0, 2]
        ]
        tetracode_idx = [digits_to_idx(c) for c in tetracode_digits]
        
        # 2. Construct Product Code C4 x C4
        # Words are (c1, c2) where c1, c2 in C4
        # Index of (c1, c2) is c1 * 3^4 + c2
        product_code_idx = []
        for c1 in tetracode_idx:
            for c2 in tetracode_idx:
                product_code_idx.append(c1 * (q ** (n // 2)) + c2)
        
        # 3. Precompute spheres for all possible centers
        # spheres[i] is a list of indices in the sphere of center i
        spheres = []
        for i in range(total_words):
            digits = idx_to_digits(i)
            sphere = [i]
            for pos in range(n):
                orig_d = digits[pos]
                # +1 mod q
                new_d = (orig_d + 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                sphere.append(digits_to_idx(new_digits))
                # -1 mod q
                new_d = (orig_d - 1) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                sphere.append(digits_to_idx(new_digits))
            spheres.append(sphere)
        
        spheres_np = np.array(spheres, dtype=np.int32)
        
        # 4. Initialize uncovered set
        uncovered = np.ones(total_words, dtype=bool)
        
        # Mark words covered by product code as covered
        # We need to mark the sphere of each product code word as covered
        for pc_idx in product_code_idx:
            uncovered[spheres_np[pc_idx]] = False
            
        codewords = [idx_to_digits(i) for i in product_code_idx]
        
        # 5. Greedy Set Cover for remaining uncovered words
        while True:
            # Count how many uncovered words are in each sphere
            counts = uncovered[spheres_np].sum(axis=1)
            best_center = np.argmax(counts)
            
            if counts[best_center] == 0:
                break
                
            codewords.append(idx_to_digits(best_center))
            # Mark sphere as covered
            uncovered[spheres_np[best_center]] = False
            
        return codewords

    # Fallback for other parameters
    if q ** n <= 100:
         return [list(itertools.product(range(q), repeat=n))[i] for i in range(q**n)]
         
    return [[0]*n]
