import numpy as np
import itertools

def construct(q, n, radius):
    # Specific optimization for the known hard case: q=3, n=8, radius=1
    # Using efficient Greedy Set Cover with incremental counting
    
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** 8  # 6561
        sphere_size = 1 + 2 * 8  # 17
        
        # Precompute sphere offsets for each center
        # For center c with digits d, the sphere contains:
        # c itself, and c +/- 3^i for each position i (with wraparound handled)
        # Actually, we need to handle modular arithmetic properly.
        
        # Precompute: for each center index, store the 17 indices in its sphere
        spheres = np.empty((total_words, sphere_size), dtype=np.int32)
        
        # Helper to convert index to digits (base 3)
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
        
        # Precompute powers of 3
        pow3 = [1] * n
        for i in range(1, n):
            pow3[i] = pow3[i-1] * 3
        
        # For each center, compute its sphere
        for i in range(total_words):
            digits = idx_to_digits(i)
            spheres[i, 0] = i
            for pos in range(n):
                orig_d = digits[pos]
                # +1 mod 3
                new_d = (orig_d + 1) % 3
                # The change in index: (new_d - orig_d) * 3^pos
                # If orig_d = 2, new_d = 0, change = -2 * 3^pos
                # If orig_d = 0, new_d = 1, change = +1 * 3^pos
                # If orig_d = 1, new_d = 2, change = +1 * 3^pos
                if orig_d == 2:
                    delta = -2 * pow3[pos]
                else:
                    delta = 1 * pow3[pos]
                spheres[i, 1 + 2*pos] = i + delta
                
                # -1 mod 3
                new_d = (orig_d - 1) % 3
                if orig_d == 0:
                    delta = 2 * pow3[pos]
                else:
                    delta = -1 * pow3[pos]
                spheres[i, 2 + 2*pos] = i + delta
        
        # Greedy set cover with incremental count updates
        uncovered = np.ones(total_words, dtype=bool)
        # count[j] = number of uncovered words in sphere j
        count = np.full(total_words, sphere_size, dtype=np.int32)
        
        codewords = []
        
        while True:
            # Find the center with maximum count
            best_center = int(np.argmax(count))
            if count[best_center] == 0:
                break
            
            codewords.append(best_center)
            
            # Update: mark sphere as covered and update counts for overlapping spheres
            sphere = spheres[best_center]
            
            # For each word in this sphere that was uncovered,
            # decrement the count of all spheres that contain that word
            # We need an inverse mapping: for each word w, which spheres contain w?
            # This is expensive to precompute. Instead, we can do a simpler update.
            
            # Simpler approach: just recompute counts for affected spheres
            # A word w in sphere[best_center] is in sphere[j] if w is within distance 1 of center j.
            # This means j is in sphere[w] (symmetry of Hamming distance).
            # So for each uncovered word w in sphere[best_center], we need to decrement count[j] for all j in sphere[w].
            
            # Precompute inverse: inv_spheres[w] = list of centers whose sphere contains w
            # This is the same as spheres since the relation is symmetric!
            # sphere[j] contains w iff sphere[w] contains j.
            
            # For efficiency, only process words that were actually uncovered
            for w in sphere:
                if uncovered[w]:
                    uncovered[w] = False
                    # Decrement count for all spheres containing w
                    # Those are the centers in spheres[w]
                    count[spheres[w]] -= 1
            
            # Optimization: skip spheres with count 0 in future argmax by setting to -1
            # Actually, argmax will naturally avoid them if they're 0 and others are positive.
            # But if all are 0, we break.
        
        # Convert indices back to list of digits
        result = []
        for idx in codewords:
            result.append(idx_to_digits(idx))
            
        return result

    # Generic fallback for other parameters
    # For small spaces, use brute force or greedy
    total = q ** n
    if total <= 1000:
        # Brute force: return all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For larger spaces, use a simple greedy with basic counting
    # This may not be optimal but should be valid
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
    
    # Compute spheres
    spheres = []
    for i in range(total):
        digits = idx_to_digits(i)
        sphere = set([i])
        for pos in range(n):
            for delta in range(1, q):
                new_d = (digits[pos] + delta) % q
                new_digits = digits.copy()
                new_digits[pos] = new_d
                sphere.add(digits_to_idx(new_digits))
        spheres.append(list(sphere))
    
    spheres_np = np.array(spheres, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while True:
        counts = uncovered[spheres_np].sum(axis=1)
        best_center = int(np.argmax(counts))
        if counts[best_center] == 0:
            break
        codewords.append(best_center)
        uncovered[spheres_np[best_center]] = False
    
    return [idx_to_digits(idx) for idx in codewords]
