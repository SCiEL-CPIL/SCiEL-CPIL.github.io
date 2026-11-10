import itertools
import time

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        # Precompute neighbors for all 6561 words
        total_words = 3**n
        neighbors = [[] for _ in range(total_words)]
        
        # Helper to get neighbors
        for idx in range(total_words):
            # Convert idx to base 3 digits
            digits = [0]*n
            temp = idx
            for i in range(n-1, -1, -1):
                digits[i] = temp % 3
                temp //= 3
            
            # Self
            neighbors[idx].append(idx)
            
            # Neighbors at distance 1
            for i in range(n):
                orig = digits[i]
                for d in range(3):
                    if d == orig: continue
                    digits[i] = d
                    # Convert back to index
                    new_idx = 0
                    for j in range(n):
                        new_idx = new_idx * 3 + digits[j]
                    neighbors[idx].append(new_idx)
                digits[i] = orig
        
        # 1. Greedy Approach
        covered = [False] * total_words
        candidates = set(range(total_words))
        greedy_code = []
        
        while candidates:
            best_center = -1
            max_uncovered = -1
            
            for c in candidates:
                # Count uncovered neighbors
                unc = 0
                for x in neighbors[c]:
                    if not covered[x]:
                        unc += 1
                if unc > max_uncovered:
                    max_uncovered = unc
                    best_center = c
            
            if best_center == -1:
                break
            
            greedy_code.append(best_center)
            
            # Update covered and candidates
            to_remove = set()
            for x in neighbors[best_center]:
                if not covered[x]:
                    covered[x] = True
                    to_remove.add(x)
            
            candidates -= to_remove
        
        greedy_size = len(greedy_code)
        
        # 2. Seed Code + Reduction
        # Generate Seed Code (486 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                w = h + list(tail)
                # Convert to index
                idx = 0
                for val in w:
                    idx = idx * 3 + val
                seed_codewords.append(idx)
        
        # Reduction
        code = list(seed_codewords)
        # coverers[x] = list of indices in 'code' that cover x
        coverers = [[] for _ in range(total_words)]
        
        # Initialize coverers
        for i, c in enumerate(code):
            for x in neighbors[c]:
                coverers[x].append(i)
        
        changed = True
        while changed:
            changed = False
            # We need to check each codeword
            # Note: indices in 'code' change when we remove, so we iterate by value or use a set
            # Better: iterate through a copy of the list, or use indices in the original list?
            # Let's use a set of active codeword indices (values from seed_codewords)
            # Actually, just iterating the list and removing is fine if we are careful.
            # But 'coverers' stores indices into 'code'.
            # If we remove an element, indices shift.
            # So it's better to store the *value* (the word index) in coverers?
            # No, coverers[x] needs to know *which* codeword covers it to check redundancy.
            # If we store the word index (e.g., 123), then to check if codeword 123 is redundant,
            # we check if 123 is in coverers[x] for all x in neighbors[123].
            
            # Let's restart the reduction logic with values.
            # code = set of word indices
            # coverers[x] = set of word indices in code that cover x
            
            code_set = set(seed_codewords)
            coverers = [set() for _ in range(total_words)]
            for c in code_set:
                for x in neighbors[c]:
                    coverers[x].add(c)
            
            changed = True
            while changed:
                changed = False
                # Iterate over a copy of the list
                for c in list(code_set):
                    redundant = True
                    for x in neighbors[c]:
                        if len(coverers[x]) == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove c
                        code_set.remove(c)
                        for x in neighbors[c]:
                            coverers[x].remove(c)
                        changed = True
        
        seed_reduced_size = len(code_set)
        
        # Compare
        if greedy_size <= seed_reduced_size:
            # Convert greedy_code indices back to lists of digits
            result = []
            for idx in greedy_code:
                w = [0]*n
                temp = idx
                for i in range(n-1, -1, -1):
                    w[i] = temp % 3
                    temp //= 3
                result.append(w)
            return result
        else:
            result = []
            for idx in code_set:
                w = [0]*n
                temp = idx
                for i in range(n-1, -1, -1):
                    w[i] = temp % 3
                    temp //= 3
                result.append(w)
            return result

    else:
        # Fallback for other parameters (not expected in this task)
        # Just return a trivial code or seed
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
