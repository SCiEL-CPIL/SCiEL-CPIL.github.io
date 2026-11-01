import itertools
import random

def construct(q, n, radius):
    total_words = q ** n
    
    # Helper to convert index to tuple of digits
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    # Helper to convert tuple of digits to index
    def digits_to_index(digits):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + digits[k]
        return idx

    # Precompute all words' digits
    all_digits = [index_to_digits(i) for i in range(total_words)]
    
    # Precompute spheres as bitmasks and lists
    sphere_masks = [0] * total_words
    sphere_lists = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        base_digits = all_digits[i]
        sphere_set = set()
        for r in range(radius + 1):
            for pos_subset in itertools.combinations(range(n), r):
                if r == 0:
                    sphere_set.add(i)
                    continue
                values = range(q)
                for vals in itertools.product(values, repeat=r):
                    new_digits = base_digits[:]
                    for k, pos in enumerate(pos_subset):
                        new_digits[pos] = vals[k]
                    new_idx = digits_to_index(new_digits)
                    sphere_set.add(new_idx)
        
        sphere_list = list(sphere_set)
        sphere_lists[i] = sphere_list
        mask = 0
        for w in sphere_list:
            mask |= (1 << w)
        sphere_masks[i] = mask

    best_codewords = None
    best_size = float('inf')
    
    # Run multiple greedy instances
    # Bitmask operations are very fast, so we can increase runs
    num_runs = 500 
    all_mask = (1 << total_words) - 1
    
    for seed in range(num_runs):
        rng = random.Random(seed)
        covered_mask = 0
        cover_count = [0] * total_words
        codewords = []
        remaining = total_words
        
        while remaining > 0:
            best_idx = -1
            best_count = -1
            candidates = []
            
            # Calculate uncovered mask once per step
            # ~covered_mask has infinite 1s, so mask with all_mask
            uncovered_mask = all_mask & (~covered_mask)
            
            for i in range(total_words):
                # Count bits set in intersection
                # .bit_count() is available in Python 3.8+
                new_count = (sphere_masks[i] & uncovered_mask).bit_count()
                
                if new_count > best_count:
                    best_count = new_count
                    candidates = [i]
                elif new_count == best_count and new_count > 0:
                    candidates.append(i)
            
            if best_count == 0:
                break
            
            best_idx = rng.choice(candidates)
            
            codewords.append(best_idx)
            
            # Update state
            # We need to update cover_count for post-processing
            for w in sphere_lists[best_idx]:
                if cover_count[w] == 0:
                    remaining -= 1
                cover_count[w] += 1
            
            covered_mask |= sphere_masks[best_idx]
        
        # Post-processing: Remove redundant codewords
        changed = True
        while changed:
            changed = False
            for i in range(len(codewords)):
                c = codewords[i]
                redundant = True
                for w in sphere_lists[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    for w in sphere_lists[c]:
                        cover_count[w] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
    # Convert indices back to list of lists
    result = []
    for idx in best_codewords:
        result.append(index_to_digits(idx))
        
    return result
