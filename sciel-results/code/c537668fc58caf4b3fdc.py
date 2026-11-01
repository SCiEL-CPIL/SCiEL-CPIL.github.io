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
    
    # Precompute spheres as lists of indices
    # sphere[i] contains indices of all words within Hamming distance radius of i
    spheres = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        base_digits = all_digits[i]
        sphere = set()
        for r in range(radius + 1):
            for pos_subset in itertools.combinations(range(n), r):
                if r == 0:
                    sphere.add(i)
                    continue
                values = range(q)
                for vals in itertools.product(values, repeat=r):
                    new_digits = base_digits[:]
                    for k, pos in enumerate(pos_subset):
                        new_digits[pos] = vals[k]
                    new_idx = digits_to_index(new_digits)
                    sphere.add(new_idx)
        spheres[i] = list(sphere)

    best_codewords = None
    best_size = float('inf')
    
    # Run multiple greedy instances with different random seeds for tie-breaking
    num_runs = 200
    for seed in range(num_runs):
        rng = random.Random(seed)
        covered = [False] * total_words
        cover_count = [0] * total_words
        codewords = []
        remaining = total_words
        
        while remaining > 0:
            best_idx = -1
            best_count = -1
            candidates = []
            
            for i in range(total_words):
                # Count how many words in sphere[i] are NOT yet covered
                new_count = 0
                for w in spheres[i]:
                    if not covered[w]:
                        new_count += 1
                
                if new_count > best_count:
                    best_count = new_count
                    candidates = [i]
                elif new_count == best_count and new_count > 0:
                    candidates.append(i)
            
            if best_count == 0:
                break # Should not happen if code is valid
            
            # Randomly select from candidates with max coverage
            best_idx = rng.choice(candidates)
            
            codewords.append(best_idx)
            for w in spheres[best_idx]:
                if not covered[w]:
                    covered[w] = True
                    remaining -= 1
                cover_count[w] += 1
        
        # Post-processing: Remove redundant codewords
        changed = True
        while changed:
            changed = False
            # Iterate through a copy to allow removal
            for i in range(len(codewords)):
                c = codewords[i]
                # Check if c is redundant
                # c is redundant if for all w in spheres[c], cover_count[w] > 1
                redundant = True
                for w in spheres[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove c
                    for w in spheres[c]:
                        cover_count[w] -= 1
                    codewords.pop(i)
                    changed = True
                    break # Restart the loop to check new redundancies
        
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
            
    # Convert indices back to list of lists
    result = []
    for idx in best_codewords:
        result.append(index_to_digits(idx))
        
    return result
