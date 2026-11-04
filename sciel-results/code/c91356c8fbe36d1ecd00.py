import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Finds a covering code for q-ary words of length n with covering radius radius.
    Returns a list of codewords (lists of ints).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Parameters for this specific task instance
    # q=3, n=6, radius=1
    total_words = q ** n
    
    # Generate all words as integers 0..total_words-1
    # Convert integer to tuple of digits
    # We will work with indices 0..728
    
    # Precompute the coverage mask: mask[i] is a bitset or boolean array of words covered by word i
    # Since 729 is small, we can use a boolean matrix of shape (total_words, total_words)
    # Or better, list of sets? No, numpy arrays are faster for sums.
    
    # Convert index to tuple
    def int_to_tuple(x):
        t = []
        for _ in range(n):
            t.append(x % q)
            x //= q
        return tuple(t)
        
    # Precompute distances
    # dist[i, j] = 1 if hamming distance <= radius else 0
    # Actually, we just need to know if j is covered by i.
    # coverage[i] is a boolean array of length total_words
    # coverage[i][j] = True if dist(i, j) <= radius
    
    # Efficient way to generate all tuples
    words_tuples = [int_to_tuple(i) for i in range(total_words)]
    
    # Precompute coverage masks
    # coverage_mask[i] is a numpy array of bools
    coverage_mask = np.zeros((total_words, total_words), dtype=bool)
    
    for i in range(total_words):
        wi = words_tuples[i]
        for j in range(total_words):
            wj = words_tuples[j]
            # Hamming distance
            dist = sum(1 for k in range(n) if wi[k] != wj[k])
            if dist <= radius:
                coverage_mask[i, j] = True

    best_codewords = None
    best_size = float('inf')
    
    # Run multiple random restarts of Greedy + Local Search
    for restart in range(50): # Cap restarts, check time
        if time.time() > deadline - 5.0:
            break
            
        # 1. Greedy Phase
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        
        # To break ties in greedy, we can shuffle the order of candidates or use a random tie-breaker
        # Standard greedy: pick word with max new coverage.
        
        while not np.all(covered):
            if time.time() > deadline - 10.0:
                break
            
            # Calculate new coverage for each candidate
            # new_covered[i] = sum(coverage_mask[i] & ~covered)
            # This is a matrix-vector product effectively, or sum along axis
            # ~covered is a boolean array.
            # We want sum(coverage_mask[i, :] & ~covered) for all i.
            # This is equivalent to coverage_mask @ (~covered) if we treat bools as 0/1?
            # No, & is element-wise.
            # Let's use numpy:
            # uncovered_mask = ~covered
            # new_counts = np.sum(coverage_mask & uncovered_mask[np.newaxis, :], axis=1)
            
            uncovered_mask = ~covered
            new_counts = np.sum(coverage_mask & uncovered_mask[np.newaxis, :], axis=1)
            
            # Find max
            max_count = np.max(new_counts)
            if max_count == 0:
                break # Should not happen if not all covered
            
            # Get all indices with max count
            candidates = np.where(new_counts == max_count)[0]
            
            # Pick one randomly to introduce diversity
            chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            codewords.append(chosen_idx)
            covered = covered | coverage_mask[chosen_idx]
            
        # 2. Post-Optimization Phase (Remove redundant codewords)
        # Try to remove each codeword. If the set remains valid, keep it removed.
        # We iterate in a random order to avoid bias.
        indices_to_check = list(range(len(codewords)))
        random.shuffle(indices_to_check)
        
        current_codewords = list(codewords)
        
        for idx in indices_to_check:
            if time.time() > deadline - 5.0:
                break
            
            if idx >= len(current_codewords):
                continue
                
            # Try removing current_codewords[idx]
            temp_codewords = current_codewords[:idx] + current_codewords[idx+1:]
            
            # Check if temp_codewords covers all words
            # Sum of coverage masks of temp_codewords
            # If any word is covered by at least one, it's in the union.
            # We can check if the sum of coverage masks has any 0.
            
            if not temp_codewords:
                # Empty set only valid if total_words == 0
                if total_words > 0:
                    continue
            
            # Efficient check:
            # union_covered = np.zeros(total_words, dtype=bool)
            # for c in temp_codewords:
            #    union_covered |= coverage_mask[c]
            # if not np.all(union_covered):
            #    pass (put it back)
            # else:
            #    current_codewords = temp_codewords
            
            # Vectorized check is hard because it's a reduction.
            # But len(current_codewords) is ~70-80. total_words=729.
            # 70 * 729 operations is trivial.
            
            union_covered = np.zeros(total_words, dtype=bool)
            for c in temp_codewords:
                union_covered |= coverage_mask[c]
                
            if np.all(union_covered):
                current_codewords = temp_codewords
        
        if len(current_codewords) < best_size:
            best_size = len(current_codewords)
            best_codewords = current_codewords[:]
            
            # If we hit the theoretical lower bound or a very good bound, we can stop early
            # Lower bound is 71. 73 is best known.
            if best_size <= 73:
                # We can try to see if we can do better with more time, but 73 is very good.
                # Let's continue a bit to see if 72 is possible (unlikely) or just break if time is tight.
                pass

    if best_codewords is None:
        # Fallback: return the seed solution if search failed or timed out completely
        # Seed: Direct sum of tetracode and full space on last 2
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to tuples/lists
    result = [list(words_tuples[idx]) for idx in best_codewords]
    return result
