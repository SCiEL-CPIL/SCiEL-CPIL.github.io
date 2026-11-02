"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # For the specific case of q=3, n=6, radius=1, we use a greedy set cover algorithm
    # followed by a local search improvement phase to minimize the number of codewords.
    # The total number of words is 3^6 = 729, which is small enough for efficient search.
    
    # If parameters differ from the known hard case, fall back to a simple construction
    # to ensure validity, though the task specifies this specific instance.
    if q != 3 or n != 6 or radius != 1:
        # Fallback: Direct sum of tetracode if n>=4, else brute force small cases
        if n < 4:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            uncovered = set(all_words)
            code = []
            while uncovered:
                best_word = None
                best_count = -1
                for w in all_words:
                    count = 0
                    for u in uncovered:
                        dist = sum(1 for a, b in zip(w, u) if a != b)
                        if dist <= radius:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_word = w
                code.append(list(best_word))
                to_remove = set()
                for u in uncovered:
                    dist = sum(1 for a, b in zip(best_word, u) if a != b)
                    if dist <= radius:
                        to_remove.add(u)
                uncovered -= to_remove
            return code
        else:
            # Use the tetracode construction as a fallback for larger n
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Specific optimization for q=3, n=6, radius=1
    # 1. Generate all 729 words and assign them integer indices 0..728
    # 2. Precompute the neighborhood (words within distance 1) for each word
    # 3. Greedy Set Cover: repeatedly pick the word that covers the most uncovered words
    # 4. Local Search: try to swap codewords or remove redundant ones to reduce size
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighborhoods for all words
    neighborhoods = []
    for i in indices:
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh.append(word_to_idx(new_w))
        neighborhoods.append(neigh)
    
    # Greedy Set Cover
    uncovered = set(indices)
    code_indices = []
    
    while uncovered:
        best_idx = -1
        best_count = -1
        
        for i in indices:
            count = 0
            for nb in neighborhoods[i]:
                if nb in uncovered:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_idx = i
                
        code_indices.append(best_idx)
        
        for nb in neighborhoods[best_idx]:
            if nb in uncovered:
                uncovered.remove(nb)
                
        if not uncovered:
            break

    # Local Search Phase: Try to improve the solution by swapping or removing codewords
    # We'll use a simple local search with random restarts and time limit
    
    def is_valid_code(code_set):
        # Check if all words are covered
        covered = set()
        for c in code_set:
            covered.update(neighborhoods[c])
        return len(covered) == total_words

    def get_coverage_counts(code_set):
        # Return array where coverage[i] = number of codewords covering word i
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for nb in neighborhoods[c]:
                coverage[nb] += 1
        return coverage

    # Try to improve the solution
    best_code = list(code_indices)
    best_size = len(best_code)
    
    # Set a time limit for local search
    start_time = time.time()
    time_limit = 100  # seconds, leaving 20s for the rest
    
    # Local search: try to remove redundant codewords first
    # Then try swaps that allow removal
    
    code_set = set(best_code)
    coverage = get_coverage_counts(code_set)
    
    # First pass: try to remove any redundant codewords
    improved = True
    while improved:
        improved = False
        for c in list(code_set):
            # Check if removing c still leaves all words covered
            # A word is still covered if its coverage count > 1
            # or if it's not covered by any other codeword
            removable = True
            for nb in neighborhoods[c]:
                if coverage[nb] <= 1:
                    removable = False
                    break
            if removable:
                code_set.remove(c)
                for nb in neighborhoods[c]:
                    coverage[nb] -= 1
                improved = True
                best_size -= 1
                best_code = list(code_set)
    
    # Second pass: try swaps that allow removal
    # For each pair (c_in, c_out) where c_in not in code, c_out in code,
    # check if adding c_in and removing c_out keeps validity and allows further removal
    
    random.seed(42)
    
    for iteration in range(50):  # Limit number of local search iterations
        if time.time() - start_time > time_limit:
            break
        
        # Try to find a swap that improves the solution
        found_improvement = False
        
        # Get list of codewords in current solution
        code_list = list(code_set)
        
        # Try random swaps
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a random codeword to remove
            c_out = random.choice(code_list)
            
            # Pick a random word to add (not already in code)
            c_in = random.choice(indices)
            if c_in in code_set:
                continue
            
            # Tentatively swap
            code_set.remove(c_out)
            code_set.add(c_in)
            
            # Update coverage
            for nb in neighborhoods[c_out]:
                coverage[nb] -= 1
            for nb in neighborhoods[c_in]:
                coverage[nb] += 1
            
            # Check if still valid
            if np.min(coverage) >= 1:
                # Try to remove redundant codewords
                removed_any = False
                for c in list(code_set):
                    removable = True
                    for nb in neighborhoods[c]:
                        if coverage[nb] <= 1:
                            removable = False
                            break
                    if removable:
                        code_set.remove(c)
                        for nb in neighborhoods[c]:
                            coverage[nb] -= 1
                        removed_any = True
                        best_size -= 1
                        best_code = list(code_set)
                
                if removed_any:
                    found_improvement = True
                    break
                else:
                    # No improvement, revert swap
                    code_set.remove(c_in)
                    code_set.add(c_out)
                    for nb in neighborhoods[c_in]:
                        coverage[nb] -= 1
                    for nb in neighborhoods[c_out]:
                        coverage[nb] += 1
            else:
                # Invalid, revert
                code_set.remove(c_in)
                code_set.add(c_out)
                for nb in neighborhoods[c_in]:
                    coverage[nb] -= 1
                for nb in neighborhoods[c_out]:
                    coverage[nb] += 1
        
        if not found_improvement:
            # No improvement in this iteration, try a different strategy
            # or break if we've been stuck for a while
            pass
    
    return [list(idx_to_word(i)) for i in best_code]

# EVOLVE-BLOCK-END
