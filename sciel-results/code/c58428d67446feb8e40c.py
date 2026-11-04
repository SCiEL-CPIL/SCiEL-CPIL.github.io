"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    # This gives 9 * 3^4 = 729 words for n=8.
    # We then try to reduce this via local search.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use the direct sum construction
        # For general case, use a greedy approach
        return _greedy_construct(q, n, radius)
    
    # Start with the tetracode direct sum (729 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Set a deadline: 120 seconds total, leave 10 seconds safety margin
    deadline = time.time() + 110.0
    
    # Local search to reduce code size
    code = _local_search_reduce(code, q, n, radius, deadline)
    
    return code


def _local_search_reduce(code, q, n, radius, deadline):
    """
    Local search to reduce the size of the covering code.
    Strategy:
    1. Try to remove codewords that are redundant (their covered words are still covered by others).
    2. Try swaps: replace a codeword with a neighbor that allows further reductions.
    """
    code = list(code)
    random.seed(42)
    
    # Precompute: for each codeword, the set of words it covers (distance <= radius)
    # For efficiency, we'll use a coverage counter approach
    
    # For n=8, q=3, radius=1: each codeword covers 1 + 8*2 = 17 words
    # Total space: 3^8 = 6561 words
    
    # Build coverage map: for each word in space, count how many codewords cover it
    # We'll use a dictionary or array indexed by word index
    
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
    
    def neighbors(word, rad=1):
        """Generate all words within Hamming distance rad of word."""
        if rad == 0:
            yield word
            return
        # For radius 1: the word itself + all single-position changes
        yield word
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = val
                    yield new_word
    
    # Initial coverage count
    cov_count = [0] * (q ** n)
    for cw in code:
        for w in neighbors(cw, radius):
            cov_count[word_to_idx(w)] += 1
    
    # Check if code is valid (all words covered)
    def is_valid():
        return all(c > 0 for c in cov_count)
    
    # Try to remove redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(code):
                continue
            
            cw = code[i]
            # Check if removing this codeword still leaves all words covered
            # We need to check if any word that this codeword uniquely covers exists
            redundant = True
            for w in neighbors(cw, radius):
                wi = word_to_idx(w)
                if cov_count[wi] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove this codeword
                for w in neighbors(cw, radius):
                    cov_count[word_to_idx(w)] -= 1
                code.pop(i)
                improved = True
                # Continue scanning from the beginning since indices shifted
                break
    
    # Try swaps: replace a codeword with a neighbor to enable further reductions
    # This is more complex; try a few rounds
    for round_num in range(5):
        if time.time() > deadline:
            break
        
        # Try to find a codeword and a neighbor that, when swapped, allows removal of another codeword
        # Simplified: try moving each codeword to a random neighbor and see if it helps
        for i in range(len(code)):
            if time.time() > deadline:
                break
            
            cw = code[i]
            # Generate some random neighbors of cw
            pos = random.randint(0, n - 1)
            old_val = cw[pos]
            new_val = random.randint(0, q - 1)
            if new_val == old_val:
                continue
            
            new_cw = cw[:]
            new_cw[pos] = new_val
            
            # Check if new_cw is already in the code
            if new_cw in code:
                continue
            
            # Simulate the swap: remove cw, add new_cw
            # Check if after this swap, we can remove some other codeword
            # This is expensive; do a quick check: does new_cw cover any word that was uniquely covered by another codeword?
            # If so, that codeword might become removable.
            
            # Count how many words new_cw uniquely covers (cov_count[wi] == 1 and not covered by cw)
            # Actually, let's just try the swap and see if the code remains valid, then try to reduce.
            
            # Remove cw from coverage
            for w in neighbors(cw, radius):
                cov_count[word_to_idx(w)] -= 1
            
            # Add new_cw to coverage
            for w in neighbors(new_cw, radius):
                cov_count[word_to_idx(w)] += 1
            
            # Check validity
            valid = all(c > 0 for c in cov_count)
            
            if not valid:
                # Revert
                for w in neighbors(new_cw, radius):
                    cov_count[word_to_idx(w)] -= 1
                for w in neighbors(cw, radius):
                    cov_count[word_to_idx(w)] += 1
                continue
            
            # Try to see if this swap enables a reduction
            # Check if any codeword (other than i) is now redundant
            reduction_found = False
            for j in range(len(code)):
                if j == i:
                    continue
                cj = code[j]
                redundant = True
                for w in neighbors(cj, radius):
                    wi = word_to_idx(w)
                    if cov_count[wi] == 1:
                        redundant = False
                        break
                if redundant:
                    # We found a reduction! Perform the swap and the removal
                    code[i] = new_cw
                    # Remove cj
                    for w in neighbors(cj, radius):
                        cov_count[word_to_idx(w)] -= 1
                    code.pop(j)
                    reduction_found = True
                    break
            
            if not reduction_found:
                # Revert the swap
                for w in neighbors(new_cw, radius):
                    cov_count[word_to_idx(w)] -= 1
                for w in neighbors(cw, radius):
                    cov_count[word_to_idx(w)] += 1
    
    # Final cleanup: remove any remaining redundant codewords
    for i in range(len(code) - 1, -1, -1):
        if time.time() > deadline:
            break
        cw = code[i]
        redundant = True
        for w in neighbors(cw, radius):
            wi = word_to_idx(w)
            if cov_count[wi] == 1:
                redundant = False
                break
        if redundant:
            for w in neighbors(cw, radius):
                cov_count[word_to_idx(w)] -= 1
            code.pop(i)
    
    return code


def _greedy_construct(q, n, radius):
    """Greedy construction for general parameters."""
    # This is a simple greedy: start with empty code, repeatedly add the word that covers the most uncovered words
    # For the specific case of q=3, n=8, radius=1, we use the tetracode-based approach above.
    
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
    
    def neighbors(word, rad=1):
        if rad == 0:
            yield word
            return
        yield word
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = val
                    yield new_word
    
    total_words = q ** n
    covered = [False] * total_words
    
    code = []
    
    while any(not c for c in covered):
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        for idx in range(total_words):
            w = idx_to_word(idx)
            count = 0
            for nw in neighbors(w, radius):
                ni = word_to_idx(nw)
                if not covered[ni]:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = w
            if count == 0:
                break  # No word covers any uncovered word? Shouldn't happen if not all covered
        
        if best_word is None:
            break
        
        # Add best_word to code
        code.append(best_word)
        for nw in neighbors(best_word, radius):
            covered[word_to_idx(nw)] = True
    
    return code
# EVOLVE-BLOCK-END
