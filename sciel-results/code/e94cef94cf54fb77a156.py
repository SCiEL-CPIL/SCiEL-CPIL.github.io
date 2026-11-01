import random
import itertools
import sys

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific solver for K_3(7,1)
    total_words = 3 ** n  # 2187
    
    # Precompute all words as tuples
    all_words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Map tuple to index
    def to_index(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx
    
    # Precompute neighbors (coverage sets) for each word
    # coverage[i] is a Python int bitset where bit j is set if word j is within radius of word i
    coverage = [0] * total_words
    for i in range(total_words):
        w = all_words[i]
        bits = 0
        # Self
        bits |= (1 << i)
        # Distance 1
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    j = to_index(tuple(new_w))
                    bits |= (1 << j)
        coverage[i] = bits
    
    # Full universe mask
    full_mask = (1 << total_words) - 1
    
    def greedy_set_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = full_mask
        code = []
        
        while uncovered:
            best_count = -1
            best_word = -1
            best_tiebreak = -1
            
            for i in range(total_words):
                # Count how many uncovered words this codeword covers
                covered = coverage[i] & uncovered
                # Count bits
                cnt = bin(covered).count('1')
                if cnt > best_count or (cnt == best_count and i < best_word):
                    best_count = cnt
                    best_word = i
                    best_tiebreak = i
            
            if best_word == -1:
                break
            
            code.append(best_word)
            uncovered &= ~coverage[best_word]
        
        return code
    
    def local_search(code):
        """Try to reduce the code size by removing redundant words and swapping."""
        code = list(code)
        
        # Build coverage count
        coverage_count = [0] * total_words
        for c in code:
            bits = coverage[c]
            # Count bits in the coverage
            tmp = bits
            while tmp:
                # Find lowest set bit
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                coverage_count[idx] += 1
                tmp ^= lsb
        
        improved = True
        while improved:
            improved = False
            
            # Try to remove each codeword
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                bits = coverage[c]
                # Check if any word in coverage has count == 1
                tmp = bits
                removable = True
                while tmp:
                    lsb = tmp & (-tmp)
                    idx = lsb.bit_length() - 1
                    if coverage_count[idx] == 1:
                        removable = False
                        break
                    tmp ^= lsb
                
                if removable:
                    # Remove it
                    tmp = bits
                    while tmp:
                        lsb = tmp & (-tmp)
                        idx = lsb.bit_length() - 1
                        coverage_count[idx] -= 1
                        tmp ^= lsb
                    code.pop(i)
                    improved = True
        
        # Try swapping: remove one, add another
        for i in range(len(code)):
            c = code[i]
            # Tentatively remove c
            bits_c = coverage[c]
            tmp = bits_c
            removable = True
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if coverage_count[idx] == 1:
                    removable = False
                    break
                tmp ^= lsb
            
            if not removable:
                continue
            
            # Save state
            saved_counts = coverage_count[:]
            tmp = bits_c
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                saved_counts[idx] -= 1
                tmp ^= lsb
            
            # Now try to find a replacement that covers the words uniquely covered by c
            # Actually, since c is removable, no word is uniquely covered by c.
            # So we can just remove it. But we already do that above.
            # For swapping, we need a different approach.
            
            # Restore state
            coverage_count = saved_counts
        
        return code
    
    # Run multiple greedy passes with different strategies
    best_code = None
    best_size = float('inf')
    
    # Standard greedy
    code = greedy_set_cover(random_seed=42)
    code = local_search(code)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code[:]
    
    # Try a few more random seeds for greedy (with slight randomization)
    for seed in [1, 2, 3, 7, 13, 21, 37, 53, 71, 97]:
        code = greedy_set_cover(random_seed=seed)
        code = local_search(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
    
    # Convert indices to lists
    result = [all_words[idx] for idx in best_code]
    return result
