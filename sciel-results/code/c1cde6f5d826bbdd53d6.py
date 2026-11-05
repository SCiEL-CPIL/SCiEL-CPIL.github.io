"""Ternary covering code K_3(7,1) via greedy set cover with vectorized coverage tracking."""
import numpy as np
import itertools
import time


def construct(q, n, radius):
    # For the specific task: q=3, n=7, radius=1
    # We use a greedy set-cover approach with NumPy for speed.
    # The search space is 3^7 = 2187 words.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct sum construction for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # For small n, use all words
            return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Precompute all 2187 words as indices 0..2186
    total_words = 3 ** n  # 2187
    
    # Convert word index to its ternary representation (list of n digits)
    # index -> word
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word
    
    # Precompute for each word, the list of words within Hamming distance 1
    # For radius 1 in ternary: the word itself + for each position, change to other 2 symbols
    # Total neighbors per word: 1 + 7*2 = 15
    
    # Precompute coverage: for each codeword candidate, which words it covers
    # We'll use a bitset approach or boolean array
    
    # For speed, precompute the set of covered indices for each possible codeword
    # covered_by[c] = set of word indices covered by codeword c
    
    # Actually, let's precompute for each word index, which codewords cover it
    # Or better: for each codeword, the list of words it covers
    
    # Let's precompute the neighbor lists
    # neighbors[c] = list of word indices within distance 1 of word c
    neighbors = []
    for c in range(total_words):
        word = idx_to_word(c)
        covered = [c]
        for pos in range(n):
            orig = word[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                new_word = word[:]
                new_word[pos] = new_val
                # Convert back to index
                idx = 0
                for i in range(n):
                    idx = idx * 3 + new_word[i]
                covered.append(idx)
        neighbors.append(covered)
    
    # Greedy set cover
    # uncovered = set of all words
    # At each step, pick the codeword that covers the most uncovered words
    
    uncovered = np.ones(total_words, dtype=bool)
    codewords = []
    
    # Precompute neighbors as numpy arrays for fast intersection
    neighbor_arrays = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    start_time = time.time()
    
    while uncovered.any():
        if time.time() - start_time > 100:  # Leave some time for finishing
            break
        
        # Find the codeword that covers the most uncovered words
        best_count = -1
        best_code = -1
        
        for c in range(total_words):
            # Count how many of neighbors[c] are still uncovered
            nb = neighbor_arrays[c]
            count = np.sum(uncovered[nb])
            if count > best_count:
                best_count = count
                best_code = c
                if best_count == 0:
                    break
        
        if best_code == -1 or best_count == 0:
            # Should not happen, but break to avoid infinite loop
            break
        
        # Add this codeword
        codewords.append(best_code)
        # Update uncovered
        nb = neighbor_arrays[best_code]
        uncovered[nb] = False
    
    # Convert codeword indices back to word lists
    result = [idx_to_word(c) for c in codewords]
    
    # Ensure distinctness (should be guaranteed by construction)
    # The greedy algorithm never picks the same codeword twice since once picked,
    # it's still in the candidate set but its coverage is already accounted for.
    # Actually, we might pick it again if it still covers uncovered words? No,
    # because after picking, all words it covers are marked uncovered=False,
    # so next time its count would be lower. But it could still be picked if
    # it covers some newly uncovered... no, we only mark as covered, never uncovered.
    # So each codeword is picked at most once.
    
    return result
