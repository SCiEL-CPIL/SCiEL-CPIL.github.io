"""Greedy set-cover for ternary covering code K_3(6,1) with NumPy vectorization."""
# EVOLVE-BLOCK-START
import numpy as np

def construct(q, n, radius):
    total = q ** n
    
    # Generate all words
    all_words = np.zeros((total, n), dtype=np.uint8)
    for i in range(total):
        v = i
        for j in range(n):
            all_words[i, j] = v % q
            v //= q
            
    # Precompute coverage matrix M where M[i, j] is 1 if word j is within radius of word i
    # For radius=1, this is efficient to compute directly
    M = np.zeros((total, total), dtype=np.uint8)
    
    # We can vectorize the generation of neighbors
    # For each position pos, and each alternative value val != original,
    # the index of the neighbor can be calculated.
    
    # Let's use a faster method for M construction using index arithmetic
    # Index of a word w is sum(w[k] * q^(n-1-k))
    # Actually, my _all_words uses little-endian (v % q for j=0 is LSB).
    # So index = sum(all_words[i, j] * q^j)
    
    # Let's just loop through positions and values to fill M, it's small enough (729*6*2)
    # But we can do it smarter.
    
    # For each word i, it covers itself.
    M[np.arange(total), np.arange(total)] = 1
    
    # For each position pos in 0..n-1
    for pos in range(n):
        # For each alternative value val in 0..q-1
        for val in range(q):
            # We need to find all words i such that all_words[i, pos] != val
            # And for those, the neighbor j is obtained by changing pos to val.
            
            # Calculate the change in index.
            # index = sum(w[k] * q^k)
            # delta = (val - all_words[i, pos]) * (q ** pos)
            
            # We can compute the target indices j for all i at once?
            # It's tricky because the mapping depends on i's current value at pos.
            
            # Let's stick to a semi-vectorized approach or just simple loops if total is small.
            # Total is 729. 729 * 6 * 2 = ~8748 operations. This is negligible.
            
            for i in range(total):
                if all_words[i, pos] == val:
                    continue
                # Calculate j
                # j = i - all_words[i, pos] * (q**pos) + val * (q**pos)
                # Wait, i is the index. 
                # Let's precompute powers of q
                pass # This loop structure is getting messy. Let's just use the set-based index mapping which is robust, but store results in M.
    
    # Robust M construction
    word_to_idx = {}
    for i in range(total):
        word_to_idx[tuple(all_words[i])] = i
        
    M = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        cw = all_words[i]
        M[i, i] = 1
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = cw.copy()
                new_word[pos] = val
                j = word_to_idx.get(tuple(new_word))
                if j is not None:
                    M[i, j] = 1

    # Greedy Set Cover
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while np.any(uncovered):
        # Count how many uncovered words each candidate covers
        # M is (total, total). uncovered is (total,).
        # We want sum(M[i, :] * uncovered) for each i.
        # This is equivalent to M @ uncovered
        counts = M @ uncovered
        best_c = np.argmax(counts)
        
        if counts[best_c] == 0:
            break # Should not happen if coverage is complete
            
        codewords.append(all_words[best_c].tolist())
        
        # Update uncovered
        # uncovered = uncovered & ~M[best_c, :]
        uncovered[ M[best_c, :] == 1 ] = False
        
    return codewords
# EVOLVE-BLOCK-END
