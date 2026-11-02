"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    deadline = time.time() + 115.0  # safety margin before 120s limit

    # Base construction: direct sum of ternary tetracode with full tails
    # This gives 9 * 3^(n-4) codewords
    tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])
    if n == 4:
        codewords = tetracode.tolist()
        return codewords
    
    # For n > 4, generate all tail combinations
    tails = np.array(list(itertools.product(range(3), repeat=n-4)))
    # codewords = [tetracode[i] + tail for i in range(9) for tail in tails]
    # More efficiently: broadcast
    base = np.tile(tetracode, (len(tails), 1))
    tail_exp = np.repeat(tails, 9, axis=0)
    # Actually need to interleave properly: for each tetracode word, combine with each tail
    # Let's do it more carefully
    all_cw = []
    for i in range(9):
        for j in range(len(tails)):
            all_cw.append(list(tetracode[i]) + list(tails[j]))
    
    codewords = all_cw
    n_cw = len(codewords)
    
    if n_cw <= 156:
        return codewords
    
    # Now try to reduce the code using a greedy removal approach
    # A codeword can be removed if every word in its sphere is covered by another codeword
    # This is expensive for large n, so we'll use a heuristic approach
    
    if n <= 7 and time.time() < deadline:
        # Try iterative removal: repeatedly find a codeword whose removal doesn't break coverage
        # and remove it
        cw_arr = np.array(codewords)
        removed = [False] * n_cw
        
        # For efficiency, we'll track which words are covered by which codewords
        # But for n=7, the space is 2187, so we can afford some computation
        
        # Precompute spheres for each codeword (as sets of indices in the full space)
        # Map each word to an index: sum(cw[i] * 3^(n-1-i))
        def word_to_idx(word):
            idx = 0
            for i, v in enumerate(word):
                idx += v * (3 ** (n - 1 - i))
            return idx
        
        # Generate all possible neighbors (distance <= radius) for each codeword
        # For radius 1, neighbors are: the word itself + n*q words at distance 1
        # Total: 1 + n*(q-1) = 1 + 7*2 = 15 for n=7, q=3
        
        # Build coverage count: for each word in the space, how many codewords cover it
        coverage = np.zeros(3**n, dtype=np.int32)
        
        # Precompute neighbor indices for each codeword
        cw_indices = [word_to_idx(cw) for cw in codewords]
        
        # For each codeword, compute its sphere
        spheres = []
        for i in range(n_cw):
            sphere = set()
            cw = codewords[i]
            # Distance 0
            sphere.add(cw_indices[i])
            # Distance 1
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = cw[:]
                        new_word[pos] = new_val
                        sphere.add(word_to_idx(new_word))
            spheres.append(sphere)
        
        # Count coverage
        for i in range(n_cw):
            for idx in spheres[i]:
                coverage[idx] += 1
        
        # Now try to remove codewords that are "redundant"
        # A codeword is redundant if for every word in its sphere, coverage >= 2
        # We do this iteratively
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(n_cw):
                if removed[i]:
                    continue
                # Check if all words in sphere[i] have coverage >= 2
                redundant = True
                for idx in spheres[i]:
                    if coverage[idx] < 2:
                        redundant = False
                        break
                if redundant:
                    # Remove this codeword
                    removed[i] = True
                    for idx in spheres[i]:
                        coverage[idx] -= 1
                    changed = True
        
        # Build the reduced code
        result = [codewords[i] for i in range(n_cw) if not removed[i]]
        return result
    
    return codewords
# EVOLVE-BLOCK-END
