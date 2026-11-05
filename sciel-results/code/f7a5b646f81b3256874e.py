"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    # Peeling heuristic based on the dual distance-2 packing.
    # Start with the tetracode direct sum, then iteratively remove redundant codewords.
    
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # q=3, n=8, radius=1
    # Start with the direct sum of the ternary tetracode (486 codewords)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Precompute word_to_idx mapping using base-3 representation
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx
    
    # Precompute neighborhoods for each codeword as numpy arrays
    neighborhoods = []
    for c in code:
        neigh = [word_to_idx(c)]
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    new_c = c[:]
                    new_c[pos] = val
                    neigh.append(word_to_idx(new_c))
        neighborhoods.append(np.array(neigh, dtype=np.int32))
    
    # Coverage count: how many codewords cover each word
    coverage = np.zeros(6561, dtype=np.int32)
    for neigh in neighborhoods:
        np.add.at(coverage, neigh, 1)
    
    m = len(code)
    
    # Peeling loop: repeatedly find and remove the most redundant codeword
    # A codeword is removable if all words in its neighborhood have coverage >= 2
    # We prioritize codewords whose neighborhood has the highest minimum coverage
    
    changed = True
    while changed:
        changed = False
        best_idx = -1
        best_min_cov = 0
        
        # Vectorized check: for each codeword, find min coverage in its neighborhood
        # We need to check which codewords have all neighbors with coverage >= 2
        for i in range(m):
            min_cov = np.min(coverage[neighborhoods[i]])
            if min_cov >= 2 and min_cov > best_min_cov:
                best_min_cov = min_cov
                best_idx = i
        
        if best_idx >= 0:
            # Remove codeword best_idx
            np.add.at(coverage, neighborhoods[best_idx], -1)
            neighborhoods.pop(best_idx)
            code.pop(best_idx)
            m -= 1
            changed = True
    
    return code
# EVOLVE-BLOCK-END
