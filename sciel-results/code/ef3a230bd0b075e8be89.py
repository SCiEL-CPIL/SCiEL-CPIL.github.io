"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    # Use a peeling heuristic based on the dual distance-2 packing.
    # For q=3, n=8, radius=1:
    # 1. Generate a valid covering code (initially the direct sum construction).
    # 2. Iteratively remove the codeword whose 17-neighborhood is most densely covered
    #    by other selected codewords (maximizing redundancy).
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use the direct sum construction
        # For general case, we'll use a simple greedy or the tetracode direct sum if applicable
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
    
    # Specific optimization for q=3, n=8, radius=1
    # Start with the direct sum of the ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    m = len(code)
    if m == 0:
        return code
    
    # Map from word tuple to index in 0..6560
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx
    
    # Precompute neighborhoods for each codeword
    # neighborhoods[i] is a list of indices in 0..6560
    neighborhoods = []
    # Also build inverse mapping: covers[word_idx] = list of codeword indices
    covers = [[] for _ in range(3**n)]
    
    for i, c in enumerate(code):
        neigh = [word_to_idx(c)]
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    new_c = c[:]
                    new_c[pos] = val
                    idx = word_to_idx(new_c)
                    neigh.append(idx)
                    covers[idx].append(i)
        neighborhoods.append(neigh)
    
    # Coverage count: how many codewords cover each word
    coverage = np.zeros(3**n, dtype=np.int32)
    for neigh in neighborhoods:
        for idx in neigh:
            coverage[idx] += 1
            
    # bad_count[i]: number of words in N(i) with coverage <= 1
    # If bad_count[i] == 0, codeword i is removable
    bad_count = np.zeros(m, dtype=np.int32)
    
    for i in range(m):
        for idx in neighborhoods[i]:
            if coverage[idx] <= 1:
                bad_count[i] += 1
    
    # List of candidates that are initially removable
    candidates = [i for i in range(m) if bad_count[i] == 0]
    
    # We will remove candidates one by one.
    # Since removing a codeword only decreases coverage, bad_count only increases.
    # So a codeword that is not a candidate initially will never become one.
    # We just need to verify if a candidate is *still* removable when we process it.
    
    in_code = np.ones(m, dtype=bool)
    
    for c_idx in candidates:
        # Check if it's still removable
        if bad_count[c_idx] > 0:
            continue
        if not in_code[c_idx]:
            continue
            
        # Remove codeword c_idx
        in_code[c_idx] = False
        
        for w_idx in neighborhoods[c_idx]:
            old_cov = coverage[w_idx]
            coverage[w_idx] -= 1
            new_cov = coverage[w_idx]
            
            # If coverage dropped from 2 to 1, this word becomes "bad" for all coverers
            if old_cov == 2 and new_cov == 1:
                for c_prime in covers[w_idx]:
                    if in_code[c_prime]:
                        bad_count[c_prime] += 1
                        
    # Return remaining codewords
    remaining_code = [list(code[i]) for i in range(m) if in_code[i]]
    return remaining_code
# EVOLVE-BLOCK-END
