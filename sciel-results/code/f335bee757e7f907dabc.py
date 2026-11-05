import numpy as np
import itertools

def construct(q, n, radius):
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Build Sphere Matrix M (N x N)
    # M[i, j] = 1 if j is within distance <= radius of i
    M = np.zeros((total_words, total_words), dtype=np.bool_)
    
    for i, word in enumerate(all_words):
        # Self
        M[i, i] = True
        if radius == 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word_list = list(word)
                    new_word_list[pos] = val
                    new_word = tuple(new_word_list)
                    j = word_to_idx[new_word]
                    M[i, j] = True
        else:
             # Generic radius handling (slower)
             for dist in range(1, radius + 1):
                 for positions in itertools.combinations(range(n), dist):
                     for offsets in itertools.product(range(1, q), repeat=dist):
                         new_word_list = list(word)
                         for pos, off in zip(positions, offsets):
                             new_word_list[pos] = (word[pos] + off) % q
                         new_word = tuple(new_word_list)
                         j = word_to_idx[new_word]
                         M[i, j] = True

    # Greedy Phase using Matrix Multiplication
    uncovered = np.ones(total_words, dtype=np.bool_)
    selected = []
    
    while uncovered.any():
        # coverage[i] = number of uncovered words covered by i
        # M @ uncovered
        coverage = M @ uncovered
        best_idx = np.argmax(coverage)
        
        if coverage[best_idx] == 0:
            break
            
        selected.append(best_idx)
        
        # Update uncovered
        uncovered[M[best_idx]] = False

    # Local Search: Redundancy Removal
    if selected:
        cov_count = M[selected].sum(axis=0)
    else:
        cov_count = np.zeros(total_words, dtype=np.int32)
        
    changed = True
    while changed:
        changed = False
        new_selected = []
        for c in selected:
            # Check if c is redundant
            # If min(cov_count[sphere(c)]) > 1, then removing c leaves all neighbors covered by >= 1 other codeword.
            # sphere(c) is M[c] (True where covered by c)
            min_cov = np.min(cov_count[M[c]])
            
            if min_cov > 1:
                # c is redundant
                cov_count[M[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
                
        selected = new_selected

    return [all_words[i] for i in selected]
