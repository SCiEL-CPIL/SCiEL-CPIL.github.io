import time
import numpy as np
import itertools


def construct(q, n, radius):
    deadline = time.time() + 130  # safety margin before 150s kill
    total = q ** n

    # Precompute all words as a 2D array of shape (total, n)
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)

    # Precompute balls: for each word, which words are within Hamming distance radius.
    # Use a bitset representation for fast coverage checks.
    # For n=6, q=3, total=729, we can use Python ints as bitsets.
    
    # Compute the list of words within distance <= radius of each word
    # ball[word_index] = set of word indices within radius
    # For efficiency, precompute using numpy
    # dist_matrix[i,j] = Hamming distance between all_words[i] and all_words[j]
    # This is 729x729 = ~531K entries, very fast
    
    diff = np.abs(all_words[:, None, :] - all_words[None, :, :])  # (total, total, n)
    dist = (diff != 0).sum(axis=2)  # (total, total)
    
    # ball_mask[i] = boolean array of which words are within radius of word i
    # For bitset: convert to Python int
    # Actually, let's use numpy boolean matrix for coverage counting
    ball_mask = dist <= radius  # (total, total) boolean
    
    # Greedy construction
    # uncovered: boolean array, True if not yet covered
    uncovered = np.ones(total, dtype=bool)
    uncovered_count = total
    code = []
    
    while uncovered_count > 0:
        if time.time() > deadline - 30:
            break
        
        # For each candidate word, count how many uncovered words it covers
        # coverage_counts[i] = number of uncovered words within radius of word i
        # = sum of uncovered over ball_mask[i]
        coverage_counts = ball_mask @ uncovered.astype(np.int32)
        
        # Pick the word with max coverage
        best = int(np.argmax(coverage_counts))
        if coverage_counts[best] == 0:
            break
        
        code.append(best)
        uncovered[ball_mask[best]] = False
        uncovered_count -= coverage_counts[best]
    
    # Redundancy removal phase
    # coverage_count[w] = number of codewords that cover word w
    # A codeword c is removable if for all w in ball(c), coverage_count[w] >= 2
    
    # Build coverage counts
    code_set = set(code)
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in code:
        coverage_count[ball_mask[c]] += 1
    
    # Iteratively remove redundant codewords
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        new_code = []
        for c in code:
            if time.time() > deadline - 10:
                new_code.append(c)
                continue
            # Check if removing c leaves all words covered
            # c is removable if min coverage_count over ball(c) >= 2
            if coverage_count[ball_mask[c]].min() >= 2:
                # Remove c
                coverage_count[ball_mask[c]] -= 1
                changed = True
            else:
                new_code.append(c)
        code = new_code
    
    # Convert to list of lists
    result = [all_words[c].tolist() for c in code]
    return result
