"""Rewritten program for the suite task; improved greedy set-cover with time guard."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill
    
    # Generate all words in the space
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = q ** n
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (words within Hamming distance <= radius) for each word
    # For radius=1, each word has 1 + n*(q-1) neighbors
    neighbors = [None] * total_words
    for i, w in enumerate(all_words):
        covered = set()
        covered.add(i)
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_w = list(w)
                    new_w[pos] = val
                    idx = word_to_idx[tuple(new_w)]
                    covered.add(idx)
        if radius >= 2:
            # For radius 2, add all pairs of positions
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    orig1, orig2 = w[pos1], w[pos2]
                    for val1 in range(q):
                        for val2 in range(q):
                            if val1 == orig1 and val2 == orig2:
                                continue
                            if val1 == orig1 or val2 == orig2:
                                continue  # Already counted in radius 1
                            new_w = list(w)
                            new_w[pos1] = val1
                            new_w[pos2] = val2
                            idx = word_to_idx[tuple(new_w)]
                            covered.add(idx)
        neighbors[i] = covered
    
    # Greedy set cover with efficient tracking
    # uncovered_mask: boolean array, True if word is uncovered
    uncovered_mask = np.ones(total_words, dtype=bool)
    # coverage_count[i] = number of codewords whose neighborhood contains word i
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    codewords = []
    
    while np.any(uncovered_mask) and time.time() < deadline:
        # For each candidate center i, marginal gain = number of uncovered words in neighbors[i]
        # We compute this efficiently: for each i, gain[i] = sum(uncovered_mask[j] for j in neighbors[i])
        
        best_idx = -1
        best_gain = 0
        
        for i in range(total_words):
            # Count uncovered words in neighborhood of i
            neigh_indices = list(neighbors[i])
            gain = np.sum(uncovered_mask[neigh_indices])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == 0:
                    break
        
        if best_idx == -1 or best_gain == 0:
            break
        
        codewords.append(list(all_words[best_idx]))
        
        # Update: mark words in neighbors[best_idx] as covered
        neigh_indices = list(neighbors[best_idx])
        uncovered_mask[neigh_indices] = False
        
        # Update coverage_count for all words (for potential future use, but mainly for tracking)
        coverage_count[neigh_indices] += 1
    
    # If we timed out and haven't covered everything, fall back to a valid construction
    if np.any(uncovered_mask):
        # Direct product construction as fallback
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            fallback = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
            return fallback
        else:
            return [list(w) for w in all_words]
    
    return codewords
# EVOLVE-BLOCK-END
