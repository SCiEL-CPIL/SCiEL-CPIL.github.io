import numpy as np
import random

def construct(q, n, radius):
    total_words = q ** n
    
    def index_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_index(word):
        idx = 0
        for i in range(n-1, -1, -1):
            idx = idx * q + word[i]
        return idx

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        word = index_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            original_val = word[pos]
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (original_val + delta) % q
                new_idx = word_to_index(new_word)
                neighbors[i].append(new_idx)

    # Phase 1: Greedy Construction
    covered = np.zeros(total_words, dtype=bool)
    gain = np.full(total_words, len(neighbors[0]), dtype=np.int32)
    codewords_indices = []
    
    while True:
        max_gain = np.max(gain)
        if max_gain == 0:
            break
        k = np.argmax(gain)
        codewords_indices.append(k)
        for u in neighbors[k]:
            if not covered[u]:
                covered[u] = True
                for j in neighbors[u]:
                    gain[j] -= 1
    
    # Phase 2: Local Search / Removal Optimization
    # We have a list of indices. Let's try to remove redundant ones.
    # We maintain cover_count for each word.
    cover_count = np.zeros(total_words, dtype=np.int32)
    for k in codewords_indices:
        for u in neighbors[k]:
            cover_count[u] += 1
    
    # Function to check if all covered
    def is_covered():
        return np.all(cover_count > 0)
        
    # Try to remove codewords
    # We iterate multiple times to allow for cascading removals
    # Randomize order to escape local minima? Or just iterate.
    # Since the set is small (~190), iterating is fast.
    
    changed = True
    while changed:
        changed = False
        # Shuffle to avoid bias? Or just iterate backwards to avoid index issues if we remove
        # We'll iterate over a copy of the list of indices
        indices_to_try = list(codewords_indices)
        random.shuffle(indices_to_try)
        
        for k in indices_to_try:
            if k not in codewords_indices:
                continue
                
            # Tentatively remove k
            for u in neighbors[k]:
                cover_count[u] -= 1
                
            # Check if still covered
            if np.all(cover_count > 0):
                # Valid removal
                codewords_indices.remove(k)
                changed = True
            else:
                # Invalid, restore
                for u in neighbors[k]:
                    cover_count[u] += 1
    
    # Convert indices back to words
    return [index_to_word(k) for k in codewords_indices]
