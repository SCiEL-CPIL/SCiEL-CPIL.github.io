import numpy as np
import itertools

def construct(q, n, radius):
    # Greedy covering code construction using vectorized coverage computation
    total_words = q ** n
    
    # Generate all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute sphere indices: for each word, the indices of all words within Hamming distance <= radius
    # For radius=1, sphere size = 1 + q*(n-1)
    sphere_size = 1 + q * (n - 1) if radius == 1 else None
    
    if radius == 1:
        # Precompute sphere indices for each word
        sphere_indices = np.zeros((total_words, sphere_size), dtype=np.int64)
        for i in range(total_words):
            word = all_words[i]
            sphere_indices[i, 0] = i
            idx = 1
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_word = word.copy()
                    new_word[pos] = val
                    # Find index of new_word
                    # Convert to integer index
                    new_idx = 0
                    for j in range(n):
                        new_idx = new_idx * q + new_word[j]
                    sphere_indices[i, idx] = new_idx
                    idx += 1
    else:
        # General case: compute sphere by Hamming distance
        sphere_indices_list = []
        for i in range(total_words):
            word = all_words[i]
            sphere = [i]
            # Generate all words within radius
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for values in itertools.product(range(1, q), repeat=dist):
                        new_word = word.copy()
                        for pos, offset in zip(positions, values):
                            new_word[pos] = (word[pos] + offset) % q
                        new_idx = 0
                        for j in range(n):
                            new_idx = new_idx * q + new_word[j]
                        if new_idx not in sphere:
                            sphere.append(new_idx)
            sphere_indices_list.append(sphere)
        # Pad to same length
        max_sph = max(len(s) for s in sphere_indices_list)
        sphere_indices = np.zeros((total_words, max_sph), dtype=np.int64)
        for i, s in enumerate(sphere_indices_list):
            sphere_indices[i, :len(s)] = s
    
    # Greedy selection
    is_uncovered = np.ones(total_words, dtype=bool)
    codewords = []
    selected = set()
    
    while is_uncovered.any():
        if radius == 1:
            # Vectorized coverage computation
            coverage = is_uncovered[sphere_indices].sum(axis=1)
        else:
            coverage = np.zeros(total_words, dtype=np.int64)
            for i in range(total_words):
                coverage[i] = is_uncovered[sphere_indices[i, :len(sphere_indices_list[i])]].sum()
        
        # Pick the candidate with maximum coverage
        best_idx = np.argmax(coverage)
        
        # Add to codewords
        codewords.append(all_words[best_idx].tolist())
        selected.add(best_idx)
        
        # Update uncovered: mark all words in sphere as covered
        if radius == 1:
            is_uncovered[sphere_indices[best_idx]] = False
        else:
            is_uncovered[sphere_indices[best_idx, :len(sphere_indices_list[best_idx])]] = False
    
    return codewords
