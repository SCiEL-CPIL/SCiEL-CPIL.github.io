"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q with covering radius r.
    """
    if q == 3 and n == 6 and radius == 1:
        return _cover_ternary_6_1_optimized()
    else:
        return _generic_greedy_cover(q, n, radius)

def _cover_ternary_6_1_optimized():
    """
    Optimized solver for K_3(6,1) using Greedy Set Cover with Local Search and Restarts.
    """
    alphabet = range(3)
    all_words = list(itertools.product(alphabet, repeat=6))
    n_words = len(all_words) # 729
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets (indices of words covered by each word)
    # Each word covers itself + 12 neighbors (6 positions * 2 changes)
    cover_sets = []
    for w in all_words:
        covered = {word_to_idx[w]}
        for pos in range(6):
            for val in alphabet:
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                covered.add(word_to_idx[tuple(new_w)])
        cover_sets.append(covered)
        
    full_set = set(range(n_words))
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = set(full_set)
        codewords = []
        
        while uncovered:
            best_word = -1
            best_count = -1
            # Iterate through all potential words
            # Optimization: If we have many ties, randomize the order or pick the first one?
            # To allow restarts to work, we should randomize the tie-breaking.
            candidates = []
            for i in range(n_words):
                # Only consider if it covers at least one uncovered word
                # Intersection size
                count = len(cover_sets[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_word = i
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break # Should not happen if uncovered is not empty
                
            if len(candidates) > 1:
                # Random tie-breaking to enable diverse restarts
                best_word = random.choice(candidates)
                
            codewords.append(best_word)
            uncovered -= cover_sets[best_word]
            
        return codewords

    def local_search(codewords):
        """
        Try to remove redundant codewords and perform swaps.
        """
        # 1. Remove redundant codewords
        # Maintain a count of how many codewords cover each word
        counts = [0] * n_words
        for cw in codewords:
            for idx in cover_sets[cw]:
                counts[idx] += 1
                
        improved = True
        while improved:
            improved = False
            for i in range(len(codewords)):
                cw = codewords[i]
                # Check if removing cw leaves any word with count 0
                can_remove = True
                for idx in cover_sets[cw]:
                    if counts[idx] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    for idx in cover_sets[cw]:
                        counts[idx] -= 1
                    codewords.pop(i)
                    improved = True
                    break # Restart the loop as indices changed
        
        # 2. Try swaps: Replace one codeword with another to allow removal of a third?
        # Or simply replace one with another that covers "more" uncovered area?
        # Since we already removed all redundancies, every codeword covers at least one unique word.
        # A swap is useful if it allows the removal of another codeword.
        # This is O(N^2) per check, might be slow. 
        # Let's try a simpler swap: 
        # For each pair (i, j) in codewords, can we replace i with some k not in codewords 
        # such that j becomes redundant?
        
        # Given time constraints, we might skip complex swaps and rely on multiple restarts.
        # However, a simple 1-swap (replace i with k to minimize total "waste") is hard to define.
        # Let's stick to the deletion pass which is very effective after greedy.
        
        return codewords

    best_codewords = None
    best_size = float('inf')
    
    start_time = time.time()
    timeout = 110 # Leave 10s for evaluation
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() - start_time < timeout:
        seed = num_restarts
        cw = greedy_cover(seed)
        cw = local_search(cw)
        
        if len(cw) < best_size:
            best_size = len(cw)
            best_codewords = cw
            # If we hit the known optimum, we can stop early? 
            # The known optimum is 73.
            if best_size <= 73:
                break
                
        num_restarts += 1
        
    if best_codewords is None:
        # Fallback to a single deterministic run
        best_codewords = greedy_cover(0)
        best_codewords = local_search(best_codewords)
        
    # Convert indices back to words
    result = [list(all_words[i]) for i in best_codewords]
    return result

def _generic_greedy_cover(q, n, radius):
    """Generic greedy set cover for arbitrary q, n, radius."""
    alphabet = range(q)
    all_words = list(itertools.product(alphabet, repeat=n))
    n_words = len(all_words)
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage sets
    covered_sets = []
    for w in all_words:
        covered = []
        for candidate in all_words:
            dist = sum(1 for a, b in zip(w, candidate) if a != b)
            if dist <= radius:
                covered.append(candidate)
        covered_sets.append(set(word_to_idx[c] for c in covered))
    
    # Greedy set cover
    uncovered = set(range(n_words))
    codewords = []
    
    while uncovered:
        best_word = -1
        best_count = -1
        for i in range(n_words):
            count = len(covered_sets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_word = i
            if best_count == len(uncovered):
                break
        
        codewords.append(best_word)
        uncovered -= covered_sets[best_word]
    
    # Try to remove redundant codewords
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            temp_codewords = codewords[:i] + codewords[i+1:]
            covered = set()
            for cw in temp_codewords:
                covered |= covered_sets[cw]
            if covered == set(range(n_words)):
                codewords = temp_codewords
                improved = True
                break
    
    result = [list(all_words[i]) for i in codewords]
    return result
# EVOLVE-BLOCK-END
