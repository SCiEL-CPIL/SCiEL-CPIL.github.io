"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _all_neighbors(word, q, n, radius):
    """Yield all words within Hamming distance <= radius of word."""
    if radius == 0:
        yield tuple(word)
        return
    # For radius 1, yield word itself and all single-position changes
    for pos in range(n):
        orig = word[pos]
        for val in range(q):
            if val == orig:
                continue
            new_word = list(word)
            new_word[pos] = val
            yield tuple(new_word)
    # Yield the word itself
    yield tuple(word)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s to be safe within 150s limit
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute all possible words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Coverage array: how many codewords cover each word
    coverage = np.zeros(total_words, dtype=np.int32)
    
    # Greedy construction: repeatedly pick the word that covers the most uncovered words
    codewords = []
    
    # Precompute neighbor lists for each word (for radius 1, this is efficient)
    neighbor_indices = {}
    for w in all_words:
        neighbors = list(_all_neighbors(w, q, n, radius))
        neighbor_indices[w] = [word_to_idx[nei] for nei in neighbors if nei in word_to_idx]
    
    while True:
        # Check deadline
        if time.time() > deadline - 5.0:
            break
        
        # Find the word that covers the most currently uncovered words
        best_word = None
        best_new_coverage = 0
        
        # Sample-based search: check a subset of words to find good candidates
        # Start with words that have low coverage
        uncovered_words = np.where(coverage == 0)[0]
        
        if len(uncovered_words) == 0:
            break
        
        # Try all uncovered words and some covered words
        candidate_indices = list(uncovered_words[:500])  # First 500 uncovered
        # Also add some random covered words for diversity
        if len(coverage) > 100:
            random_candidates = np.random.choice(len(coverage), size=min(200, len(coverage)), replace=False)
            candidate_indices.extend(random_candidates.tolist())
        
        for idx in candidate_indices:
            word = all_words[idx]
            if word in neighbor_indices:
                new_cov = sum(1 for ni in neighbor_indices[word] if coverage[ni] == 0)
                if new_cov > best_new_coverage:
                    best_new_coverage = new_cov
                    best_word = word
        
        if best_word is None or best_new_coverage == 0:
            # Try all words if sampling failed
            for idx in range(total_words):
                word = all_words[idx]
                if word in neighbor_indices:
                    new_cov = sum(1 for ni in neighbor_indices[word] if coverage[ni] == 0)
                    if new_cov > best_new_coverage:
                        best_new_coverage = new_cov
                        best_word = word
        
        if best_word is None:
            break
        
        # Add the best word
        codewords.append(best_word)
        for ni in neighbor_indices.get(best_word, []):
            coverage[ni] += 1
        
        # Check if fully covered
        if np.all(coverage > 0):
            break
    
    # Local search improvement: try to remove codewords that don't reduce coverage
    if time.time() < deadline - 10.0:
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline - 10.0:
                    break
                
                # Temporarily remove codeword i
                word_to_remove = codewords[i]
                for ni in neighbor_indices.get(word_to_remove, []):
                    coverage[ni] -= 1
                
                # Check if any word becomes uncovered
                if np.any(coverage == 0):
                    # Restore
                    for ni in neighbor_indices.get(word_to_remove, []):
                        coverage[ni] += 1
                else:
                    # Keep it removed
                    codewords.pop(i)
                    improved = True
    
    return [list(w) for w in codewords]
# EVOLVE-BLOCK-END
