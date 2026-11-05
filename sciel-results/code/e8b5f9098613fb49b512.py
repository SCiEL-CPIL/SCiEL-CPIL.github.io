"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = 110.0  # safety margin before 120s limit

    # For the specific case q=3, n=8, radius=1
    # We know the optimal is 486 (van Laarhoven et al.)
    # The seed gives 9 * 3^4 = 729
    # We can do better with a greedy set cover approach

    if q != 3 or n != 8 or radius != 1:
        # Fallback to the direct sum construction
        # For general q, n, radius, use a simple construction
        if n <= 0:
            return [[]]
        # Simple construction: all words of length n-1 with 0 appended, 
        # but this is huge. Better to use the tetracode-like approach if possible.
        # For now, return a valid but potentially suboptimal solution
        # Use the seed approach generalized
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: greedy set cover
            all_words = list(itertools.product(range(q), repeat=n))
            uncovered = set(all_words)
            codewords = []
            
            # Precompute neighbors for each word
            # For efficiency, we'll compute on the fly
            while uncovered:
                if time.time() > deadline - 10:
                    break
                # Pick the word that covers the most uncovered words
                best_word = None
                best_count = -1
                # Sample if too large
                if len(uncovered) > 1000:
                    candidates = list(uncovered)[:1000]
                else:
                    candidates = list(uncovered)
                
                for w in candidates:
                    # Count how many uncovered words are within radius
                    count = 0
                    for u in uncovered:
                        if sum(1 for i in range(n) if w[i] != u[i]) <= radius:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_word = w
                
                if best_word is None:
                    break
                
                codewords.append(list(best_word))
                # Remove covered words
                to_remove = set()
                for u in uncovered:
                    if sum(1 for i in range(n) if best_word[i] != u[i]) <= radius:
                        to_remove.add(u)
                uncovered -= to_remove
            
            if uncovered:
                # Add remaining uncovered as codewords to ensure validity
                for u in uncovered:
                    codewords.append(list(u))
            
            return codewords

    # Specific case: q=3, n=8, radius=1
    # Use greedy set cover with efficient neighbor computation
    
    all_words = list(itertools.product(range(3), repeat=8))
    n_words = len(all_words)  # 6561
    
    # Map word to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    idx_to_word = all_words  # list indexed by idx
    
    # Precompute neighbors for each word
    # A word's neighbors within radius 1 include itself and all words differing in 1 position
    # For each word, there are 1 + 8*2 = 17 neighbors
    
    neighbors = []
    for w in all_words:
        nb = [w]  # include itself
        for pos in range(8):
            for val in range(3):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    nb.append(tuple(nw))
        neighbors.append(set(word_to_idx[x] for x in nb))
    
    # Greedy set cover
    uncovered = set(range(n_words))
    codewords = []
    
    # Precompute coverage: for each potential codeword, which words it covers
    # This is the neighbors list above
    
    # Greedy: repeatedly pick the word that covers the most uncovered words
    # For efficiency, maintain a count of uncovered words covered by each potential codeword
    
    # coverage_count[i] = number of uncovered words that codeword i covers
    coverage_count = [len(neighbors[i]) for i in range(n_words)]
    
    while uncovered:
        if time.time() > deadline - 5:
            break
        
        # Find the codeword that covers the most uncovered words
        best_idx = -1
        best_count = -1
        
        # Only consider indices that could potentially cover uncovered words
        # For efficiency, we can check all, but 6561 is manageable
        for i in range(n_words):
            if coverage_count[i] > best_count:
                best_count = coverage_count[i]
                best_idx = i
        
        if best_idx == -1 or best_count <= 0:
            break
        
        # Add this codeword
        codewords.append(list(all_words[best_idx]))
        
        # Remove covered words from uncovered
        covered = neighbors[best_idx]
        to_remove = uncovered & covered
        uncovered -= to_remove
        
        # Update coverage counts
        for i in range(n_words):
            if to_remove:
                # Count how many of to_remove are in neighbors[i]
                # This is expensive, so we do it lazily
                pass
        
        # Recompute coverage counts for all potential codewords
        # This is O(n_words * avg_neighbors) = 6561 * 17 ~ 111k per iteration
        # With ~486 iterations, that's ~54M operations, which might be tight
        # Let's optimize: only recompute when needed
        
        # Actually, let's just recompute coverage_count efficiently
        # coverage_count[i] = len(neighbors[i] & uncovered)
        for i in range(n_words):
            coverage_count[i] = len(neighbors[i] & uncovered)
    
    # Ensure validity: add any remaining uncovered words as codewords
    if uncovered:
        for idx in uncovered:
            codewords.append(list(all_words[idx]))
    
    return codewords
# EVOLVE-BLOCK-END
