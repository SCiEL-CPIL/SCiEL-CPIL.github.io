"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _neighbors_fast(word, q, n, radius):
    """Generate all words within Hamming distance <= radius from word."""
    if radius == 0:
        return [tuple(word)]
    if radius == 1:
        result = [tuple(word)]
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    w = list(word)
                    w[i] = v
                    result.append(tuple(w))
        return result
    # For larger radius, use combinatorial generation
    result = [tuple(word)]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(q), repeat=r):
                if all(values[j] != word[positions[j]] for j in range(r)):
                    w = list(word)
                    for j, pos in enumerate(positions):
                        w[pos] = values[j]
                    result.append(tuple(w))
    return result


def _greedy_cover(q, n, radius, deadline):
    """Greedy set cover: repeatedly pick the word that covers the most uncovered words."""
    total_words = q ** n
    # Use a bitset to track covered words
    covered = np.zeros(total_words, dtype=np.uint8)
    codewords = []
    
    # Precompute neighbor sets for efficiency
    # For radius=1, each word has 1 + n*(q-1) neighbors
    # We'll compute on the fly for memory efficiency
    
    start_time = time.time()
    
    while time.time() < deadline:
        # Find the word that covers the most uncovered words
        best_word = None
        best_count = -1
        best_neighbors = None
        
        # Sample a subset of candidate words to speed up
        # For small spaces, check all; for large, sample
        if total_words <= 100000:
            candidates = range(total_words)
        else:
            # Sample candidates
            candidates = random.sample(range(total_words), min(5000, total_words))
        
        for idx in candidates:
            # Convert index to word
            word = []
            temp = idx
            for i in range(n):
                word.append(temp % q)
                temp //= q
            word = tuple(word)
            
            # Count uncovered neighbors
            neighbors = _neighbors_fast(list(word), q, n, radius)
            count = sum(1 for nb in neighbors if covered[_word_to_idx(nb, q, n)] == 0)
            
            if count > best_count:
                best_count = count
                best_word = word
                best_neighbors = neighbors
        
        if best_word is None or best_count <= 0:
            break
        
        codewords.append(list(best_word))
        for nb in best_neighbors:
            covered[_word_to_idx(nb, q, n)] = 1
        
        # Check if all covered
        if covered.all():
            break
        
        # Check deadline
        if time.time() >= deadline:
            break
    
    return codewords


def _word_to_idx(word, q, n):
    """Convert a word (tuple of ints) to its index in [0, q^n)."""
    idx = 0
    for i in range(n):
        idx = idx * q + word[i]
    return idx


def _idx_to_word(idx, q, n):
    """Convert an index to a word (list of ints)."""
    word = []
    for i in range(n):
        word.append(idx % q)
        idx //= q
    word.reverse()
    return word


def _local_search(codewords, q, n, radius, deadline):
    """Local search: try to remove codewords or swap to reduce count while maintaining coverage."""
    codewords = [list(cw) for cw in codewords]
    start_time = time.time()
    
    # Precompute coverage: for each word in space, which codewords cover it
    # This is expensive, so we'll use a simpler approach:
    # Try removing each codeword and check if coverage is still complete
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each codeword
        for i in range(len(codewords)):
            if time.time() >= deadline:
                break
            # Temporarily remove codeword i
            test_codewords = codewords[:i] + codewords[i+1:]
            # Check if all words are still covered
            if _is_valid_cover(test_codewords, q, n, radius):
                codewords = test_codewords
                improved = True
                break  # Restart from beginning after removal
    
    return codewords


def _is_valid_cover(codewords, q, n, radius):
    """Check if codewords form a valid covering."""
    if not codewords:
        return q ** n == 0
    
    total_words = q ** n
    covered = np.zeros(total_words, dtype=np.uint8)
    
    for cw in codewords:
        neighbors = _neighbors_fast(cw, q, n, radius)
        for nb in neighbors:
            covered[_word_to_idx(nb, q, n)] = 1
    
    return covered.all()


def construct(q, n, radius):
    """Construct a covering code using greedy + local search."""
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # For small cases, use exact/greedy approach
    # For the specific task: q=3, n=8, radius=1
    
    # Start with a good initial solution
    # Direct sum construction as baseline
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        initial = []
    
    # Try greedy cover if initial is empty or we want to improve
    if len(initial) == 0:
        codewords = _greedy_cover(q, n, radius, deadline)
    else:
        codewords = initial
    
    # Local search to improve
    if time.time() < deadline:
        codewords = _local_search(codewords, q, n, radius, deadline)
    
    # Ensure we have a valid cover
    if not _is_valid_cover(codewords, q, n, radius):
        # Fallback to greedy
        codewords = _greedy_cover(q, n, radius, deadline)
    
    return codewords
# EVOLVE-BLOCK-END
