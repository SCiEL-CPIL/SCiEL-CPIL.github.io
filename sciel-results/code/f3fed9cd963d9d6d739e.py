import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Specialized solver for q=3, n=6, radius=1
    return _solve_q3_n6_r1(deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    total_words = q ** n  # 729
    
    # Generate all words as integers 0..728
    all_words = list(range(total_words))
    
    # Convert integer to list of digits
    def int_to_word(w):
        return [w % q, (w // q) % q, (w // (q*q)) % q, (w // (q*q*q)) % q, (w // (q*q*q*q)) % q, (w // (q*q*q*q*q)) % q]
    
    def word_to_int(word):
        result = 0
        for i, d in enumerate(word):
            result += d * (q ** i)
        return result
    
    # Precompute neighbors for each word (words at Hamming distance <= 1)
    # For each word, neighbors = {word} + {words at distance 1}
    # At distance 1: change one position to a different value (2 choices per position, 6 positions = 12)
    # Total: 1 + 12 = 13 neighbors
    
    # Precompute neighbor lists
    neighbor_lists = []
    for w in all_words:
        word = int_to_word(w)
        neighbors = [w]
        for pos in range(n):
            orig = word[pos]
            for new_val in range(q):
                if new_val != orig:
                    new_word = word[:]
                    new_word[pos] = new_val
                    neighbors.append(word_to_int(new_word))
        neighbor_lists.append(neighbors)
    
    # Initialize with tetracode direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            cw = h + list(tail)
            initial_codewords.append(word_to_int(cw))
    
    # Remove duplicates (shouldn't be any, but just in case)
    initial_codewords = list(set(initial_codewords))
    initial_codewords.sort()
    
    codewords = set(initial_codewords)
    
    # Coverage count: for each word, how many codewords cover it
    coverage = np.zeros(total_words, dtype=np.int32)
    for cw in codewords:
        for nb in neighbor_lists[cw]:
            coverage[nb] += 1
    
    # Greedy removal
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try to remove each codeword
        for cw in list(codewords):
            if time.time() > deadline:
                break
            # Check if cw is removable: all words it covers have coverage >= 2
            # (i.e., after removing cw, they still have coverage >= 1)
            removable = True
            for nb in neighbor_lists[cw]:
                if coverage[nb] <= 1:
                    removable = False
                    break
            if removable:
                # Remove cw
                codewords.remove(cw)
                for nb in neighbor_lists[cw]:
                    coverage[nb] -= 1
                improved = True
    
    # Convert back to list of words
    result = [int_to_word(cw) for cw in sorted(codewords)]
    
    return result
