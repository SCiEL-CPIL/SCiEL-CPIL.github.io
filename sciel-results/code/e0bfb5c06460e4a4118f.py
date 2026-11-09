"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    if q == 3 and n == 6 and radius == 1:
        # Total space: 3^6 = 729 words
        # Each codeword covers 1 + 6*2 = 13 words (radius 1 ball in ternary length-6)
        # Lower bound: ceil(729/13) = 57, best known is 73
        
        # Precompute all words as integers 0..728 (base-3 representation)
        def word_to_int(word):
            val = 0
            for i, d in enumerate(word):
                val = val * q + d
            return val
        
        def int_to_word(val, q, n):
            word = []
            for i in range(n):
                word.append(val % q)
                val //= q
            return word[::-1]
        
        all_words_int = set(range(q ** n))
        
        # Precompute neighborhoods for all possible words
        # For each word, store the set of integers it covers
        def get_neighbors_int(word_int, q, n, radius):
            """Get all words within Hamming distance <= radius from word_int as a set of ints."""
            word = int_to_word(word_int, q, n)
            neighbors = {word_int}
            if radius == 1:
                for pos in range(n):
                    orig = word[pos]
                    for val in range(q):
                        if val != orig:
                            new_word = word[:]
                            new_word[pos] = val
                            neighbors.add(word_to_int(new_word))
            else:
                for d in range(1, radius + 1):
                    for positions in itertools.combinations(range(n), d):
                        for values in itertools.product(range(q), repeat=d):
                            new_word = word[:]
                            for pos, val in zip(positions, values):
                                new_word[pos] = val
                            dist = sum(1 for i in range(n) if new_word[i] != word[i])
                            if dist == d:
                                neighbors.add(word_to_int(new_word))
            return neighbors
        
        # Start with a good initial code: tetracode direct sum (81 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        base_code_ints = [word_to_int(c) for c in base_code]
        
        # Greedy set cover using integer representation
        uncovered = set(all_words_int)
        selected = []
        candidate_set = set(base_code_ints)
        
        while uncovered and time.time() < deadline - 5.0:
            best_codeword = None
            best_count = -1
            
            # For each candidate codeword, count how many uncovered words it covers
            for cw_int in candidate_set:
                if time.time() > deadline - 5.0:
                    break
                neighbors = get_neighbors_int(cw_int, q, n, radius)
                count = sum(1 for w in neighbors if w in uncovered)
                if count > best_count:
                    best_count = count
                    best_codeword = cw_int
            
            if best_codeword is None or best_count <= 0:
                break
            
            selected.append(best_codeword)
            neighbors = get_neighbors_int(best_codeword, q, n, radius)
            uncovered -= neighbors
            candidate_set.discard(best_codeword)
        
        # If we still have uncovered words, try adding from all possible words
        if uncovered and time.time() < deadline - 2.0:
            for cw_int in range(q ** n):
                if time.time() > deadline - 2.0:
                    break
                if cw_int in selected or cw_int in candidate_set:
                    continue
                neighbors = get_neighbors_int(cw_int, q, n, radius)
                if any(w in uncovered for w in neighbors):
                    selected.append(cw_int)
                    uncovered -= neighbors
        
        # Convert back to list form
        return [int_to_word(cw_int, q, n) for cw_int in selected]
    else:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
