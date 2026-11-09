"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    if q == 3 and n == 6 and radius == 1:
        # Best known construction: 73 codewords
        # Use a greedy approach to find a good covering code
        # Start with a known good seed and try to improve
        
        # The tetracode direct sum gives 9 * 3^2 = 81 words
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Try greedy set cover to reduce the number of codewords
        # Represent each codeword as a tuple for hashing
        code_tuples = [tuple(c) for c in base_code]
        
        # Precompute the neighborhood of each codeword (all words within distance <= radius)
        def get_neighborhood(word, q, n, radius):
            """Get all words within Hamming distance <= radius from word."""
            neighbors = set()
            # For radius 1, we can enumerate efficiently
            if radius == 1:
                neighbors.add(tuple(word))
                for pos in range(n):
                    orig = word[pos]
                    for val in range(q):
                        if val != orig:
                            new_word = list(word)
                            new_word[pos] = val
                            neighbors.add(tuple(new_word))
            else:
                # General case using itertools
                from itertools import product, combinations
                for d in range(radius + 1):
                    for positions in combinations(range(n), d):
                        for values in product(range(q), repeat=d):
                            new_word = list(word)
                            for pos, val in zip(positions, values):
                                new_word[pos] = val
                            # Only add if actually at distance d (not less)
                            dist = sum(1 for i in range(n) if new_word[i] != word[i])
                            if dist == d:
                                neighbors.add(tuple(new_word))
            return neighbors
        
        # For efficiency with n=6, q=3, radius=1, total space is 729
        # Precompute all possible words
        all_words = set(itertools.product(range(q), repeat=n))
        
        # Greedy set cover: repeatedly pick the codeword that covers the most uncovered words
        uncovered = set(all_words)
        selected = []
        
        while uncovered and time.time() < deadline - 5.0:
            best_codeword = None
            best_count = -1
            
            # For each candidate codeword, count how many uncovered words it covers
            for cw in code_tuples:
                if time.time() > deadline - 5.0:
                    break
                neighbors = get_neighborhood(cw, q, n, radius)
                count = sum(1 for w in neighbors if w in uncovered)
                if count > best_count:
                    best_count = count
                    best_codeword = cw
            
            if best_codeword is None or best_count <= 0:
                break
            
            selected.append(list(best_codeword))
            neighbors = get_neighborhood(best_codeword, q, n, radius)
            uncovered -= neighbors
            
            # Remove selected codeword from candidates to avoid duplicates
            if best_codeword in code_tuples:
                code_tuples.remove(best_codeword)
        
        # If we still have uncovered words, add more codewords from the original set
        if uncovered:
            for cw in base_code:
                if time.time() > deadline - 2.0:
                    break
                if tuple(cw) not in [tuple(s) for s in selected]:
                    selected.append(cw)
        
        return selected
    else:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
