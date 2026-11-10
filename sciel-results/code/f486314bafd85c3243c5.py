"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1: use greedy set cover approach
    # Start with the direct sum construction (9 * 3^3 = 243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Build coverage map: for each word, which codewords cover it
    # We'll use a greedy approach: repeatedly pick the codeword that covers the most uncovered words
    
    # Represent words as integers for faster hashing
    def word_to_int(word):
        result = 0
        for w in word:
            result = result * q + w
        return result
    
    def int_to_word(val):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        word.reverse()
        return word
    
    # Precompute neighbors: for each word, list of words within Hamming distance radius
    # For radius=1, this is the word itself plus all single-position changes
    def get_neighbors(word):
        neighbors = [word]
        for i in range(n):
            for new_val in range(q):
                if new_val != word[i]:
                    new_word = word.copy()
                    new_word[i] = new_val
                    neighbors.append(new_word)
        return neighbors
    
    # Convert all words to integers
    all_words = set(range(q ** n))
    
    # Precompute neighbor sets for all possible codewords (only need for candidates)
    # But we'll compute on the fly for efficiency
    
    # Start with initial code
    code = list(initial_code)
    code_ints = set(word_to_int(c) for c in code)
    
    # Track which words are covered
    covered = set()
    for c in code:
        for neighbor in get_neighbors(c):
            covered.add(word_to_int(neighbor))
    
    # Greedy: try to remove codewords that don't add unique coverage
    # Actually, let's try a different approach: start from scratch with greedy
    
    # Better approach: greedy set cover
    # 1. Start with empty code
    # 2. Repeatedly add the word that covers the most uncovered words
    # 3. Then try to remove words that are redundant
    
    # But computing coverage for all 2187 words is expensive.
    # Let's use a smarter approach: start with the direct sum and try to prune
    
    # Pruning: remove a codeword if all words it uniquely covers are also covered by other codewords
    # We'll iterate until no more removals are possible
    
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # For each codeword, check if it's redundant
        redundant = []
        for c in code:
            c_int = word_to_int(c)
            # Get all words covered by this codeword
            c_neighbors = [word_to_int(nb) for nb in get_neighbors(c)]
            
            # Check if all these words are covered by other codewords
            is_redundant = True
            for w in c_neighbors:
                # Check if w is covered by any other codeword
                found = False
                for other in code:
                    if other == c:
                        continue
                    if w in [word_to_int(nb) for nb in get_neighbors(other)]:
                        found = True
                        break
                if not found:
                    is_redundant = False
                    break
            
            if is_redundant:
                redundant.append(c)
        
        # Remove all redundant codewords
        if redundant:
            for c in redundant:
                code.remove(c)
            # Recompute covered set
            covered = set()
            for c in code:
                for neighbor in get_neighbors(c):
                    covered.add(word_to_int(neighbor))
        
        if not redundant:
            break
    
    # Try to add better codewords
    # For each uncovered word (if any), try to find a codeword that covers it and many others
    # But we should already have full coverage from the initial construction
    
    # Verify coverage
    if covered != all_words:
        # Add missing words' neighbors as needed
        for w in all_words - covered:
            word = int_to_word(w)
            code.append(word)
            covered.add(w)
            for neighbor in get_neighbors(word):
                covered.add(word_to_int(neighbor))
    
    return code
# EVOLVE-BLOCK-END
