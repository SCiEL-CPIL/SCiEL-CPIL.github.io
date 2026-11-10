"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys

def construct(q, n, radius):
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: use all words (worst case)
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # For q=3, n=7, radius=1
    # Use a local search approach: start with a good initial code, then try swaps to reduce size
    
    # Helper functions
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
    
    # Precompute all words as integers
    total_words = q ** n  # 2187
    
    # Precompute neighbor sets for efficiency
    # For radius=1, neighbors are the word itself plus all single-position changes
    # We'll compute coverage counts incrementally
    
    # Initial code: direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code = list(initial_code)
    code_ints = [word_to_int(c) for c in code]
    code_set = set(code_ints)
    
    # Compute coverage: for each word in the space, how many codewords cover it
    # covered_count[w] = number of codewords c such that dist(c, w) <= radius
    covered_count = [0] * total_words
    
    # Helper to get neighbors of an integer word
    def get_neighbor_ints(w_int):
        """Return list of integer values of words within Hamming distance 1 of w_int."""
        w = int_to_word(w_int)
        neighbors = [w_int]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val != orig:
                    w[i] = new_val
                    neighbors.append(word_to_int(w))
            w[i] = orig
        return neighbors
    
    # Build coverage counts
    for c_int in code_ints:
        for nb in get_neighbor_ints(c_int):
            covered_count[nb] += 1
    
    # Check if fully covered
    def is_fully_covered():
        return all(c > 0 for c in covered_count)
    
    # Try to remove codewords that are redundant (all their covered words are covered by others)
    def try_remove_redundant():
        """Try to remove redundant codewords. Returns True if any were removed."""
        removed = False
        for idx in range(len(code)):
            if time.time() > deadline - 20:
                break
            c_int = code_ints[idx]
            # Check if removing this codeword would leave any word uncovered
            neighbors = get_neighbor_ints(c_int)
            can_remove = True
            for nb in neighbors:
                if covered_count[nb] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove this codeword
                for nb in neighbors:
                    covered_count[nb] -= 1
                # Remove from lists
                code.pop(idx)
                code_ints.pop(idx)
                code_set.remove(c_int)
                removed = True
                # Restart scanning since indices changed
                return True
        return removed
    
    # Try swap: remove one codeword, add another that covers more uncovered words
    def try_swap():
        """Try to find a beneficial swap. Returns True if a swap was made."""
        if time.time() > deadline - 20:
            return False
        
        # Find words that are only covered by one codeword (critical words)
        # For a swap to be valid, we need to replace a codeword with one that covers all its critical words
        
        # Strategy: for each codeword, check if there's another word that covers all the same words it covers
        # This is expensive, so let's be smarter.
        
        # Alternative: try to replace a codeword with a random word that covers at least as many unique words
        # Actually, let's try a different approach: hill climbing on the code size
        
        # For now, let's just try random swaps and see if we can reduce size
        # This is a simple local search
        
        # Pick a random codeword to remove
        idx = random.randint(0, len(code) - 1)
        c_int = code_ints[idx]
        c_neighbors = get_neighbor_ints(c_int)
        
        # Check if we can remove it
        can_remove = all(covered_count[nb] > 1 for nb in c_neighbors)
        if not can_remove:
            return False
        
        # Temporarily remove
        for nb in c_neighbors:
            covered_count[nb] -= 1
        
        # Now find a word that covers the most uncovered words (words with count 0)
        # But we need to be efficient. Let's sample candidate positions.
        
        best_gain = 0
        best_pos = None
        
        # Sample some random positions to try
        samples = 50
        for _ in range(samples):
            # Generate a random word
            rand_word = [random.randint(0, q - 1) for _ in range(n)]
            rand_int = word_to_int(rand_word)
            if rand_int in code_set:
                continue
            
            # Compute how many currently uncovered words this would cover
            gain = 0
            for nb in get_neighbor_ints(rand_int):
                if covered_count[nb] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_pos = (rand_word, rand_int)
        
        # If we found a good position, add it
        if best_pos is not None and best_gain > 0:
            new_word, new_int = best_pos
            for nb in get_neighbor_ints(new_int):
                covered_count[nb] += 1
            code.append(new_word)
            code_ints.append(new_int)
            code_set.add(new_int)
            # Remove the old one
            code.pop(idx)
            code_ints.pop(idx)
            code_set.remove(c_int)
            return True
        
        # Roll back
        for nb in c_neighbors:
            covered_count[nb] += 1
        
        return False
    
    # Main loop: iterate removal and swap attempts
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline - 30:
            break
        
        progress = False
        
        # Try to remove redundant codewords
        while try_remove_redundant():
            progress = True
        
        # Try swaps
        if not progress:
            if try_swap():
                progress = True
            else:
                # Try a few more swaps
                for _ in range(10):
                    if time.time() > deadline - 30:
                        break
                    if try_swap():
                        progress = True
                        break
        
        if not progress:
            break
    
    # Final verification and repair
    # Check if fully covered
    if not is_fully_covered():
        # Add codewords for uncovered words
        for w in range(total_words):
            if covered_count[w] == 0:
                word = int_to_word(w)
                if word_to_int(word) not in code_set:
                    code.append(word)
                    code_ints.append(word_to_int(word))
                    code_set.add(word_to_int(word))
                    for nb in get_neighbor_ints(word_to_int(word)):
                        covered_count[nb] += 1
    
    # One more pass of removal
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        if not try_remove_redundant():
            break
    
    return code
# EVOLVE-BLOCK-END
