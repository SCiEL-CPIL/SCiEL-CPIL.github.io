"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
import random

def construct(q, n, radius):
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Generate all words in the space as a flat array
    # For q=3, n=8: 6561 words, manageable
    if q * n == 24 and q == 3:
        # Specialized fast path for the main task
        all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int8)
        total_words = q ** n  # 6561
        num_coords = n  # 8
        radius_val = radius  # 1
        
        # Precompute neighbors for each word (word itself + all single-coordinate changes)
        # For radius 1, q=3: each word has 1 + 8*2 = 17 neighbors
        powers = np.array([q ** (n - 1 - c) for c in range(n)], dtype=np.int32)
        num_neighbors = 1 + num_coords * (q - 1)  # 17
        neighbor_indices = np.zeros((total_words, num_neighbors), dtype=np.int32)
        digits = np.zeros((total_words, n), dtype=np.int8)
        for c in range(n):
            digits[:, c] = (np.arange(total_words) // powers[c]) % q
        
        for i in range(total_words):
            neighbor_indices[i, 0] = i
            pos = 1
            for c in range(n):
                old_val = digits[i, c]
                for v in range(q):
                    if v != old_val:
                        neighbor_indices[i, pos] = i - old_val * powers[c] + v * powers[c]
                        pos += 1
        
        # Start with the direct sum construction (486 words)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code_indices = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                word = h + list(tail)
                # Find the index of this word in all_words
                idx = 0
                for c in range(n):
                    idx = idx * q + word[c]
                code_indices.append(idx)
        
        code_indices = np.array(code_indices, dtype=np.int32)
        
        # Build coverage counts: for each word, how many codewords cover it
        # coverage[word] = number of codewords whose neighbors include word
        coverage = np.zeros(total_words, dtype=np.int32)
        for ci in code_indices:
            coverage[neighbor_indices[ci]] += 1
        
        # Verify initial cover is valid
        assert (coverage > 0).all(), "Initial direct sum does not cover all words"
        
        # Iteratively try to remove codewords
        # A codeword can be removed if all words it covers are covered by at least one other codeword
        # i.e., for all neighbors w of ci, coverage[w] >= 2 (or w is covered by another codeword)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove each codeword in random order
            order = list(range(len(code_indices)))
            random.shuffle(order)
            
            for pos in order:
                if time.time() >= deadline:
                    break
                
                ci = code_indices[pos]
                
                # Check if removing ci would leave any word uncovered
                # For each neighbor w of ci, check if coverage[w] > 1
                # (i.e., some other codeword also covers w)
                neighbors = neighbor_indices[ci]
                can_remove = True
                for w in neighbors:
                    if coverage[w] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove this codeword
                    # Update coverage
                    coverage[neighbors] -= 1
                    # Remove from code_indices
                    code_indices = np.delete(code_indices, pos)
                    improved = True
                    break  # Restart the pass since indices changed
        
        # Additional local search: try swapping codewords to enable more removals
        # For each codeword that can't be removed, try replacing it with a nearby word
        # that might allow the old word's unique neighbors to be covered differently
        for _ in range(3):  # A few passes of swap-based improvement
            if time.time() >= deadline:
                break
            
            improved = False
            order = list(range(len(code_indices)))
            random.shuffle(order)
            
            for pos in order:
                if time.time() >= deadline:
                    break
                
                ci = code_indices[pos]
                neighbors = neighbor_indices[ci]
                
                # Find words that are only covered by this codeword
                unique_words = []
                for w in neighbors:
                    if coverage[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    # Can remove directly
                    coverage[neighbors] -= 1
                    code_indices = np.delete(code_indices, pos)
                    improved = True
                    break
                
                # Try to find a replacement that covers all unique_words
                # and doesn't make things worse
                # The replacement must cover all unique_words
                # Score: number of new words covered (that weren't covered before)
                
                # Candidate replacements: words that cover all unique_words
                # For each unique_word, find codeword candidates that cover it
                # A word x covers unique_word w if w is in neighbors[x]
                # i.e., x is in neighbors[w] (symmetric for radius 1)
                
                # Find intersection of neighbors of all unique_words
                # These are the words that cover all unique_words
                candidates = None
                for uw in unique_words:
                    uw_neighbors = neighbor_indices[uw]  # words that cover uw
                    if candidates is None:
                        candidates = set(uw_neighbors.tolist())
                    else:
                        candidates &= set(uw_neighbors.tolist())
                
                if not candidates:
                    continue
                
                # Remove current codeword temporarily
                coverage[neighbors] -= 1
                
                # For each candidate, check if it's a good replacement
                best_candidate = None
                best_gain = 0
                
                for cand in list(candidates)[:50]:  # Limit to 50 candidates for speed
                    if cand in code_indices:
                        continue
                    
                    cand_neighbors = neighbor_indices[cand]
                    # Gain: how many newly covered words does cand add?
                    # These are words in cand_neighbors that have coverage == 0
                    gain = 0
                    for w in cand_neighbors:
                        if coverage[w] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = cand
                
                if best_candidate is not None and best_gain > 0:
                    # Apply the swap
                    code_indices[pos] = best_candidate
                    coverage[neighbor_indices[best_candidate]] += 1
                    improved = True
                    break
                else:
                    # Revert
                    coverage[neighbors] += 1
        
        # Final verification
        coverage_final = np.zeros(total_words, dtype=np.int32)
        for ci in code_indices:
            coverage_final[neighbor_indices[ci]] += 1
        
        if not (coverage_final > 0).all():
            # Fallback to direct sum
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
        
        # Convert back to list of lists
        result = []
        for ci in code_indices:
            word = all_words[ci].tolist()
            result.append(word)
        
        return result
    else:
        # General fallback: direct sum construction for q=3
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n - 4)]
        else:
            # Very general fallback: just use all words (trivially valid)
            return [list(w) for w in product(range(q), repeat=n)]
# EVOLVE-BLOCK-END
