"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Special case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _construct_3_6_1(start_time, deadline)
    
    # General fallback: direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]


def _construct_3_6_1(start_time, deadline):
    q, n, radius = 3, 6, 1
    total_words = 3 ** n  # 729
    full_mask = (1 << total_words) - 1
    
    # Precompute coverage masks for each word
    # mask[i] is an integer where bit j is 1 if word j is covered by word i
    masks = [0] * total_words
    for i in range(total_words):
        # Convert i to word to find neighbors
        w = [0] * n
        temp = i
        for k in range(n - 1, -1, -1):
            w[k] = temp % 3
            temp //= 3
        
        mask = 0
        # Cover self
        mask |= (1 << i)
        # Cover neighbors at distance 1
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[pos] = new_val
                # Convert back to int
                j = 0
                for k in range(n):
                    j = j * 3 + w[k]
                mask |= (1 << j)
            w[pos] = orig
        masks[i] = mask

    def word_to_int(w):
        result = 0
        for i, digit in enumerate(w):
            result = result * 3 + digit
        return result

    def int_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # 1. Greedy Initialization
    # We want to pick words that cover the most currently uncovered bits.
    # uncovered_mask starts as full_mask
    uncovered = full_mask
    codewords = []
    
    while uncovered and time.time() < deadline - 30:
        best_idx = -1
        best_gain = 0
        # Optimization: We can't iterate all 729 words every step if it's slow, 
        # but 729 * (729/64 ops) is small. 
        # To speed up, we can just pick the word with max (masks[i] & uncovered)
        # We'll do a full scan.
        for i in range(total_words):
            gain = bin(masks[i] & uncovered).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if best_gain == bin(uncovered).count('1'):
                    break
        if best_idx == -1:
            break
        codewords.append(best_idx)
        uncovered &= ~masks[best_idx]
        
    # If greedy didn't finish, we have a partial cover. 
    # But for 3^6, greedy usually finishes fast. 
    # If it didn't, we force completion by adding arbitrary words? 
    # Actually, if uncovered is not 0, the greedy loop above would have run out of time or found no gain (which shouldn't happen if uncovered > 0).
    # Let's ensure we have a valid cover. If not, add words until covered.
    while uncovered and time.time() < deadline - 10:
         # Pick any word that covers some uncovered bit
         for i in range(total_words):
             if masks[i] & uncovered:
                 codewords.append(i)
                 uncovered &= ~masks[i]
                 break
         else:
             break # Should not happen

    # 2. Local Search: Minimize size
    # We represent the current set of codeword indices.
    # We maintain the union mask.
    
    current_set = set(codewords)
    current_mask = 0
    for i in current_set:
        current_mask |= masks[i]
        
    # Verify validity
    if current_mask != full_mask:
        # Emergency fix: add words
        while current_mask != full_mask and time.time() < deadline - 5:
             for i in range(total_words):
                 if masks[i] & ~current_mask:
                     current_set.add(i)
                     current_mask |= masks[i]
                     break
             else:
                 break

    best_size = len(current_set)
    best_set = set(current_set)
    
    # Iterative Local Search
    # Strategy: Try to remove a word. If invalid, try to swap it with another word that helps cover the deficit.
    # Or simply: Try removing each word. If valid, keep it. Repeat.
    # Then try swapping: For each word w, and each other word v, if replacing w with v keeps coverage valid, do it (might help future removals).
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Phase 1: Removal
        # Try to remove each codeword
        for w in list(current_set):
            if time.time() > deadline - 5:
                break
            new_mask = 0
            for i in current_set:
                if i == w:
                    continue
                new_mask |= masks[i]
            if new_mask == full_mask:
                current_set.remove(w)
                current_mask = new_mask
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
        
        # Phase 2: Swapping (to escape local minima where no single removal works)
        # For each w in current_set, try replacing with v not in current_set
        # This is O(N^2 * Cost). N~73. 73*729 ~ 50k checks.
        # Check: (current_mask ^ masks[w] | masks[v]) == full_mask?
        # Note: current_mask includes masks[w]. 
        # Let base_mask = current_mask & ~masks[w] (coverage without w)
        # We need base_mask | masks[v] == full_mask
        
        if time.time() < deadline - 5:
            for w in list(current_set):
                if time.time() > deadline - 5:
                    break
                base_mask = current_mask & ~masks[w]
                # Find v such that base_mask | masks[v] == full_mask
                # i.e., masks[v] must cover (full_mask & ~base_mask)
                needed = full_mask & ~base_mask
                if needed == 0:
                    continue # w is redundant, should have been removed in Phase 1
                
                for v in range(total_words):
                    if v in current_set:
                        continue
                    if (masks[v] & needed) == needed:
                        # Replacement is valid
                        current_set.remove(w)
                        current_set.add(v)
                        current_mask = base_mask | masks[v]
                        improved = True
                        if len(current_set) < best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                        break # One swap per w is enough to perturb state
        
    # Convert best_set to list of lists
    result = [int_to_word(idx) for idx in best_set]
    
    # Final sanity check and cleanup
    # Ensure distinct
    unique_result = []
    seen = set()
    for r in result:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique_result.append(r)
            
    return unique_result
# EVOLVE-BLOCK-END
