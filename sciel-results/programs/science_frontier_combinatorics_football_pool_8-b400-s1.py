# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    # Precompute all neighbors using vectorized approach
    # For radius=1, each word has 1 + n*(q-1) neighbors
    # We'll store as a list of arrays for fast access
    n_neighbors = 1 + n * (q - 1)
    
    # Build neighbor table
    # neighbors[w] = array of neighbor indices
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    # Convert all words to digit arrays at once
    all_words = np.zeros((total_words, n), dtype=np.int32)
    for i in range(total_words):
        val = i
        for pos in range(n - 1, -1, -1):
            all_words[i, pos] = val % q
            val //= q
    
    # For each word, compute its neighbors
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            orig = all_words[w, pos]
            for v in range(q):
                if v == orig:
                    continue
                # Compute new word index
                # The position pos contributes (v - orig) * q^(n-1-pos)
                new_val = w + (v - orig) * (q ** (n - 1 - pos))
                neighbors[w, idx] = new_val
                idx += 1
    
    # Convert to list of lists for easier manipulation
    neighbor_list = [neighbors[w].tolist() for w in range(total_words)]
    
    # Also create a set for each word for O(1) membership in swap validation
    neighbor_sets = [set(neighbors[w]) for w in range(total_words)]
    
    # Phase 1: Greedy set cover initialization
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Use vectorized gain computation
    # For each candidate, gain = sum of uncovered neighbors
    # We can compute this efficiently
    
    remaining = total_words
    while remaining > 0:
        if time.time() > deadline - 10.0:
            break
        
        # Compute gain for all words at once
        # gain[w] = number of uncovered words in neighbors[w]
        gains = np.zeros(total_words, dtype=np.int32)
        for w in range(total_words):
            if covered[w]:
                continue
            # Count uncovered neighbors
            nb = neighbors[w]
            gains[w] = np.sum(~covered[nb])
        
        # Pick the word with maximum gain
        best_word = np.argmax(gains)
        best_gain = gains[best_word]
        
        if best_gain == 0:
            # Should not happen if remaining > 0, but just in case
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            best_word = uncovered[0]
        
        codewords.append(best_word)
        # Update covered
        for u in neighbors[best_word]:
            if not covered[u]:
                covered[u] = True
                remaining -= 1
    
    # Phase 2: Local search to minimize
    cw = codewords[:]  # Working list
    best_cw = cw[:]
    best_size = len(cw)
    
    # Compute coverage counts
    count = np.zeros(total_words, dtype=np.int32)
    for ci in cw:
        for u in neighbors[ci]:
            count[u] += 1
    
    # Local search loop
    while time.time() < deadline - 3.0:
        improved = False
        
        # Phase 2a: Remove redundant codewords
        # A codeword is redundant if all its neighbors are covered by other codewords
        # i.e., count[u] > 1 for all u in neighbors[ci]
        
        # We'll iterate in random order and remove redundancies
        indices = list(range(len(cw)))
        random.shuffle(indices)
        
        removed_this_pass = 0
        for i in indices:
            if i >= len(cw):
                continue
            ci = cw[i]
            
            # Check if redundant: all neighbors have count > 1
            nb = neighbors[ci]
            if np.all(count[nb] > 1):
                # Remove ci
                for u in nb:
                    count[u] -= 1
                cw.pop(i)
                removed_this_pass += 1
                improved = True
                
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
        
        # Phase 2b: Try swaps to enable more removals
        if not improved and len(cw) > 0:
            # Try random swaps
            swaps_tried = 0
            max_swaps = 50
            
            while swaps_tried < max_swaps and time.time() < deadline - 3.0:
                swaps_tried += 1
                
                i = random.randint(0, len(cw) - 1)
                ci = cw[i]
                
                # Find words uniquely covered by ci (count == 1)
                nb_ci = neighbors[ci]
                critical = [u for u in nb_ci if count[u] == 1]
                
                if not critical:
                    # ci is redundant, remove it
                    for u in nb_ci:
                        count[u] -= 1
                    cw.pop(i)
                    improved = True
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]
                    break
                
                # Try to find a replacement cj that covers all critical words
                # Sample from neighbors of critical words
                candidates = set()
                for u in critical:
                    for v in neighbors[u]:
                        if v != ci:
                            candidates.add(v)
                
                if not candidates:
                    continue
                
                # Try a few random candidates
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                found_swap = False
                for cj in cand_list[:20]:  # Try up to 20 candidates
                    # Check if cj covers all critical words
                    nb_cj = neighbors[cj]
                    valid = all(u in neighbor_sets[cj] for u in critical)
                    
                    if valid:
                        # Perform swap
                        for u in nb_ci:
                            count[u] -= 1
                        for u in nb_cj:
                            count[u] += 1
                        cw[i] = cj
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
        
        # If no improvement in this pass, try a different strategy
        if not improved:
            # Try to find a codeword that can be replaced by one that enables removal
            # Pick a codeword with many "wasted" coverage (neighbors with high count)
            if len(cw) > 0:
                # Score each codeword by how much it "wastes" coverage
                scores = []
                for i, ci in enumerate(cw):
                    nb = neighbors[ci]
                    # Waste = sum of (count[u] - 1) for u in nb, i.e., how many extra covers it provides
                    waste = np.sum(count[nb] - 1)
                    scores.append((waste, i))
                
                # Try replacing the most wasteful codeword
                scores.sort(reverse=True)
                for waste, i in scores[:10]:
                    if waste <= 0:
                        break
                    ci = cw[i]
                    nb_ci = neighbors[ci]
                    critical = [u for u in nb_ci if count[u] == 1]
                    
                    if not critical:
                        # Redundant, remove
                        for u in nb_ci:
                            count[u] -= 1
                        cw.pop(i)
                        improved = True
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
                        break
                    
                    # Try to find replacement
                    candidates = set()
                    for u in critical:
                        for v in neighbors[u]:
                            if v != ci:
                                candidates.add(v)
                    
                    if not candidates:
                        continue
                    
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    
                    found_swap = False
                    for cj in cand_list[:30]:
                        valid = all(u in neighbor_sets[cj] for u in critical)
                        if valid:
                            # Also check that this swap doesn't create new problems
                            # (i.e., doesn't leave any word uncovered)
                            # Since cj covers all critical words of ci, and ci's other neighbors
                            # have count > 1, the swap is valid.
                            for u in nb_ci:
                                count[u] -= 1
                            nb_cj = neighbors[cj]
                            for u in nb_cj:
                                count[u] += 1
                            cw[i] = cj
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        break
    
    # Convert back to list of lists
    result = []
    for ci in best_cw:
        word = []
        val = ci
        for pos in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        word.reverse()
        result.append(word)
    
    return result

# EVOLVE-BLOCK-END
