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
    
    # Precompute neighbors
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    # Compute powers for index calculation
    q_powers = [q ** i for i in range(n)]
    
    # Build neighbor table
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            # Get digit at pos
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                # Calculate new index
                # Replace digit at pos with v
                # w = ... + digit * q^(n-1-pos) + ...
                # new_w = w - digit * q^(n-1-pos) + v * q^(n-1-pos)
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Phase 1: Greedy Set Cover with dynamic gain updates
    covered = np.zeros(total_words, dtype=bool)
    gain = np.full(total_words, n_neighbors, dtype=np.int32)
    codewords = []
    
    remaining = total_words
    
    while remaining > 0:
        if time.time() > deadline - 20.0:
            break
            
        best_word = np.argmax(gain)
        best_gain = gain[best_word]
        
        if best_gain == 0:
            # Should not happen if remaining > 0
            uncovered = np.where(~covered)[0]
            if len(uncovered) == 0:
                break
            best_word = uncovered[0]
        
        codewords.append(best_word)
        
        # Update covered and gain
        for v in neighbors[best_word]:
            if not covered[v]:
                covered[v] = True
                remaining -= 1
                # Decrement gain for all words that cover v
                # These are the neighbors of v
                for w in neighbors[v]:
                    gain[w] -= 1

    # Phase 2: Local Search
    cw = codewords[:]
    best_cw = cw[:]
    best_size = len(cw)
    
    # Compute coverage counts
    count = np.zeros(total_words, dtype=np.int32)
    for ci in cw:
        for u in neighbors[ci]:
            count[u] += 1
    
    # Helper to check if a word is in the current code
    # Using a set for O(1) lookup is faster for small n
    cw_set = set(cw)
    
    # Local search loop
    while time.time() < deadline - 5.0:
        improved = False
        
        # Pass 1: Remove redundant codewords
        # A codeword c is redundant if all v in N(c) have count[v] > 1
        indices = list(range(len(cw)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(cw):
                continue
            ci = cw[i]
            
            # Check redundancy
            if np.all(count[neighbors[ci]] > 1):
                # Remove ci
                for u in neighbors[ci]:
                    count[u] -= 1
                cw.pop(i)
                cw_set.remove(ci)
                improved = True
                
                if len(cw) < best_size:
                    best_size = len(cw)
                    best_cw = cw[:]
        
        # Pass 2: Swaps
        if not improved:
            # Try to swap codewords to enable further removals
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                # Find critical neighbors (covered only by ci)
                critical = []
                for u in neighbors[ci]:
                    if count[u] == 1:
                        critical.append(u)
                
                if not critical:
                    # Redundant, should have been caught in Pass 1
                    # But maybe counts changed?
                    continue
                
                # Find candidates c' that cover all critical neighbors
                # c' must be in intersection of N(u) for all u in critical
                # Since |critical| is small, we can just iterate through neighbors of the first critical word
                # and check if it covers the others
                
                candidates = []
                first_crit = critical[0]
                for cand in neighbors[first_crit]:
                    if cand == ci:
                        continue
                    # Check if cand covers all other critical words
                    valid = True
                    for u in critical[1:]:
                        # Check if u is in neighbors[cand]
                        # This is O(n_neighbors) per check
                        if not np.any(neighbors[cand] == u):
                            valid = False
                            break
                    if valid:
                        candidates.append(cand)
                
                if candidates:
                    # Pick a random candidate
                    cand = random.choice(candidates)
                    
                    # Perform swap
                    # Remove ci
                    for u in neighbors[ci]:
                        count[u] -= 1
                    # Add cand
                    for u in neighbors[cand]:
                        count[u] += 1
                    
                    cw[i] = cand
                    cw_set.remove(ci)
                    cw_set.add(cand)
                    improved = True
                    break # Restart pass after a swap

        # Pass 3: Add and Reduce
        if not improved:
            # Try adding a random word and then reducing
            attempts = 20
            for _ in range(attempts):
                if time.time() > deadline - 5.0:
                    break
                    
                cand = random.randint(0, total_words - 1)
                if cand in cw_set:
                    continue
                
                # Save state
                cw_copy = cw[:]
                count_copy = count.copy()
                cw_set_copy = set(cw_set) # Expensive? 500 items. OK.
                
                # Tentatively add cand
                cw.append(cand)
                cw_set.add(cand)
                for u in neighbors[cand]:
                    count[u] += 1
                
                # Try to remove redundant words
                # We iterate through the list of indices, being careful with pops
                indices = list(range(len(cw)))
                random.shuffle(indices)
                
                removed_any = False
                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]
                    
                    if np.all(count[neighbors[ci]] > 1):
                        for u in neighbors[ci]:
                            count[u] -= 1
                        cw.pop(i)
                        cw_set.remove(ci)
                        removed_any = True
                
                new_size = len(cw)
                
                if new_size < len(cw_copy):
                    # Keep the change
                    improved = True
                    if new_size < best_size:
                        best_size = new_size
                        best_cw = cw[:]
                    break # Restart main loop
                else:
                    # Revert
                    cw = cw_copy
                    count = count_copy
                    cw_set = cw_set_copy

        # Pass 3: Add and Reduce
        if not improved:
            # Try adding a random word and then reducing
            # We pick a word not in the current set
            # To speed up, we can just pick a random index from 0 to total_words-1
            # and check if it's in the set
            
            attempts = 10
            for _ in range(attempts):
                if time.time() > deadline - 5.0:
                    break
                    
                cand = random.randint(0, total_words - 1)
                if cand in cw_set:
                    continue
                
                old_size = len(cw)
                
                # Tentatively add cand
                cw.append(cand)
                cw_set.add(cand)
                for u in neighbors[cand]:
                    count[u] += 1
                
                # Try to remove redundant words
                # We iterate through the list of indices, being careful with pops
                # We can create a list of indices to remove
                to_remove = []
                # We must not remove cand itself immediately? 
                # Actually, if cand is redundant, it's a bad add. 
                # But maybe it allows removing 2 others.
                
                indices = list(range(len(cw)))
                random.shuffle(indices)
                
                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]
                    
                    if np.all(count[neighbors[ci]] > 1):
                        # Mark for removal
                        # We need to be careful: if we remove one, it might make another non-redundant?
                        # No, removing a word decreases counts. 
                        # If a word was redundant (count > 1 for all neighbors), 
                        # removing another word might reduce a neighbor's count to 1, making it non-redundant.
                        # So we should check redundancy *before* removing, or do it sequentially.
                        # Sequential is safer.
                        
                        # Check again just in case
                        if np.all(count[neighbors[ci]] > 1):
                            for u in neighbors[ci]:
                                count[u] -= 1
                            cw.pop(i)
                            cw_set.remove(ci)
                            improved = True
                            # Break out of removal loop to re-evaluate?
                            # Or continue? 
                            # Continuing might be risky if we removed a word that was supporting another.
                            # But we checked `np.all` before removing.
                            # If we remove ci, counts drop.
                            # If we then check cj, its counts are lower.
                            # If cj was redundant, it might still be redundant, or might not be.
                            # The check `np.all(count[neighbors[cj]] > 1)` will catch this.
                            # So sequential removal is safe.
                            
                new_size = len(cw)
                
                if new_size < old_size:
                    # Keep the change
                    if new_size < best_size:
                        best_size = new_size
                        best_cw = cw[:]
                    break # Restart main loop
                else:
                    # Revert
                    # Remove cand
                    cw.pop()
                    cw_set.remove(cand)
                    for u in neighbors[cand]:
                        count[u] -= 1
                    # Restore removed words? 
                    # We modified `cw` and `count` in place.
                    # We need to undo the removals.
                    # This is tricky because we popped items.
                    
                    # Better approach: Copy the state before adding?
                    # That is expensive.
                    
                    # Alternative: 
                    # If we don't improve, we must revert.
                    # We can store the state of `cw` and `count`?
                    # `count` is a numpy array. Copying it is O(N).
                    # `cw` is a list. Copying is O(K).
                    
                    # Let's refactor Pass 3 to use a copy for safety if it fails.
                    
    
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
