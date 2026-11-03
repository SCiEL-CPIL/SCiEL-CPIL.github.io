import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Only handle the specific case q=3, n=6, radius=1 efficiently
    # For other cases, fall back to a simple greedy or the seed method
    if q != 3 or n != 6 or radius != 1:
        # Fallback: direct sum if possible, else greedy
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Simple greedy for small n
            all_words = list(itertools.product(range(q), repeat=n))
            word_to_idx = {w: i for i, w in enumerate(all_words)}
            uncovered = set(range(len(all_words)))
            selected = []
            while uncovered:
                best_word = None
                best_count = 0
                for w in all_words:
                    # Count how many uncovered words are within distance radius
                    count = 0
                    for u in uncovered:
                        if sum(1 for a, b in zip(w, all_words[u]) if a != b) <= radius:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_word = w
                selected.append(best_word)
                # Update uncovered
                new_uncovered = set()
                for u in uncovered:
                    if sum(1 for a, b in zip(best_word, all_words[u]) if a != b) > radius:
                        new_uncovered.add(u)
                uncovered = new_uncovered
            return selected

    # Specific case: q=3, n=6, radius=1
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    total_words = len(all_words)
    
    # Precompute cover sets for all possible codewords
    # cover_mask[c] is a bitmask of words covered by codeword c
    # Since total_words = 729, we can use Python integers as bitmasks
    cover_masks = [0] * total_words
    for i, c in enumerate(all_words):
        mask = 0
        # The codeword itself
        mask |= (1 << word_to_idx[c])
        # Words at distance 1
        for pos in range(6):
            for val in range(3):
                if val == c[pos]:
                    continue
                new_word = list(c)
                new_word[pos] = val
                new_word = tuple(new_word)
                mask |= (1 << word_to_idx[new_word])
        cover_masks[i] = mask
    
    best_solution = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    while time.time() < deadline:
        uncovered = (1 << total_words) - 1  # All bits set
        selected_indices = []
        
        while uncovered:
            # Find the codeword that covers the most uncovered words
            best_idx = -1
            best_count = 0
            # Randomization: shuffle the order of checking if we want, 
            # but standard greedy with tie-breaking is better.
            # To introduce randomness, we can pick randomly among top candidates.
            
            # Optimization: only check codewords that cover at least one uncovered word
            # But with 729 codewords, checking all is fast enough.
            # We can speed up by noting that if a codeword is already selected, 
            # it might still be useful if it covers new words, but typically 
            # we don't need to re-select. However, in set cover, we just pick 
            # the best from the entire pool each time.
            
            for i in range(total_words):
                # Intersection of cover_masks[i] and uncovered
                # Bitwise AND
                common = cover_masks[i] & uncovered
                # Count bits
                # Python 3.10+ has int.bit_count()
                count = common.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                elif count == best_count and count > 0:
                    # Tie-breaking: random or deterministic
                    # Let's use a simple random tie-break to explore different paths
                    if random.random() < 0.5:
                        best_idx = i
            
            if best_idx == -1:
                break # Should not happen if uncovered is not empty
            
            selected_indices.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
            if time.time() > deadline:
                break
        
        if uncovered == 0:
            size = len(selected_indices)
            if size < best_size:
                best_size = size
                best_solution = [list(all_words[i]) for i in selected_indices]
                
                # If we found 73, which is the best known, we can stop early 
                # or continue to see if we can beat it (unlikely, but possible).
                # Given the time limit, let's keep searching for a bit longer 
                # to ensure we don't miss a 72-code if one exists (though lower bound is 71, 72 is possible).
                # Actually, if we find 73, it's a strong candidate. 
                # Let's run a few more iterations to confirm stability or find better.
                
        if best_size <= 73:
            # We have a very good solution. Let's do a quick local search / pruning.
            # Try to remove codewords from best_solution
            sol_indices = set(selected_indices) # Note: selected_indices is from the last run that achieved best_size? 
            # Wait, best_solution was updated, but selected_indices is local to the loop.
            # We need to store the indices or re-derive them.
            # Let's store the indices in best_solution_indices.
            
            # Refine: We should store the indices.
            # Let's modify the loop to store indices.
            
            pass # Handled below

    # The loop above has a slight issue: best_solution is a list of lists, 
    # but we lose the indices. Let's store indices.
    
    # Re-implementing the loop with index storage
    best_solution_indices = None
    best_size = float('inf')
    
    # Reset for the second pass with proper storage
    while time.time() < deadline:
        uncovered = (1 << total_words) - 1
        selected_indices = []
        
        while uncovered:
            best_idx = -1
            best_count = 0
            
            for i in range(total_words):
                common = cover_masks[i] & uncovered
                count = common.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                elif count == best_count and count > 0:
                    if random.random() < 0.5:
                        best_idx = i
            
            if best_idx == -1:
                break
            
            selected_indices.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
            if time.time() > deadline:
                break
        
        if uncovered == 0:
            size = len(selected_indices)
            if size < best_size:
                best_size = size
                best_solution_indices = selected_indices[:]
                
                # Optional: Local search / Pruning
                # Try to remove any codeword that is redundant
                # This can be done by checking if the union of the remaining covers all words.
                # Since size is small (~73), checking redundancy is O(size * total_words) which is fast.
                pruned = True
                while pruned:
                    pruned = False
                    for i in range(len(best_solution_indices)):
                        # Temporarily remove best_solution_indices[i]
                        remaining = best_solution_indices[:i] + best_solution_indices[i+1:]
                        # Check if remaining covers all
                        union_mask = 0
                        for idx in remaining:
                            union_mask |= cover_masks[idx]
                        if union_mask == (1 << total_words) - 1:
                            # It is redundant, remove it
                            best_solution_indices.pop(i)
                            best_size -= 1
                            pruned = True
                            break # Restart the while loop since indices changed
                    
                    if time.time() > deadline:
                        break

    if best_solution_indices is not None:
        return [list(all_words[i]) for i in best_solution_indices]
    
    # Fallback if no valid solution found (should not happen)
    return [list(all_words[i]) for i in range(73)] # Arbitrary fallback

