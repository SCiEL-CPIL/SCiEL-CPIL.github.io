import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Handle specific case q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius, start_time, deadline)

    total_words = q ** n  # 729
    
    # Precompute index to word and word to index
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute masks
    # mask[i] is a bitset of words within distance <= radius of word i
    masks = [0] * total_words
    powers = [1] * n
    for i in range(n-1, 0, -1):
        powers[i-1] = powers[i] * q
        
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = new_val - orig_val
                idx_new = i + diff * powers[pos]
                mask |= (1 << idx_new)
        masks[i] = mask

    full_mask = (1 << total_words) - 1
    
    best_code_size = total_words
    best_code_set = set(range(total_words)) # Fallback
    
    def get_code_size(code_set):
        return len(code_set)

    def is_covered(code_set):
        cov = 0
        for idx in code_set:
            cov |= masks[idx]
        return cov == full_mask

    def reduce_code(code_set):
        """
        Local search to minimize the size of code_set.
        Returns the reduced set.
        """
        changed = True
        while changed:
            changed = False
            # Try to remove
            for c in list(code_set):
                if time.time() > deadline:
                    return code_set
                code_set.remove(c)
                if is_covered(code_set):
                    changed = True
                    break
                else:
                    code_set.add(c)
            
            if time.time() > deadline:
                return code_set

            # If no removal, try swap
            # This is more expensive. We only do it if we have time and haven't found a small code yet.
            # To keep it fast, we might limit swap attempts.
            # But swap is crucial for escaping 73 -> 72.
            
            # Heuristic: Only try swaps if we are stuck at a local minimum.
            # We can try a few random swaps.
            # Actually, let's implement a proper swap search.
            # For each c in code_set:
            #   Remove c
            #   Find best c' to cover gap
            #   Add c'
            #   Try to remove any c''
            #   If success, keep. Else revert.
            
            # This is O(K^2 * N) roughly. K=73, N=729. 73*73*729 ~ 3.8M ops.
            # In Python, this might be too slow if done naively in every iteration.
            # We should optimize or limit.
            
            # Let's try a limited number of swap attempts per "stuck" state.
            # Or just try all c's for swap.
            
            swapped = False
            # To avoid getting stuck in a cycle, we might want to be careful.
            # But if a swap leads to a reduction, it's good.
            
            # Optimization: Precompute "essential" words? No.
            
            for c in list(code_set):
                if time.time() > deadline:
                    return code_set
                
                code_set.remove(c)
                
                # Calculate uncovered
                # We need union of masks of remaining code_set
                # This is O(K).
                union_others = 0
                for c2 in code_set:
                    union_others |= masks[c2]
                
                uncovered = full_mask & ~union_others
                
                if uncovered == 0:
                    # Should have been caught by removal
                    changed = True
                    swapped = True
                    break
                
                # Find best c'
                best_candidate = -1
                best_gain = -1
                # We can restrict candidates to those that cover some uncovered words
                # Iterate through all words? 729 iterations.
                for c_prime in range(total_words):
                    if c_prime in code_set:
                        continue
                    gain = (masks[c_prime] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c_prime
                        
                if best_candidate == -1:
                    code_set.add(c)
                    continue
                
                # Add best_candidate
                code_set.add(best_candidate)
                
                # Try to remove another c''
                found_removal = False
                # We can check if any c'' is redundant now.
                # To speed up, we can just check a few? No, check all.
                for c_double_prime in list(code_set):
                    if c_double_prime == best_candidate:
                        continue
                    code_set.remove(c_double_prime)
                    if is_covered(code_set):
                        found_removal = True
                        swapped = True
                        changed = True
                        break
                    else:
                        code_set.add(c_double_prime)
                
                if found_removal:
                    break
                else:
                    # Revert swap
                    code_set.remove(best_candidate)
                    code_set.add(c)
                    
        return code_set

    # Main loop: Multiple restarts
    # Start with Tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code_set = set()
    for tail_a in range(3):
        for tail_b in range(3):
            for h in tetracode:
                word = h + [tail_a, tail_b]
                idx = word_to_idx(word)
                initial_code_set.add(idx)
                
    # Reduce
    current_code = reduce_code(set(initial_code_set))
    if len(current_code) < best_code_size:
        best_code_size = len(current_code)
        best_code_set = set(current_code)
        
    # If we found 72 or less, we can stop or continue to see if we find 71?
    # 71 is the lower bound.
    
    # If we are still at 73, try random restarts.
    # Generate random covering codes and reduce them.
    
    # To generate a random covering code:
    # 1. Start with empty set.
    # 2. Greedily add words until covered.
    # This yields a code of size ~75-80.
    # Then reduce.
    
    # We can also perturb the current best code.
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        # Strategy 1: Perturb best_code_set
        if best_code_size > 72:
             # Try to mutate best_code_set
             mutated = set(best_code_set)
             # Remove 2-3 random words
             to_remove = random.sample(list(mutated), min(3, len(mutated)))
             for r in to_remove:
                 mutated.remove(r)
             
             # Complete coverage
             covered = 0
             for idx in mutated:
                 covered |= masks[idx]
                 
             # Greedy completion
             while covered != full_mask:
                 best_idx = -1
                 best_gain = -1
                 for i in range(total_words):
                     if i in mutated:
                         continue
                     gain = (masks[i] & ~covered).bit_count()
                     if gain > best_gain:
                         best_gain = gain
                         best_idx = i
                 if best_gain == 0:
                     break
                 mutated.add(best_idx)
                 covered |= masks[best_idx]
                 
             # Reduce
             reduced = reduce_code(mutated)
             if len(reduced) < best_code_size:
                 best_code_size = len(reduced)
                 best_code_set = set(reduced)
                 if best_code_size <= 72:
                     break # Found a better code!
                     
        # Strategy 2: Random Greedy Start
        # Start with a random subset of size ~10? No, too small.
        # Start with random 20 words, then greedy complete.
        random_start = set(random.sample(range(total_words), 20))
        covered = 0
        for idx in random_start:
            covered |= masks[idx]
            
        while covered != full_mask:
            best_idx = -1
            best_gain = -1
            for i in range(total_words):
                if i in random_start:
                    continue
                gain = (masks[i] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            random_start.add(best_idx)
            covered |= masks[best_idx]
            
        reduced = reduce_code(random_start)
        if len(reduced) < best_code_size:
            best_code_size = len(reduced)
            best_code_set = set(reduced)
            if best_code_size <= 72:
                break

    # Final verification
    if not is_covered(best_code_set):
        # Fallback to a valid code (e.g. greedy from scratch or tetracode reduced)
        # Re-run tetracode reduction
        current_code = reduce_code(set(initial_code_set))
        best_code_set = current_code
        best_code_size = len(current_code)

    return [idx_to_word(idx) for idx in best_code_set]

def _generic_construct(q, n, radius, start_time, deadline):
    total_words = q ** n
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute masks
    masks = [0] * total_words
    powers = [1] * n
    for i in range(n-1, 0, -1):
        powers[i-1] = powers[i] * q
        
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                diff = new_val - orig_val
                idx_new = i + diff * powers[pos]
                mask |= (1 << idx_new)
        masks[i] = mask

    full_mask = (1 << total_words) - 1
    covered = 0
    code_set = set()

    while covered != full_mask:
        if time.time() > deadline:
            break
        best_idx = -1
        best_gain = -1
        for i in range(total_words):
            if i in code_set:
                continue
            gain = (masks[i] & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break
        code_set.add(best_idx)
        covered |= masks[best_idx]
        
    return [idx_to_word(i) for i in code_set]

if __name__ == "__main__":
    code = construct(3, 6, 1)
    print(f"Code size: {len(code)}")
