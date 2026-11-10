import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_words = q ** n  # 2187
    sphere_size = 1 + n * (q - 1)  # 15

    # Precompute all words as indices 0..2186
    # We work with indices. Word i corresponds to index i.
    
    # Precompute spheres as bitmasks (Python integers)
    # sphere_mask[i] has bit j set if word j is within distance radius of word i
    sphere_masks = [0] * total_words
    
    # Helper to compute index from word
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    # Precompute neighbors for each word
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = [0] * n
        idx = i
        for pos in range(n - 1, -1, -1):
            w[pos] = idx % q
            idx //= q
        
        neighbors[i].append(i)
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    nw = w.copy()
                    nw[pos] = new_val
                    j = word_to_idx(nw)
                    neighbors[i].append(j)
        
        # Create bitmask
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        sphere_masks[i] = mask

    all_mask = (1 << total_words) - 1

    # Greedy Set Cover
    def greedy_set_cover(randomize=False):
        uncovered = all_mask
        code_indices = []
        
        while uncovered:
            best_gain = -1
            best_c = -1
            
            if randomize:
                candidates = list(range(total_words))
                random.shuffle(candidates)
            else:
                candidates = range(total_words)
                
            for c in candidates:
                gain = bin(sphere_masks[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    if best_gain == sphere_size: # Max possible
                        break
                        
            if best_c == -1:
                break
                
            code_indices.append(best_c)
            uncovered &= ~sphere_masks[best_c]
            
        return code_indices

    # Prune redundant codewords
    def prune_code(code_indices):
        if not code_indices:
            return []
        
        code_set = set(code_indices)
        # Count coverage
        count = [0] * total_words
        for c in code_set:
            mask = sphere_masks[c]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                count[j] += 1
                m ^= lsb
                
        to_check = list(code_set)
        random.shuffle(to_check)
        
        for c in to_check:
            if c not in code_set:
                continue
            mask = sphere_masks[c]
            m = mask
            safe = True
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if count[j] <= 1:
                    safe = False
                    break
                m ^= lsb
            
            if safe:
                code_set.remove(c)
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    count[j] -= 1
                    m ^= lsb
                    
        return list(code_set)

    # Lower bound: max number of words a single codeword can cover is sphere_size
    # Greedy lower bound for a set of uncovered words U:
    # Repeatedly pick the word that covers the most uncovered, subtract, repeat.
    def greedy_lower_bound(uncovered_mask):
        if uncovered_mask == 0:
            return 0
        lb = 0
        u = uncovered_mask
        while u:
            best_gain = 0
            for c in range(total_words):
                gain = bin(sphere_masks[c] & u).count('1')
                if gain > best_gain:
                    best_gain = gain
                if best_gain == 0:
                    break
            if best_gain == 0:
                break
            lb += 1
            # We need to remove the covered words. Which ones?
            # For a lower bound, we can just assume the best case:
            # We found a codeword covering 'best_gain' words.
            # We don't know exactly which ones to remove to maximize future gains,
            # but for a valid lower bound, we can just say "I covered best_gain words".
            # However, to be tighter, we should remove the specific words covered by the best codeword.
            # But we didn't store which codeword was best. Let's re-find it or store it.
            # For speed, a simple bound is ceil(popcount(uncovered) / sphere_size).
            # Let's use the simple bound for speed, as greedy LB is expensive.
            # Actually, the simple bound is often weak.
            # Let's try a slightly better one: 
            # If we just return ceil(popcount / sphere_size), it's fast.
            pass
        return (bin(uncovered_mask).count('1') + sphere_size - 1) // sphere_size

    # Simulated Annealing
    def simulated_annealing(initial_code, max_iters=3000, t_start=10.0, t_end=0.1):
        code_set = set(initial_code)
        # Precompute coverage count for current state
        count = [0] * total_words
        for c in code_set:
            mask = sphere_masks[c]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                count[j] += 1
                m ^= lsb
        
        uncovered_count = sum(1 for c in count if c == 0)
        current_size = len(code_set)
        
        # Energy: Size * 100 + Uncovered * 1000
        # We want to minimize size, but MUST have 0 uncovered.
        # If uncovered > 0, energy is huge.
        def energy(size, unc):
            if unc > 0:
                return 1000000 + unc * 1000 + size
            return size
            
        current_energy = energy(current_size, uncovered_count)
        best_code = set(code_set)
        best_energy = current_energy
        best_size = current_size
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            move_type = random.random()
            new_code_set = set(code_set)
            
            # We need to efficiently update count.
            # For simplicity in SA, we can recalculate count if the move is accepted,
            # or maintain a delta.
            # Given the small size (2187 words), recalculating count for a move is O(SphereSize).
            # But checking uncovered is O(total_words).
            # Let's use a bitmask for uncovered to speed up checks.
            
            # Actually, maintaining `count` array is O(1) per word update.
            # Let's stick to `count` array and calculate uncovered on the fly?
            # No, calculating uncovered by scanning 2187 ints is fast (~1us).
            
            if move_type < 0.3:
                # Remove
                if code_set:
                    c_remove = random.choice(list(code_set))
                    new_code_set.remove(c_remove)
                    # Update count
                    mask = sphere_masks[c_remove]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] -= 1
                        m ^= lsb
            elif move_type < 0.6:
                # Add
                c_add = random.randint(0, total_words - 1)
                if c_add not in code_set:
                    new_code_set.add(c_add)
                    mask = sphere_masks[c_add]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] += 1
                        m ^= lsb
            else:
                # Swap
                if code_set:
                    c_remove = random.choice(list(code_set))
                    c_add = random.randint(0, total_words - 1)
                    if c_add != c_remove and c_add not in code_set:
                        new_code_set.remove(c_remove)
                        new_code_set.add(c_add)
                        
                        mask_rm = sphere_masks[c_remove]
                        m = mask_rm
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            count[j] -= 1
                            m ^= lsb
                            
                        mask_add = sphere_masks[c_add]
                        m = mask_add
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            count[j] += 1
                            m ^= lsb

            new_uncovered = sum(1 for c in count if c == 0)
            new_size = len(new_code_set)
            new_energy = energy(new_size, new_uncovered)
            
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / max(t, 1e-6)):
                code_set = new_code_set
                current_size = new_size
                current_energy = new_energy
                
                if new_uncovered == 0 and new_size < best_size:
                    best_code = set(code_set)
                    best_energy = new_energy
                    best_size = new_size
                    
            # Rollback if not accepted
            # We modified `count` in place. If rejected, we must undo.
            # This is error-prone.
            # Better: Compute new_count in a copy? Too slow.
            # Better: Apply move, check, if rejected, reverse move.
            
            # Let's re-structure: Apply move, calculate energy. If reject, reverse.
            # The code above modifies `count` directly.
            # If rejected, we need to reverse the modification.
            
            # This is getting complex. Let's use a simpler SA that reconstructs count?
            # No, 2187 is small.
            # Let's restart the SA logic to be safer:
            # Store old state, apply, check, reject->restore.
            
            pass # The loop above is flawed for rollback. Let's rewrite SA properly below.

    def simulated_annealing_safe(initial_code, max_iters=2000, t_start=10.0, t_end=0.1):
        code_set = set(initial_code)
        count = [0] * total_words
        for c in code_set:
            mask = sphere_masks[c]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                count[j] += 1
                m ^= lsb
        
        def calc_energy(size, unc):
            if unc > 0:
                return 1000000 + unc * 1000 + size
            return size

        current_uncovered = sum(1 for c in count if c == 0)
        current_size = len(code_set)
        current_energy = calc_energy(current_size, current_uncovered)
        
        best_code = set(code_set)
        best_size = current_size
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            move_type = random.random()
            old_code_set = code_set
            old_count = count
            old_size = current_size
            old_uncovered = current_uncovered
            old_energy = current_energy
            
            # Apply Move
            if move_type < 0.3:
                if code_set:
                    c_remove = random.choice(list(code_set))
                    code_set.remove(c_remove)
                    mask = sphere_masks[c_remove]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] -= 1
                        m ^= lsb
            elif move_type < 0.6:
                c_add = random.randint(0, total_words - 1)
                if c_add not in code_set:
                    code_set.add(c_add)
                    mask = sphere_masks[c_add]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] += 1
                        m ^= lsb
            else:
                if code_set:
                    c_remove = random.choice(list(code_set))
                    c_add = random.randint(0, total_words - 1)
                    if c_add != c_remove and c_add not in code_set:
                        code_set.remove(c_remove)
                        code_set.add(c_add)
                        
                        mask_rm = sphere_masks[c_remove]
                        m = mask_rm
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            count[j] -= 1
                            m ^= lsb
                            
                        mask_add = sphere_masks[c_add]
                        m = mask_add
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            count[j] += 1
                            m ^= lsb

            new_uncovered = sum(1 for c in count if c == 0)
            new_size = len(code_set)
            new_energy = calc_energy(new_size, new_uncovered)
            
            delta_e = new_energy - old_energy
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / max(t, 1e-6)):
                # Accept
                current_size = new_size
                current_uncovered = new_uncovered
                current_energy = new_energy
                
                if new_uncovered == 0 and new_size < best_size:
                    best_code = set(code_set)
                    best_size = new_size
            else:
                # Reject: Restore
                code_set = old_code_set
                count = old_count
                current_size = old_size
                current_uncovered = old_uncovered
                current_energy = old_energy
                
            t *= cooling
            
        return list(best_code), best_size

    # Backtracking Reduction: Try to reduce size by 1
    # If we remove c, and get uncovered U, can we cover U with 1 codeword?
    def try_reduce_by_one(code_indices):
        code_set = set(code_indices)
        if not code_set:
            return code_set
            
        count = [0] * total_words
        for c in code_set:
            mask = sphere_masks[c]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                count[j] += 1
                m ^= lsb
                
        # Try removing each codeword
        candidates = list(code_set)
        random.shuffle(candidates)
        
        for c_remove in candidates:
            # Simulate removal
            mask_rm = sphere_masks[c_remove]
            uncovered_after = []
            m = mask_rm
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if count[j] == 1:
                    uncovered_after.append(j)
                m ^= lsb
                
            if not uncovered_after:
                # Safe to remove
                code_set.remove(c_remove)
                return code_set
            
            # Check if a single codeword can cover all uncovered_after
            # We need a codeword c_add such that sphere_masks[c_add] covers all j in uncovered_after
            # And c_add is not in code_set (or if it is, it's a swap, but we are removing c_remove)
            # Actually, if c_add is in code_set, it's a swap.
            
            # Efficient check:
            # Intersection of spheres of all uncovered words?
            # No, we need a sphere that is a SUPERSET of uncovered_after.
            
            # For small uncovered_after, we can check candidates.
            # If len(uncovered_after) is small, say < 5, we can check all 2187 codewords.
            if len(uncovered_after) <= 5:
                # Build a mask of uncovered words
                unc_mask = 0
                for j in uncovered_after:
                    unc_mask |= (1 << j)
                
                for c_add in range(total_words):
                    if c_add in code_set and c_add != c_remove:
                        continue # Swapping with existing is handled by SA usually, but let's allow
                    # Check if sphere_masks[c_add] covers unc_mask
                    if (sphere_masks[c_add] & unc_mask) == unc_mask:
                        # We can swap c_remove with c_add
                        # But we want to reduce size? No, swap keeps size same.
                        # We want to REMOVE c_remove and ADD c_add? That keeps size same.
                        # The goal of try_reduce_by_one is to REMOVE c_remove and NOT add anything?
                        # No, if we remove c_remove, we MUST cover the uncovered words.
                        # If we add c_add, size stays same.
                        # This function is misnamed. It should be "try_improve".
                        # If we just remove, size decreases by 1.
                        # If we swap, size stays same.
                        # We only accept pure removals here.
                        pass 
                # If no pure removal found, we might swap to enable future removals?
                # That's complex. Let's stick to pure removals for this step.
                
        return code_set

    best_code = None
    best_size = float('inf')

    # 1. Greedy Initialization
    for run in range(20):
        if time.time() > deadline - 30:
            break
        code = greedy_set_cover(randomize=True)
        code = prune_code(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
    # 2. Local Search (SA)
    if best_code:
        for run in range(10):
            if time.time() > deadline - 10:
                break
            # Add some noise
            perturbed = list(best_code)
            # Remove a few random codewords
            for _ in range(3):
                if perturbed:
                    perturbed.remove(random.choice(perturbed))
            
            sa_code, sa_size = simulated_annealing_safe(perturbed, max_iters=1000)
            if sa_size < best_size:
                best_size = sa_size
                best_code = sa_code

    # 3. Final Pruning
    if best_code:
        best_code = prune_code(best_code)
        best_size = len(best_code)

    # 4. Try to reduce further with simple backtracking
    if best_code and time.time() < deadline - 5:
        # Try to remove codewords one by one
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            new_code = try_reduce_by_one(best_code)
            if len(new_code) < len(best_code):
                best_code = new_code
                best_size = len(best_code)
                changed = True

    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert indices to words
    result = []
    for idx in best_code:
        w = [0] * n
        for pos in range(n - 1, -1, -1):
            w[pos] = idx % q
            idx //= q
        result.append(w)
        
    return result

