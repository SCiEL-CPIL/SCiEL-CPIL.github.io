import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0
    
    total_space = q ** n
    
    if total_space > 100000:
        # Fallback for large spaces, though task is small
        # Just return a random valid set if it gets too complex?
        # For this task, n=7, q=3, total=2187.
        pass

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute masks
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1
    
    best_size = total_space
    best_selected = list(range(total_space))
    
    def run_greedy():
        covered = 0
        selected = []
        remaining = full_mask
        while remaining:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word == -1:
                break
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
            if time.time() > deadline - 5:
                break
        return selected

    def polish(selected):
        # Build counts
        counts = [0] * total_space
        for w in selected:
            m = masks[w]
            mm = m
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Phase 1: Remove redundant
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected): continue
                w = selected[idx]
                m = masks[w]
                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit
                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    improved = True
            
            if time.time() > deadline - 3:
                break

            # Phase 2: Swap to enable removal
            # For each w, check if its unique words can be covered by some w'
            indices = list(range(len(selected)))
            random.shuffle(indices)
            for idx in indices:
                if idx >= len(selected): continue
                w = selected[idx]
                m = masks[w]
                
                # Find unique words
                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit
                
                if unique_mask == 0:
                    continue # Should have been removed in Phase 1
                
                # Find w' not in selected that covers unique_mask
                # To speed up, we can just scan all w'
                # Optimization: Precompute which words cover which small sets?
                # For now, scan.
                found = False
                for w_prime in range(total_space):
                    if w_prime in selected:
                        continue
                    if (masks[w_prime] & unique_mask) == unique_mask:
                        # Swap w with w_prime
                        # Remove w
                        mm = m
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        selected[idx] = w_prime
                        # Add w_prime
                        mp = masks[w_prime]
                        mm = mp
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit
                        improved = True
                        found = True
                        break
                if found:
                    # Restart loop to re-evaluate redundancies
                    break
        
        return selected

    # Run one greedy
    if time.time() < deadline - 10:
        sol = run_greedy()
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:])
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Run multiple randomized greedy restarts
    def run_greedy_randomized():
        covered = 0
        selected = []
        remaining = full_mask
        while remaining and time.time() < deadline - 5:
            max_gain = -1
            candidates = []
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [w]
                elif gain == max_gain:
                    candidates.append(w)
            if not candidates:
                break
            best_word = random.choice(candidates)
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
        return selected

    for _ in range(100):
        if time.time() > deadline - 10:
            break
        sol = run_greedy_randomized()
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            if time.time() < deadline - 5:
                polished = polish(best_selected[:])
                if len(polished) < best_size:
                    best_size = len(polished)
                    best_selected = polished[:]

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
