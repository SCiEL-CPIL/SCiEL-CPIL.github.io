import time
import random
import numpy as np

def construct():
    N = 17
    MIN_DISTANCE = 6
    TOTAL = 1 << N  # 131072
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate the 32-word Reed-Muller seed
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
    
    code_set = set(rm_words)
    code_list = list(rm_words)
    best_size = len(code_list)
    
    # Precompute all candidates
    all_candidates = np.arange(TOTAL, dtype=np.uint32)
    
    # Initialize min_dists: min distance from each candidate to any codeword
    min_dists = np.full(TOTAL, 100, dtype=np.uint8)
    
    # Mark seed words
    for w in code_set:
        min_dists[w] = 0
    
    # Compute initial min distances using batched vectorized ops
    BATCH_SIZE = 16384
    for batch_start in range(0, TOTAL, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, TOTAL)
        batch = all_candidates[batch_start:batch_end]
        batch_min = np.full(len(batch), 100, dtype=np.uint8)
        
        for w in code_set:
            # Vectorized XOR and bit_count
            xored = batch ^ w
            dists = np.array([x.bit_count() for x in xored], dtype=np.uint8)
            batch_min = np.minimum(batch_min, dists)
        
        min_dists[batch_start:batch_end] = batch_min
    
    # Greedy passes with random order
    no_progress_count = 0
    while time.time() < deadline:
        # Shuffle candidate indices for random order
        indices = np.arange(TOTAL, dtype=np.uint32)
        random.shuffle(indices.tolist())
        indices = np.array(indices, dtype=np.uint32)
        
        added_this_pass = []
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            i = int(idx)
            if min_dists[i] < MIN_DISTANCE:
                continue
            
            # This candidate is valid: add it
            w = int(all_candidates[i])
            code_set.add(w)
            code_list.append(w)
            min_dists[i] = 0
            added_this_pass.append(w)
        
        if added_this_pass:
            no_progress_count = 0
            # Update min_dists for all candidates based on newly added words
            for w in added_this_pass:
                # Batched update
                for batch_start in range(0, TOTAL, BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, TOTAL)
                    batch = all_candidates[batch_start:batch_end]
                    xored = batch ^ w
                    dists = np.array([x.bit_count() for x in xored], dtype=np.uint8)
                    min_dists[batch_start:batch_end] = np.minimum(min_dists[batch_start:batch_end], dists)
            
            best_size = len(code_list)
        else:
            no_progress_count += 1
            if no_progress_count >= 3:
                break
    
    # Polish phase: try random swaps to escape local optima
    polish_deadline = min(deadline, start_time + 53.0)
    while time.time() < polish_deadline:
        if len(code_list) < 2:
            break
        
        # Try a random swap: remove one codeword, try to add two others
        removed_idx = random.randint(0, len(code_list) - 1)
        removed_word = code_list[removed_idx]
        
        # Tentatively remove
        code_set.discard(removed_word)
        code_list.pop(removed_idx)
        
        # Update min_dists: distances may increase, so we need to recompute
        # For simplicity, recompute min_dists from scratch for the reduced code
        # This is expensive, so only do it if we have time
        if time.time() > polish_deadline - 2.0:
            # Not enough time, restore
            code_set.add(removed_word)
            code_list.append(removed_word)
            break
        
        # Recompute min_dists for the reduced code
        min_dists_polish = np.full(TOTAL, 100, dtype=np.uint8)
        for w in code_set:
            min_dists_polish[w] = 0
        
        BATCH_SIZE_POLISH = 16384
        for batch_start in range(0, TOTAL, BATCH_SIZE_POLISH):
            batch_end = min(batch_start + BATCH_SIZE_POLISH, TOTAL)
            batch = all_candidates[batch_start:batch_end]
            batch_min = np.full(len(batch), 100, dtype=np.uint8)
            for w in code_set:
                xored = batch ^ w
                dists = np.array([x.bit_count() for x in xored], dtype=np.uint8)
                batch_min = np.minimum(batch_min, dists)
            min_dists_polish[batch_start:batch_end] = batch_min
        
        # Try to add two new words
        added_new = []
        indices = np.arange(TOTAL, dtype=np.uint32)
        random.shuffle(indices.tolist())
        indices = np.array(indices, dtype=np.uint32)
        
        for idx in indices:
            if len(added_new) >= 2:
                break
            i = int(idx)
            if min_dists_polish[i] >= MIN_DISTANCE:
                w = int(all_candidates[i])
                code_set.add(w)
                code_list.append(w)
                min_dists_polish[i] = 0
                added_new.append(w)
                # Update min_dists_polish for this new word
                for batch_start in range(0, TOTAL, BATCH_SIZE_POLISH):
                    batch_end = min(batch_start + BATCH_SIZE_POLISH, TOTAL)
                    batch = all_candidates[batch_start:batch_end]
                    xored = batch ^ w
                    dists = np.array([x.bit_count() for x in xored], dtype=np.uint8)
                    min_dists_polish[batch_start:batch_end] = np.minimum(min_dists_polish[batch_start:batch_end], dists)
        
        # Accept the swap if we added more than we removed
        if len(added_new) > 1:
            # Keep the new code
            min_dists = min_dists_polish
            best_size = len(code_list)
        else:
            # Revert: remove the added words and restore the removed word
            for w in added_new:
                code_set.discard(w)
                if w in code_list:
                    code_list.remove(w)
            code_set.add(removed_word)
            code_list.append(removed_word)
            # Restore min_dists (recompute is too expensive; use the pre-swap version)
            # We'll just continue with the next iteration
    
    return code_list
