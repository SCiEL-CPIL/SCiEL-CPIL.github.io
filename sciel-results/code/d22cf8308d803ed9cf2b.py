import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    
    # ---- Build the 256-word linear code: shortened [18,9,6] extended QR code ----
    
    # Quadratic residues mod 19
    qres = set()
    for x in range(1, 19):
        qres.add((x*x) % 19)
    
    def build_linear_code():
        G = np.zeros((9, 18), dtype=np.uint8)
        for i in range(9):
            for j in range(17):
                pos = (j - i) % 19
                if pos in qres:
                    G[i, j] = 1
            G[i, 17] = G[i, :17].sum() % 2
        
        codewords = set()
        for mask in range(256):
            cw = np.zeros(18, dtype=np.uint8)
            for i in range(9):
                if (mask >> i) & 1:
                    cw = (cw + G[i]) % 2
            val = 0
            for j in range(18):
                val = (val << 1) | int(cw[j])
            codewords.add(val)
        
        return codewords, G
    
    all_cw_18, G = build_linear_code()
    
    cw_17 = set()
    for val in all_cw_18:
        if (val & 1) == 0:
            cw_17.add(val >> 1)
    
    cw_list = list(cw_17)
    min_d = 999
    for i in range(len(cw_list)):
        for j in range(i+1, len(cw_list)):
            d = (cw_list[i] ^ cw_list[j]).bit_count()
            if d < min_d:
                min_d = d
    
    if min_d < 6:
        best_shortened = None
        best_min_d = 0
        for coord in range(18):
            cw_short = set()
            for val in all_cw_18:
                if ((val >> (17 - coord)) & 1) == 0:
                    upper = val >> (18 - coord)
                    lower = val & ((1 << (17 - coord)) - 1)
                    cw_short.add((upper << (17 - coord)) | lower)
            if len(cw_short) == 256:
                cl = list(cw_short)
                md = 999
                for i in range(len(cl)):
                    for j in range(i+1, len(cl)):
                        d = (cl[i] ^ cl[j]).bit_count()
                        if d < md:
                            md = d
                if md > best_min_d:
                    best_min_d = md
                    best_shortened = cw_short
        
        if best_shortened is not None:
            cw_list = list(best_shortened)
            min_d = best_min_d
        else:
            cw_list = list(cw_17)
    
    if min_d < 6:
        cw_set = set()
        for w in range(1 << 17):
            if all((w ^ cw).bit_count() >= 6 for cw in cw_set):
                cw_set.add(w)
        cw_list = list(cw_set)
    
    code_list = list(cw_list)
    code_set = set(cw_list)
    
    best_code = list(code_list)
    best_size = len(code_list)
    
    rng = random.Random(42)
    
    # Precompute bits for vectorized popcount
    bits = 1 << np.arange(17)
    
    def get_conflicts_vec(new_word, code_arr):
        xors = code_arr ^ new_word
        mask = (xors[:, None] & bits[None, :]) != 0
        dists = mask.sum(axis=1)
        return np.where(dists < MIN_DIST)[0]

    def try_add_word_with_repair(new_word, code_arr, code_set):
        """
        Try to add new_word. If conflicts exist, try to resolve by flipping 
        1 or 2 bits in one of the conflicting codewords.
        Returns (success, new_code_arr, new_code_set)
        """
        conflict_idx = get_conflicts_vec(new_word, code_arr)
        if len(conflict_idx) == 0:
            return True, np.append(code_arr, new_word), code_set | {new_word}
        
        # Try to repair each conflicting codeword
        # We prioritize conflicts with smallest distance? Or random order?
        # Random order is fine for SA.
        indices = list(conflict_idx)
        rng.shuffle(indices)
        
        for idx in indices:
            cw = int(code_arr[idx])
            
            # Try 1-bit flips
            for bit1 in range(17):
                new_cw = cw ^ (1 << bit1)
                if new_cw == new_word: continue
                if (new_cw ^ new_word).bit_count() < MIN_DIST: continue
                
                # Check new_cw against all others
                temp_arr = code_arr.copy()
                temp_arr[idx] = new_cw
                xors = temp_arr ^ new_cw
                mask = (xors[:, None] & bits[None, :]) != 0
                dists = mask.sum(axis=1)
                dists[idx] = 17
                if (dists < MIN_DIST).any(): continue
                
                # Check new_word against all others in temp_arr
                xors_nw = temp_arr ^ new_word
                mask_nw = (xors_nw[:, None] & bits[None, :]) != 0
                dists_nw = mask_nw.sum(axis=1)
                if (dists_nw < MIN_DIST).any(): continue
                
                # Success
                new_set = code_set.copy()
                new_set.discard(cw)
                new_set.add(new_cw)
                new_set.add(new_word)
                return True, np.append(temp_arr, new_word), new_set
            
            # Try 2-bit flips
            for bit1 in range(17):
                for bit2 in range(bit1 + 1, 17):
                    new_cw = cw ^ (1 << bit1) ^ (1 << bit2)
                    if new_cw == new_word: continue
                    if (new_cw ^ new_word).bit_count() < MIN_DIST: continue
                    
                    temp_arr = code_arr.copy()
                    temp_arr[idx] = new_cw
                    xors = temp_arr ^ new_cw
                    mask = (xors[:, None] & bits[None, :]) != 0
                    dists = mask.sum(axis=1)
                    dists[idx] = 17
                    if (dists < MIN_DIST).any(): continue
                    
                    xors_nw = temp_arr ^ new_word
                    mask_nw = (xors_nw[:, None] & bits[None, :]) != 0
                    dists_nw = mask_nw.sum(axis=1)
                    if (dists_nw < MIN_DIST).any(): continue
                    
                    new_set = code_set.copy()
                    new_set.discard(cw)
                    new_set.add(new_cw)
                    new_set.add(new_word)
                    return True, np.append(temp_arr, new_word), new_set
        
        return False, code_arr, code_set

    # Phase 1: Greedy Polish
    # Try to add words that are "close" to existing ones or random words
    # We iterate through a pool of candidate words
    # To speed up, we don't try all 131072 words. We try random ones.
    
    current_code_arr = np.array(code_list, dtype=np.int64)
    current_set = set(code_list)
    
    # Initial polish: Try to add random words with repair
    # We do this for a short time or fixed number of attempts
    polish_attempts = 5000
    for _ in range(polish_attempts):
        if time.time() > deadline - 10: break
        new_word = rng.randint(0, (1 << N) - 1)
        if new_word in current_set: continue
        
        success, new_arr, new_set = try_add_word_with_repair(new_word, current_code_arr, current_set)
        if success:
            current_code_arr = new_arr
            current_set = new_set
            if len(current_code_arr) > best_size:
                best_code = list(current_code_arr)
                best_size = len(current_code_arr)
    
    # Phase 2: Simulated Annealing
    # Now we use SA to potentially add more words or rearrange to allow more additions.
    # Objective: Maximize size.
    # Move: Pick random word, try to add with repair. If fail, maybe try to remove a word?
    # Or: Pick existing word, try to move it (flip bits) to create space for a new word?
    
    # Let's stick to the "Add with Repair" move but allow "bad" moves (size decrease)
    # if it leads to a better configuration later?
    # Actually, if we can't add, we can try to "swap" a word with a different one to make room.
    
    T = 10.0
    cooling = 0.999
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart from best
            current_code_arr = np.array(best_code, dtype=np.int64)
            current_set = set(best_code)
            T = 10.0
            no_improve = 0
            max_no_improve += 500
        
        new_word = rng.randint(0, (1 << N) - 1)
        
        if new_word in current_set:
            no_improve += 1
            T = max(T * cooling, 0.1)
            continue
            
        # Check if it fits directly
        conflict_idx = get_conflicts_vec(new_word, current_code_arr)
        
        if len(conflict_idx) == 0:
            current_code_arr = np.append(current_code_arr, new_word)
            current_set.add(new_word)
            no_improve = 0
            if len(current_code_arr) > best_size:
                best_code = list(current_code_arr)
                best_size = len(current_code_arr)
            T = max(T * cooling, 0.1)
            continue
        
        # Try repair
        success, new_arr, new_set = try_add_word_with_repair(new_word, current_code_arr, current_set)
        
        if success:
            current_code_arr = new_arr
            current_set = new_set
            no_improve = 0
            if len(current_code_arr) > best_size:
                best_code = list(current_code_arr)
                best_size = len(current_code_arr)
        else:
            # Try a "swap" move: Replace a conflicting word with a random neighbor 
            # to see if it opens up space? 
            # This is complex. Instead, let's just accept the failure and cool down.
            # Or, try to move an existing word to a "worse" position to allow new_word?
            # For now, just cool down.
            no_improve += 1
        
        T = max(T * cooling, 0.1)

    result = best_code
    
    # Verify
    assert len(set(result)) == len(result)
    assert all(0 <= w < (1 << N) for w in result)
    if len(result) > 1:
        arr = np.array(result, dtype=np.int64)
        for i in range(len(arr)):
            xors = arr ^ arr[i]
            mask = (xors[:, None] & bits[None, :]) != 0
            dists = mask.sum(axis=1)
            dists[i] = 17
            if (dists < MIN_DIST).any():
                assert False, f"Distance < {MIN_DIST} found"
    
    return result
