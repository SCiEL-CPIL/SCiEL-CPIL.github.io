import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 17
    min_dist = 6
    full_size = 1 << n  # 131072
    
    # Precompute popcounts for all 17-bit numbers
    popcount = np.zeros(full_size, dtype=np.int32)
    for i in range(1, full_size):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Construct the [17,8,6] quadratic residue code (256 words)
    # Generator polynomial for the shortened QR code
    gen_poly = (1 << 9) | (1 << 7) | (1 << 6) | (1 << 3) | (1 << 0)
    
    def encode(msg):
        cw = 0
        for bit_pos in range(9):
            if (msg >> bit_pos) & 1:
                cw ^= (gen_poly << bit_pos)
        return cw
    
    qr_codewords = []
    seen = set()
    for msg in range(256):
        cw = encode(msg)
        if cw not in seen:
            qr_codewords.append(cw)
            seen.add(cw)
    
    # Verify the QR code
    valid_qr = (len(qr_codewords) == 256)
    if valid_qr:
        for i in range(256):
            for j in range(i + 1, 256):
                if popcount[qr_codewords[i] ^ qr_codewords[j]] < min_dist:
                    valid_qr = False
                    break
            if not valid_qr:
                break
    
    if valid_qr:
        best_codewords = list(qr_codewords)
    else:
        best_codewords = [0]
    
    best_size = len(best_codewords)
    
    def try_add_greedy(codewords, candidates_pool, time_limit):
        """Add valid candidates greedily using vectorized distance checks."""
        current = list(codewords)
        current_set = set(current)
        added = 0
        
        while time.time() < time_limit:
            if not candidates_pool:
                break
            
            cw_arr = np.array(current, dtype=np.int32)
            n_cw = len(cw_arr)
            
            # Find all valid candidates in batches
            found_valid = False
            for start in range(0, len(candidates_pool), 5000):
                if time.time() > time_limit:
                    break
                end = min(start + 5000, len(candidates_pool))
                batch = np.array(candidates_pool[start:end], dtype=np.int32)
                
                # Compute distances from batch to all codewords
                # Use broadcasting: (batch_size, n_cw)
                # Memory: 5000 * 256 * 4 bytes = 5MB, fine
                min_dists = np.full(len(batch), min_dist, dtype=np.int32)
                for cw in cw_arr:
                    d = popcount[batch ^ cw]
                    min_dists = np.minimum(min_dists, d)
                
                valid_mask = min_dists >= min_dist
                if np.any(valid_mask):
                    # Pick the first valid one
                    idx = np.argmax(valid_mask)
                    new_cw = int(batch[idx])
                    current.append(new_cw)
                    current_set.add(new_cw)
                    candidates_pool.pop(start + idx)
                    added += 1
                    found_valid = True
                    break
            
            if not found_valid:
                break
        
        return current, candidates_pool, added
    
    def local_search_swaps(codewords, candidates_pool, time_limit):
        """Try swap-based local search: remove one, add two to net +1."""
        current = list(codewords)
        current_set = set(current)
        improved = True
        total_added = 0
        
        while improved and time.time() < time_limit:
            improved = False
            n_cw = len(current)
            
            # Try each codeword as removal candidate
            for remove_idx in range(n_cw):
                if time.time() > time_limit:
                    return current, candidates_pool, total_added
                
                remove_cw = current[remove_idx]
                trial_set = current_set - {remove_cw}
                trial_cw_arr = np.array([c for i, c in enumerate(current) if i != remove_idx], dtype=np.int32)
                
                # Find candidates that are valid against trial_set
                valid_cands = []
                for start in range(0, len(candidates_pool), 5000):
                    if time.time() > time_limit:
                        break
                    end = min(start + 5000, len(candidates_pool))
                    batch = np.array(candidates_pool[start:end], dtype=np.int32)
                    
                    min_dists = np.full(len(batch), min_dist, dtype=np.int32)
                    for cw in trial_cw_arr:
                        d = popcount[batch ^ cw]
                        min_dists = np.minimum(min_dists, d)
                    
                    valid_mask = min_dists >= min_dist
                    if np.any(valid_mask):
                        for i in np.where(valid_mask)[0]:
                            valid_cands.append(int(batch[i]))
                    
                    if len(valid_cands) >= 2:
                        break
                
                if len(valid_cands) < 2:
                    continue
                
                # Try pairs from valid_cands
                for i in range(min(len(valid_cands), 50)):
                    for j in range(i + 1, min(len(valid_cands), 50)):
                        c1, c2 = valid_cands[i], valid_cands[j]
                        if popcount[c1 ^ c2] < min_dist:
                            continue
                        
                        # Check c1 and c2 are valid against each other and trial set
                        # Already checked against trial set, now check pair
                        # Also need to ensure they're not in current
                        if c1 in current_set or c2 in current_set:
                            continue
                        
                        # This swap gives net +1
                        new_current = current[:remove_idx] + current[remove_idx+1:] + [c1, c2]
                        new_set = trial_set | {c1, c2}
                        
                        # Verify
                        valid = True
                        for a in range(len(new_current)):
                            for b in range(a+1, len(new_current)):
                                if popcount[new_current[a] ^ new_current[b]] < min_dist:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            current = new_current
                            current_set = new_set
                            # Remove c1, c2 from candidates_pool
                            if c1 in candidates_pool:
                                candidates_pool.remove(c1)
                            if c2 in candidates_pool:
                                candidates_pool.remove(c2)
                            total_added += 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current, candidates_pool, total_added
    
    # Phase 1: Greedy addition
    current_set = set(best_codewords)
    candidates_pool = [x for x in range(full_size) if x not in current_set]
    
    best_codewords, candidates_pool, added = try_add_greedy(
        best_codewords, candidates_pool, deadline - 20.0
    )
    best_size = len(best_codewords)
    best_codewords = list(best_codewords)
    
    # Phase 2: Local search with swaps
    if time.time() < deadline - 5.0:
        current_set = set(best_codewords)
        candidates_pool = [x for x in range(full_size) if x not in current_set]
        best_codewords, candidates_pool, sw_added = local_search_swaps(
            best_codewords, candidates_pool, deadline - 5.0
        )
        best_size = len(best_codewords)
    
    # Phase 3: More greedy if time remains
    if time.time() < deadline - 3.0:
        current_set = set(best_codewords)
        candidates_pool = [x for x in range(full_size) if x not in current_set]
        best_codewords, candidates_pool, added2 = try_add_greedy(
            best_codewords, candidates_pool, deadline - 3.0
        )
        best_size = len(best_codewords)
    
    return best_codewords
