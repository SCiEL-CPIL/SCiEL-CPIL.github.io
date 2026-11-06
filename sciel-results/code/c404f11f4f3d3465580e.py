import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 17
    min_dist = 6
    max_size = 340
    
    # Precompute popcounts for all 17-bit numbers
    popcount = np.zeros(1 << n, dtype=np.int32)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Construct the [17,8,6] quadratic residue code
    # The generator polynomial for the [17,8,6] QR code
    # We need to be careful with the polynomial representation
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
    
    valid_qr = (len(qr_codewords) == 256)
    if valid_qr:
        # Verify pairwise distances
        for i in range(len(qr_codewords)):
            for j in range(i + 1, len(qr_codewords)):
                if popcount[qr_codewords[i] ^ qr_codewords[j]] < min_dist:
                    valid_qr = False
                    break
            if not valid_qr:
                break
    
    if valid_qr:
        best_codewords = qr_codewords
        best_size = 256
    else:
        # Fallback to greedy from 0
        best_codewords = [0]
        best_size = 1
        for candidate in range(1, 1 << n):
            if time.time() > deadline - 30.0:
                break
            valid = True
            for cw in best_codewords:
                if popcount[candidate ^ cw] < min_dist:
                    valid = False
                    break
            if valid:
                best_codewords.append(candidate)
                best_size += 1
    
    if best_size >= max_size:
        return best_codewords
    
    # Convert to numpy arrays for vectorized operations
    codewords_arr = np.array(best_codewords, dtype=np.int32)
    
    # Candidate pool: all numbers not already in the code
    all_numbers = set(range(1 << n))
    current_set = set(best_codewords)
    candidates = [x for x in range(1 << n) if x not in current_set]
    
    def vectorized_mrv(candidates, codewords_arr, popcount, min_dist):
        """
        Find the candidate with minimum number of conflicts (fewest codewords within distance < min_dist).
        Returns the index of the best candidate or -1 if none valid.
        """
        if not candidates:
            return -1
        
        # Convert to numpy array
        cand_arr = np.array(candidates, dtype=np.int32)
        
        # Compute distances from each candidate to all codewords
        # We need to do this in batches to avoid memory issues
        # For each candidate, compute min distance to any codeword
        # Actually, we want to count how many codewords are within distance < min_dist
        
        # Let's compute for each candidate the number of codewords it conflicts with
        # conflict_count[i] = number of j such that popcount[cand_arr[i] ^ codewords_arr[j]] < min_dist
        
        # This is a matrix of size len(candidates) x len(codewords_arr)
        # For large sets this could be memory intensive, but 131072 x 256 = 33M entries, 
        # which is about 132MB for int32, might be too much. Let's do it in chunks.
        
        n_cands = len(cand_arr)
        n_cw = len(codewords_arr)
        
        # Process in chunks of 1000 candidates
        chunk_size = 1000
        best_idx = -1
        best_conflicts = float('inf')
        
        for start in range(0, n_cands, chunk_size):
            end = min(start + chunk_size, n_cands)
            chunk = cand_arr[start:end]
            
            # Compute XOR with all codewords
            # Shape: (chunk_len, n_cw)
            # We need to broadcast: chunk[:, None] ^ codewords_arr[None, :]
            # This might use too much memory for large chunks
            # Let's use a smaller chunk size
            # Actually, let's compute distances in a more memory-efficient way
            
            # For each codeword, compute distance to all candidates in chunk
            # This is n_cw x chunk_len, which is 256 x 1000 = 256K, manageable
            
            conflict_counts = np.zeros(len(chunk), dtype=np.int32)
            
            for cw in codewords_arr:
                # Compute popcount of chunk ^ cw
                xor_vals = chunk ^ cw
                # Use precomputed popcount
                dists = popcount[xor_vals]
                # Count how many are < min_dist
                conflict_counts += (dists < min_dist).astype(np.int32)
            
            # Find candidate with minimum conflicts in this chunk
            min_conflicts = np.min(conflict_counts)
            if min_conflicts < best_conflicts:
                best_conflicts = min_conflicts
                # Find index in original candidates list
                local_idx = np.argmin(conflict_counts)
                best_idx = start + local_idx
            
            # If we found a candidate with 0 conflicts, we can stop early
            if best_conflicts == 0:
                break
        
        if best_conflicts == 0:
            return best_idx
        else:
            return -1  # No valid candidate with 0 conflicts
    
    # Greedy Independent Set with MRV heuristic
    # Start from the QR code, try to add more words
    
    current_codewords = list(best_codewords)
    current_set = set(current_codewords)
    
    # We'll iterate: find the best candidate to add, add it, repeat
    # To speed up, we can batch the MRV computation
    
    added_count = 0
    while time.time() < deadline - 5.0:
        if not candidates:
            break
        
        # Find best candidate using MRV
        best_idx = vectorized_mrv(candidates, np.array(current_codewords, dtype=np.int32), popcount, min_dist)
        
        if best_idx == -1:
            # No more candidates can be added with 0 conflicts
            break
        
        # Add the candidate
        new_cw = candidates[best_idx]
        current_codewords.append(new_cw)
        current_set.add(new_cw)
        candidates.pop(best_idx)
        added_count += 1
        
        # Update best if improved
        if len(current_codewords) > best_size:
            best_size = len(current_codewords)
            best_codewords = list(current_codewords)
    
    # Try randomized restarts if we have time
    if time.time() < deadline - 5.0 and best_size < max_size:
        # Try a few random shuffles of the candidate order
        # and use a simpler greedy to see if we can do better
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            
            random.shuffle(candidates)
            trial_codewords = list(best_codewords)
            trial_set = set(trial_codewords)
            
            for cand in candidates:
                if time.time() > deadline - 5.0:
                    break
                if cand in trial_set:
                    continue
                valid = True
                for cw in trial_codewords:
                    if popcount[cand ^ cw] < min_dist:
                        valid = False
                        break
                if valid:
                    trial_codewords.append(cand)
                    trial_set.add(cand)
            
            if len(trial_codewords) > best_size:
                best_size = len(trial_codewords)
                best_codewords = trial_codewords
    
    return best_codewords
