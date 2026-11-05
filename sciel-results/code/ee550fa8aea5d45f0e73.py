"""Construct largest 17-bit binary code with min Hamming distance 6."""
import numpy as np
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_linear_code():
    """Construct the 256-word [17,8,6] code from the [18,9,6] extended QR code."""
    poly = 0x1D
    
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res
    
    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res
    
    alpha = None
    for a in range(1, 256):
        if gf_pow(a, 17) == 1 and a != 1:
            alpha = a
            break
    
    if alpha is None:
        return [0]
    
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    g_coeffs = [1]
    
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g_coeffs) + 1)
        for j in range(len(g_coeffs)):
            new_g[j + 1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
    
    G = np.zeros((9, 17), dtype=np.int64)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                pos = (j + k) % 17
                G[k, pos] = 1
    
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.int64)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
    
    words_18 = []
    for w in words_17:
        parity = int(np.sum(w) % 2)
        word_ext = np.append(w, parity)
        words_18.append(word_ext)
    
    words_short = []
    for w in words_18:
        if w[17] == 0:
            w_short = w[:17]
            val = 0
            for i in range(17):
                if w_short[i]:
                    val |= (1 << i)
            words_short.append(val)
    
    return words_short


def _generate_distance_k_words(N, k):
    """Generate all N-bit words with Hamming weight exactly k."""
    words = []
    for positions in combinations(range(N), k):
        val = 0
        for p in positions:
            val |= (1 << p)
        words.append(val)
    return np.array(words, dtype=np.int64)


def construct():
    """Construct the largest set of 17-bit words with pairwise Hamming distance at least 6."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 17
    MIN_DIST = 6
    NUM_WORDS = 2**N
    
    # Build popcount table for all 17-bit values (vectorized via bit tricks)
    _PC17 = np.zeros(NUM_WORDS, dtype=np.int8)
    for i in range(1, NUM_WORDS):
        _PC17[i] = _PC17[i >> 1] + (i & 1)
    
    # Step 1: Construct the 256-word linear code
    try:
        words = _build_linear_code()
    except Exception:
        words = [0]
    
    if not words:
        return [0]
    
    best_words = list(words)
    best_size = len(best_words)
    
    # Step 2: Generate candidate pool - words at various weights not in code
    code_set = set(best_words)
    all_candidates_list = []
    for k in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
        if time.time() > deadline - 40.0:
            break
        cw = _generate_distance_k_words(N, k)
        # Filter out words already in code
        mask = np.array([w not in code_set for w in cw], dtype=bool)
        if mask.any():
            all_candidates_list.append(cw[mask])
    
    if all_candidates_list:
        all_candidates = np.concatenate(all_candidates_list)
    else:
        all_candidates = np.array([], dtype=np.int64)
    
    # Step 3: Greedy addition with vectorized distance checks
    def try_add_candidates(cand_arr, current_list, current_set, current_arr, deadline):
        """Try to add candidates to the code using vectorized checks."""
        added = []
        n_cand = len(cand_arr)
        
        if n_cand == 0:
            return added, current_list, current_set, current_arr
        
        # Process in batches for memory efficiency
        batch_size = 4000
        for start in range(0, n_cand, batch_size):
            if time.time() > deadline - 5.0:
                break
            
            end = min(start + batch_size, n_cand)
            batch = cand_arr[start:end]
            
            # Compute distances from each candidate to each current codeword
            diffs = batch[:, None] ^ current_arr[None, :]
            dists = _PC17[diffs]
            
            # For each candidate, find min distance to any codeword
            min_dists = np.min(dists, axis=1)
            
            # Candidates with min_dist >= MIN_DIST are valid
            valid_mask = min_dists >= MIN_DIST
            valid_cands = batch[valid_mask]
            
            # Add valid candidates one by one (need to update current_arr)
            for w in valid_cands:
                if time.time() > deadline - 3.0:
                    break
                
                w = int(w)
                if w in current_set:
                    continue
                
                current_list.append(w)
                current_set.add(w)
                current_arr = np.append(current_arr, w)
                added.append(w)
        
        return added, current_list, current_set, current_arr
    
    # Try multiple random orderings to maximize additions
    rng = np.random.default_rng(seed=42)
    
    best_added = 0
    best_final_list = list(best_words)
    best_final_arr = np.array(best_words, dtype=np.int64)
    
    for restart in range(30):
        if time.time() > deadline - 15.0:
            break
        
        perm = rng.permutation(len(all_candidates))
        ordered_cands = all_candidates[perm]
        
        current_list = list(best_words)
        current_set = set(current_list)
        current_arr = np.array(current_list, dtype=np.int64)
        
        added, current_list, current_set, current_arr = try_add_candidates(
            ordered_cands, current_list, current_set, current_arr, deadline
        )
        
        if len(added) > best_added:
            best_added = len(added)
            best_final_list = current_list
            best_final_arr = current_arr
    
    if len(best_final_list) > best_size:
        best_size = len(best_final_list)
        best_words = best_final_list
    
    # Step 4: Nonlinear local search (hill climbing) - try to add more words
    # and perform swaps to free up space
    current_list = list(best_words)
    current_set = set(current_list)
    current_arr = np.array(current_list, dtype=np.int64)
    
    # Build full universe of all 2^17 words for swap search
    all_words = np.arange(NUM_WORDS, dtype=np.int64)
    
    # Remove current codewords from consideration for "add" moves
    not_in_code = np.array([w not in current_set for w in all_words], dtype=bool)
    
    # Hill climbing: iteratively try to add a word or swap
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        
        # --- Try to add a word ---
        # Check all words not in code for min distance >= 6
        # Process in batches
        add_batch_size = 8000
        n_total = NUM_WORDS
        for start in range(0, n_total, add_batch_size):
            if time.time() > deadline - 10.0:
                break
            
            end = min(start + add_batch_size, n_total)
            batch = all_words[start:end]
            
            # Filter out words already in code
            in_code_mask = np.array([w in current_set for w in batch], dtype=bool)
            if in_code_mask.all():
                continue
            candidate_batch = batch[~in_code_mask]
            
            if len(candidate_batch) == 0:
                continue
            
            # Compute min distance to current code
            diffs = candidate_batch[:, None] ^ current_arr[None, :]
            dists = _PC17[diffs]
            min_dists = np.min(dists, axis=1)
            
            valid_mask = min_dists >= MIN_DIST
            if valid_mask.any():
                # Add the first valid word found
                w = int(candidate_batch[np.argmax(valid_mask)])
                if w not in current_set:
                    current_list.append(w)
                    current_set.add(w)
                    current_arr = np.append(current_arr, w)
                    improved = True
                    break  # Restart scanning after adding
        
        if improved:
            continue
        
        # --- Try swap moves: remove one codeword, add two ---
        # For each codeword i, check if removing it allows adding >= 1 new word
        # This is expensive, so only try a subset
        if len(current_list) > 256:
            # Try random swaps
            for _ in range(50):
                if time.time() > deadline - 10.0:
                    break
                
                i = rng.integers(0, len(current_arr))
                removed = int(current_arr[i])
                
                # Temporarily remove
                temp_arr = np.delete(current_arr, i)
                temp_set = set(current_list)
                temp_set.discard(removed)
                
                # Try to add words
                add_batch_size = 8000
                n_total = NUM_WORDS
                found = None
                for start in range(0, n_total, add_batch_size):
                    if time.time() > deadline - 10.0:
                        break
                    
                    end = min(start + add_batch_size, n_total)
                    batch = all_words[start:end]
                    
                    in_code_mask = np.array([w in temp_set for w in batch], dtype=bool)
                    if in_code_mask.all():
                        continue
                    candidate_batch = batch[~in_code_mask]
                    
                    if len(candidate_batch) == 0:
                        continue
                    
                    diffs = candidate_batch[:, None] ^ temp_arr[None, :]
                    dists = _PC17[diffs]
                    min_dists = np.min(dists, axis=1)
                    
                    valid_mask = min_dists >= MIN_DIST
                    if valid_mask.any():
                        found = int(candidate_batch[np.argmax(valid_mask)])
                        break
                
                if found is not None:
                    # Perform the swap
                    current_list[i] = found
                    current_set.discard(removed)
                    current_set.add(found)
                    current_arr[i] = found
                    improved = True
                    break
    
    if len(current_list) > best_size:
        best_size = len(current_list)
        best_words = current_list
    
    return best_words

# EVOLVE-BLOCK-END
