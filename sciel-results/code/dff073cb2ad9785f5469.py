import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Pre-compute popcount table for 17-bit integers
    popcount = [0] * (1 << 17)
    for i in range(1, 1 << 17):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def popcount_fast(x):
        return popcount[x & 0x1FFFF]

    def is_valid_code(words):
        for i in range(len(words)):
            wi = words[i]
            for j in range(i + 1, len(words)):
                if popcount_fast(wi ^ words[j]) < 6:
                    return False
        return True

    # Step 1: Build the [17,8,6] linear code (256 words)
    gf_poly = 0x1d  # x^8+x^4+x^3+x^2+1, primitive in GF(2^8)

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            b >>= 1
            if a & 0x80:
                a = ((a << 1) ^ gf_poly) & 0xFF
            else:
                a = (a << 1) & 0xFF
        return res

    def gf_pow(a, n):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a)
            a = gf_mul(a, a)
            n >>= 1
        return res

    # Find element of order 17
    alpha = None
    for candidate in range(1, 256):
        if gf_pow(candidate, 17) == 1 and candidate != 1:
            alpha = candidate
            break

    if alpha is None:
        alpha = 3

    # Quadratic residues mod 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]

    # Build generator polynomial g(x) of degree 8
    g = [1]
    for i in Q:
        root = gf_pow(alpha, i)
        new_g = [0] * (len(g) + 1)
        for j in range(len(g)):
            new_g[j] ^= gf_mul(g[j], root)
            new_g[j + 1] ^= g[j]
        g = new_g

    # Generate all 512 words of the [17,9,5] code
    words_17_9_5 = []
    for m in range(512):
        c = [0] * 17
        for j in range(9):
            if (m >> j) & 1:
                for k in range(9):
                    if (g[k] != 0):
                        pos = (j + k) % 17
                        c[pos] ^= 1
        val = 0
        for bit in c:
            val = (val << 1) | bit
        words_17_9_5.append(val)

    # Extend to [18,9,6] by appending overall parity
    words_18_9_6 = []
    for w in words_17_9_5:
        parity = popcount_fast(w) % 2
        ext_val = w | (parity << 17)
        words_18_9_6.append(ext_val)

    # Shorten: keep words with parity bit = 0
    words_17_8_6 = []
    for w in words_18_9_6:
        if (w >> 17) & 1 == 0:
            w_short = w & 0x1FFFF
            words_17_8_6.append(w_short)

    words_17_8_6 = list(set(words_17_8_6))
    base_code = words_17_8_6

    if not is_valid_code(base_code):
        valid_code = [base_code[0]]
        for w in base_code[1:]:
            ok = True
            for v in valid_code:
                if popcount_fast(w ^ v) < 6:
                    ok = False
                    break
            if ok:
                valid_code.append(w)
        base_code = valid_code

    current_code = list(base_code)
    best_code = list(base_code)
    best_size = len(current_code)

    if best_size < 32:
        return [0]

    # Step 2: Vectorized Greedy Extension with Priority Ordering
    # Precompute byte popcount table for vectorized operations
    byte_popcount = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def vectorized_popcount(arr):
        """Compute popcount for an array of uint32 integers using byte lookup."""
        low = arr & 0xFF
        high = (arr >> 8) & 0xFF
        return byte_popcount[low] + byte_popcount[high]

    code_set = set(current_code)
    code_arr = np.array(current_code, dtype=np.uint32)
    
    # Generate all possible 17-bit words
    all_words = np.arange(1 << 17, dtype=np.uint32)
    
    # Try multiple random orderings and keep the best result
    num_trials = 3
    for trial in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        # Reset code to base for this trial
        current_code = list(base_code)
        code_set = set(current_code)
        code_arr = np.array(current_code, dtype=np.uint32)
        
        # Shuffle with different seed for diversity
        rng = np.random.RandomState(trial * 12345 + 67890)
        shuffled = all_words.copy()
        rng.shuffle(shuffled)
        
        # Process in batches with priority: sort by min distance to current code
        batch_size = 1024
        num_batches = len(shuffled) // batch_size + 1
        
        for b in range(num_batches):
            if time.time() > deadline - 5.0:
                break
            
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, len(shuffled))
            if start_idx >= len(shuffled):
                break
                
            batch = shuffled[start_idx:end_idx]
            
            # Check compatibility with current code
            xor_matrix = batch[:, np.newaxis] ^ code_arr[np.newaxis, :]
            pcs = vectorized_popcount(xor_matrix)
            
            # A candidate is valid if min popcount in its row >= 6
            min_pcs = np.min(pcs, axis=1)
            valid_mask = min_pcs >= 6
            
            if np.any(valid_mask):
                # Among valid candidates, prioritize those with higher min distance
                # This is a smooth surrogate: we prefer words that are "far" from the code
                valid_indices = np.where(valid_mask)[0]
                valid_scores = min_pcs[valid_indices]
                
                # Sort valid candidates by score (descending) - add farthest first
                sorted_valid = valid_indices[np.argsort(-valid_scores)]
                
                added_any = False
                for idx in sorted_valid:
                    w = int(batch[idx])
                    if w not in code_set:
                        code_set.add(w)
                        current_code.append(w)
                        added_any = True
                
                if added_any:
                    if len(current_code) > best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)
                    
                    # Update code array for next batch
                    code_arr = np.array(current_code, dtype=np.uint32)
    
    # Step 3: Local Search - Try to improve by swapping out tight words
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 2.0:
        # Start from the best code found
        current_code = list(best_code)
        code_set = set(current_code)
        
        # Compute for each word in the code, how many other words are at distance exactly 6
        # Words with many distance-6 neighbors are "tight" and might be good to remove
        tightness = np.zeros(len(current_code), dtype=np.int32)
        code_arr = np.array(current_code, dtype=np.uint32)
        
        for i in range(len(current_code)):
            for j in range(i + 1, len(current_code)):
                d = popcount_fast(current_code[i] ^ current_code[j])
                if d == 6:
                    tightness[i] += 1
                    tightness[j] += 1
        
        # Try removing the tightest words and see if we can add more
        for _ in range(50):  # Try up to 50 swap attempts
            if time.time() > deadline - 5.0:
                break
            
            # Find the tightest word
            tightest_idx = np.argmax(tightness)
            if tightness[tightest_idx] == 0:
                break  # No tight words to remove
            
            # Remove the tightest word
            removed_word = current_code[tightest_idx]
            current_code.pop(tightest_idx)
            code_set.remove(removed_word)
            tightness = np.delete(tightness, tightest_idx)
            
            # Try to add back a different word that was previously rejected
            # Scan through all_words to find a valid candidate
            found_better = False
            for w in all_words:
                w_int = int(w)
                if w_int in code_set:
                    continue
                # Check if this word is compatible with all current code words
                ok = True
                for c in current_code:
                    if popcount_fast(w_int ^ c) < 6:
                        ok = False
                        break
                if ok:
                    # Check if adding this word allows us to get a larger code overall
                    # For simplicity, just add it and see if we can add more
                    code_set.add(w_int)
                    current_code.append(w_int)
                    found_better = True
                    break
            
            if not found_better:
                # Put the word back
                current_code.append(removed_word)
                code_set.add(removed_word)
                # Recompute tightness (expensive, but we're at the end)
                break
            else:
                # Recompute tightness for the new code
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
        
        # If we improved, continue from the new best
        if len(best_code) > len(current_code):
            current_code = list(best_code)
            code_set = set(current_code)

    # Step 4: Final polish - try to add any remaining compatible words
    remaining_time = deadline - 5.0 - time.time()
    if remaining_time > 0.5:
        current_code = list(best_code)
        code_set = set(current_code)
        
        # Try a few random additions
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            w = random.randint(0, (1 << 17) - 1)
            if w in code_set:
                continue
            ok = True
            for c in current_code:
                if popcount_fast(w ^ c) < 6:
                    ok = False
                    break
            if ok:
                code_set.add(w)
                current_code.append(w)
                if len(current_code) > best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)

    # Final verification
    if len(best_code) >= 32:
        if is_valid_code(best_code):
            return best_code
        else:
            final = []
            for w in best_code:
                ok = True
                for v in final:
                    if popcount_fast(w ^ v) < 6:
                        ok = False
                        break
                if ok:
                    final.append(w)
            if len(final) >= 32:
                return final
            return base_code
    else:
        return base_code if len(base_code) >= 32 else [0]
