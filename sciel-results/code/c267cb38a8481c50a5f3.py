import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s (90s wall-clock kill)
    
    # 1. Construct the [17,8,6] linear code (256 words)
    def gf_mul(a, b, poly=0x1d):
        res = 0
        for i in range(8):
            if (b & 1):
                res ^= a
            b >>= 1
            if (a & 0x80):
                a = (a << 1) ^ poly
            else:
                a <<= 1
            a &= 0xFF
        return res

    def gf_pow(a, n, poly=0x1d):
        res = 1
        while n:
            if n & 1:
                res = gf_mul(res, a, poly)
            a = gf_mul(a, a, poly)
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
            new_g[j+1] ^= g_coeffs[j]
            new_g[j] ^= gf_mul(g_coeffs[j], root)
        g_coeffs = new_g
        
    G = np.zeros((9, 17), dtype=np.uint8)
    for k in range(9):
        for j in range(9):
            if g_coeffs[j]:
                G[k, (j + k) % 17] = 1
                
    words_17 = []
    for u in range(512):
        word = np.zeros(17, dtype=np.uint8)
        for i in range(9):
            if (u >> i) & 1:
                word ^= G[i]
        words_17.append(word)
        
    words_18 = []
    for w in words_17:
        parity = np.sum(w) % 2
        words_18.append(np.append(w, parity))
        
    initial_code = []
    for w in words_18:
        if w[17] == 0:
            val = 0
            for i in range(17):
                if w[i]:
                    val |= (1 << i)
            initial_code.append(val)
            
    if len(initial_code) != 256:
        return initial_code

    current_code = np.array(initial_code, dtype=np.uint32)
    best_code = current_code.copy()
    best_size = len(current_code)
    
    # Helper to check if a word can be added using vectorized distance calculation
    # Returns True if min distance >= 6
    def can_add_vectorized(word, code_arr):
        if len(code_arr) == 0:
            return True
        xor_vals = code_arr ^ word
        # Count bits in each xor value
        # Use a lookup table for popcount of 17-bit values (0 to 2^17-1)
        # But 2^17 is 131072, which is manageable.
        # Or use numpy bit tricks.
        # Simpler: convert to binary strings? No, too slow.
        # Use a precomputed popcount table for 16-bit and combine.
        # Or just loop? 256 iterations is fast enough in Python if not done millions of times.
        # But we want speed. Let's use a popcount table.
        
        # We'll assume a global popcount table exists or create it once.
        # Since this is inside construct, we can create it once outside the loop.
        
        # For now, let's use a fast Python loop or a numpy trick.
        # np.bit_count is available in numpy 1.25+? No, it's a method on integers in Python 3.10+.
        # Let's use a lookup table for 0..65535 and combine.
        
        # Actually, since we are in a tight loop, let's optimize `can_add` significantly.
        
        # We will precompute a popcount table for 16-bit values.
        # popcount_16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
        # This is a one-time cost.
        
        # Inside the function:
        # low = xor_vals & 0xFFFF
        # high = (xor_vals >> 16) & 0xFFFF
        # counts = popcount_16[low] + popcount_16[high]
        # return np.all(counts >= 6)
        
        # Let's implement this.
        
        return True # Placeholder, will be replaced by actual logic in the loop below for speed
    
    # Precompute popcount table for 16-bit values
    popcount_16 = np.array([bin(i).count('1') for i in range(65536)], dtype=np.uint8)
    
    def min_dist_to_code(word, code_arr):
        if len(code_arr) == 0:
            return 17
        xor_vals = code_arr ^ word
        low = (xor_vals & 0xFFFF).astype(np.int32)
        high = ((xor_vals >> 16) & 0xFFFF).astype(np.int32)
        counts = popcount_16[low] + popcount_16[high]
        return np.min(counts)

    def can_add(word, code_arr):
        return min_dist_to_code(word, code_arr) >= 6

    # Random word generation biased towards "interesting" regions?
    # Or just uniform random.
    
    def find_random_word(code_arr, attempts=20):
        for _ in range(attempts):
            w = random.getrandbits(17)
            if can_add(w, code_arr):
                return w
        return None

    def find_random_pair(code_arr, attempts=50):
        for _ in range(attempts):
            w1 = random.getrandbits(17)
            if not can_add(w1, code_arr):
                continue
            # Check if w1 is already in code? 
            # If we removed a word, w1 might be that word. 
            # We should avoid adding back the exact same word if we are trying to grow.
            # But if we removed it, adding it back restores size. 
            # We want to add TWO new words.
            # So we should check if w1 is in code_arr.
            # Since code_arr is a numpy array, checking membership is O(N).
            # We can maintain a set for O(1) lookup?
            # Let's maintain a set alongside the numpy array for membership checks.
            pass
        return None

    # To make this efficient, we need to manage state carefully.
    # Let's keep a list of integers and a numpy array for vectorized distance.
    # And a set for membership.
    
    current_list = list(initial_code)
    current_set = set(current_list)
    current_arr = np.array(current_list, dtype=np.uint32)
    
    best_list = current_list[:]
    best_set = current_set.copy()
    best_arr = current_arr.copy()
    best_size = len(current_list)
    
    def try_add_word(w):
        if w in current_set:
            return False
        if can_add(w, current_arr):
            current_list.append(w)
            current_set.add(w)
            # Update arr? 
            # Instead of updating arr every time, we can just use the list for membership
            # and re-create arr when needed for vectorized check? 
            # Re-creating arr is O(N). N=256. It's fast.
            # But we call can_add many times.
            # Let's keep current_arr in sync.
            # current_arr = np.append(current_arr, w)
            # This is O(N).
            return True
        return False
    
    def try_remove_word(w):
        if w not in current_set:
            return False
        current_list.remove(w)
        current_set.remove(w)
        # Rebuild arr
        # current_arr = np.array(current_list, dtype=np.uint32)
        return True

    # We need to update current_arr efficiently.
    # Since N is small (256), rebuilding is fine.
    
    def rebuild_arr():
        nonlocal current_arr
        current_arr = np.array(current_list, dtype=np.uint32)

    # Strategy: Hill Climbing with 1-for-2 swaps
    # 1. Pick a random word w from current_code.
    # 2. Remove it.
    # 3. Try to find 2 new words that fit.
    # 4. If found, add them.
    # 5. If not, try to find 1 word that fits (restore size).
    # 6. If not, revert.
    
    # To speed up finding new words, we can bias the search towards words close to the removed word?
    # Or just random.
    
    # Let's add a "kick" mechanism: if stuck, remove a random word and don't replace it immediately,
    # but try to fill the gap with 2 words.
    
    random.seed(42)
    
    iterations = 0
    stuck_counter = 0
    
    while time.time() < deadline - 2.0:
        iterations += 1
        
        # Pick a random word to remove
        if not current_list:
            break
            
        idx = random.randint(0, len(current_list) - 1)
        removed_word = current_list[idx]
        
        # Remove it
        current_list.pop(idx)
        current_set.discard(removed_word)
        rebuild_arr()
        
        # Try to find 2 new words
        found_pair = False
        # We try a few times to find a pair
        # To be efficient, we generate random words and check.
        # We can also try words that are at distance 6 from the removed word?
        # That might be a good heuristic.
        
        # Let's try random first.
        # We want to find w1, w2 such that:
        # 1. w1 not in current_set
        # 2. dist(w1, current_arr) >= 6
        # 3. w2 not in current_set
        # 4. dist(w2, current_arr) >= 6
        # 5. dist(w1, w2) >= 6
        
        # We can try to find w1 first.
        w1 = None
        for _ in range(50):
            cand = random.getrandbits(17)
            if cand in current_set:
                continue
            if cand == removed_word:
                continue # We want new words
            if can_add(cand, current_arr):
                w1 = cand
                break
        
        if w1 is not None:
            # Now find w2
            # w2 must be dist >= 6 from w1 and from current_arr
            # And not in current_set
            for _ in range(50):
                cand = random.getrandbits(17)
                if cand in current_set:
                    continue
                if cand == removed_word:
                    continue
                if cand == w1:
                    continue
                # Check dist to w1
                if bin(cand ^ w1).count('1') < 6:
                    continue
                # Check dist to current_arr
                if can_add(cand, current_arr):
                    w2 = cand
                    break
            
            if w1 is not None and 'w2' in locals() and w2 is not None:
                # Add both
                current_list.append(w1)
                current_set.add(w1)
                current_list.append(w2)
                current_set.add(w2)
                rebuild_arr()
                found_pair = True
                stuck_counter = 0
                if len(current_list) > best_size:
                    best_size = len(current_list)
                    best_list = current_list[:]
                    best_set = current_set.copy()
                    best_arr = current_arr.copy()
        
        if not found_pair:
            # Try to restore size with 1 word
            restored = False
            for _ in range(20):
                cand = random.getrandbits(17)
                if cand in current_set:
                    continue
                if cand == removed_word:
                    # Restoring the original word is allowed if we can't grow
                    pass
                if can_add(cand, current_arr):
                    current_list.append(cand)
                    current_set.add(cand)
                    rebuild_arr()
                    restored = True
                    break
            
            if not restored:
                # Revert
                current_list.append(removed_word)
                current_set.add(removed_word)
                rebuild_arr()
                stuck_counter += 1
            else:
                # We restored size, but didn't grow. 
                # This is a neutral move. 
                # We might want to count this as "stuck" if we don't grow for a while.
                # But it changes the local landscape.
                pass
        
        # If we are stuck for too long, try a more aggressive restart or change strategy
        # For now, just continue.
        
    return best_list
