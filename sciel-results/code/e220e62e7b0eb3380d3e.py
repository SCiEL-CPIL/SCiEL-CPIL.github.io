import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for 90s limit

    N = 17
    MIN_DIST = 6

    # Precompute masks for all distances < MIN_DIST
    # masks_by_dist[d] contains all bitmasks with exactly d bits set
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)

    # 1. Construct the [17, 9, 6] Quadratic Residue Code (256 words)
    # The generator is derived from the quadratic residues modulo 17.
    # Residues: {1, 2, 4, 8, 9, 13, 15, 16}
    # The polynomial is g(x) = prod_{q in Q} (x - alpha^q) over GF(2^8)
    # But for the binary QR code, we can use the known generator matrix or polynomial.
    # The generator polynomial for the [17, 9, 6] code is:
    # g(x) = x^8 + x^5 + x^4 + x^2 + 1
    # Let's verify this.
    # Actually, let's use the cyclic shift method which is robust.
    # The generator vector is the coefficients of g(x).
    # g(x) = 1 + x^2 + x^4 + x^5 + x^8
    # So the bits are at positions 0, 2, 4, 5, 8.
    
    gen_poly = 0
    for pos in [0, 2, 4, 5, 8]:
        gen_poly |= (1 << pos)
        
    # Generate all 2^9 codewords
    words = []
    for mask in range(2**9):
        w = 0
        for i in range(9):
            if (mask >> i) & 1:
                w ^= (gen_poly << i)
        # The code is cyclic of length 17, so we need to reduce mod x^17 - 1
        # But the generator polynomial degree is 8, and we multiply by x^i for i<9.
        # The max degree is 8+8=16, which fits in 17 bits. So no reduction needed.
        words.append(w)
        
    # Verify minimum distance for linear code
    min_w = N
    for w in words:
        if w == 0:
            continue
        weight = bin(w).count('1')
        if weight < min_w:
            min_w = weight
        if min_w < MIN_DIST:
            break
    
    if min_w < MIN_DIST:
        # Fallback: if the QR code construction is wrong, use a greedy approach
        # This shouldn't happen if the poly is correct, but just in case.
        # We can try to find a valid code or return a small one.
        # For now, let's just return the words we have, even if invalid? No, must be valid.
        # Let's use the RM code fallback.
        rm_words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                rm_words.append(word)
        code = list(rm_words)
        random.seed(42)
        # Simple greedy extension
        while time.time() < deadline - 5.0:
            added = False
            for _ in range(100):
                cand = random.randint(0, 2**N - 1)
                valid = all((cand ^ w).bit_count() >= MIN_DIST for w in code)
                if valid:
                    code.append(cand)
                    added = True
                    break
            if not added:
                break
        return code

    # We have a valid 256-word linear code.
    # Strategy: Use the "blocked" array technique from inspiration 2 for fast greedy addition.
    # This is much faster than checking against all existing words one by one.
    
    code = list(words)
    code_set = set(code)
    
    # Initialize blocked array
    # blocked[x] = 1 if x is within distance < MIN_DIST of any codeword in code
    blocked = bytearray(1 << N)
    
    # Function to add a word and update blocked
    def add_word(w):
        if blocked[w]:
            return False
        code.append(w)
        code_set.add(w)
        # Block all neighbors
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1
        return True
        
    # Block all initial codewords
    for w in code:
        for d in range(MIN_DIST):
            for mask in masks_by_dist[d]:
                blocked[w ^ mask] = 1
                
    # Phase 1: Greedy addition
    # We can iterate through all possible words, or use random.
    # Iterating through all 2^17 words is 131072 iterations.
    # Each check is O(1) if we just check blocked[w].
    # But we need to be careful: if we just iterate 0..2^17-1, we might miss words that become unblocked?
    # No, blocked only becomes more blocked, never less.
    # So a single pass through all words is sufficient for greedy.
    
    # However, we might want to be smarter.
    # Let's try a single pass first.
    for w in range(1, 1 << N):
        if time.time() > deadline - 10.0:
            break
        if not blocked[w]:
            add_word(w)
            
    # Phase 2: Local search / Swapping
    # Now we are stuck. We need to try to swap words to free up space.
    # This is harder with the blocked array because removing a word doesn't automatically unblock its neighbors
    # unless no other codeword blocks them.
    # So for swapping, we might need to recompute blocked or use a more complex check.
    
    # Alternative: Use the numpy-based check from the original program for the swap phase.
    # Convert to numpy array
    code_arr = np.array(code, dtype=np.int32)
    
    def check_candidate(cand, code_list, code_np):
        """Check if cand is valid (min distance >= MIN_DIST to all in code_list)."""
        xor_vals = code_np ^ cand
        # Compute popcount using bit operations
        total = np.zeros(len(code_np), dtype=np.int32)
        for b in range(N):
            total += ((xor_vals >> b) & 1).astype(np.int32)
        return total.min() >= MIN_DIST
        
    def check_candidate_excluding(cand, code_list, code_np, exclude_idx):
        """Check if cand is valid, excluding code_list[exclude_idx]."""
        xor_vals = code_np ^ cand
        total = np.zeros(len(code_np), dtype=np.int32)
        for b in range(N):
            total += ((xor_vals >> b) & 1).astype(np.int32)
        # Exclude the one at exclude_idx
        if len(total) > 1:
            saved = total[exclude_idx]
            total[exclude_idx] = N  # Max possible distance is N
            result = total.min() >= MIN_DIST
            total[exclude_idx] = saved
            return result
        else:
            return True

    random.seed(12345)
    
    # Try to add more words using random candidates and local search
    # We can use the blocked array for quick rejection, but for acceptance we need to be sure.
    # Actually, if blocked[w] is 0, it is guaranteed to be valid.
    # So we can just check blocked[w] for random candidates too.
    
    # Let's try random candidates first
    attempts = 0
    while time.time() < deadline - 5.0 and attempts < 100000:
        attempts += 1
        # Generate random candidate
        cand = random.randint(0, 2**N - 1)
        if not blocked[cand]:
            add_word(cand)
            attempts = 0 # Reset counter on success
            
    # If we still have time, try swapping
    # Pick a random word, remove it (conceptually), see if we can add two new words?
    # Or just replace it with a better one that allows more additions?
    # This is complex. Let's try a simple swap: replace word i with word j, then try to add.
    
    # Actually, a better strategy for escaping local optima in packing problems is often "1-move" or "2-move".
    # Since we are using a blocked array, removing a word requires unblocking its neighbors IF they are not blocked by others.
    # This is expensive to track.
    
    # Let's stick to the numpy-based check for swaps if we have time.
    # But first, let's see if the greedy pass added anything.
    
    # If we are still at 256, we might need the swap phase.
    # Let's try a few swaps.
    
    swap_attempts = 0
    max_swap_attempts = 5000
    
    while time.time() < deadline - 2.0 and swap_attempts < max_swap_attempts:
        swap_attempts += 1
        
        # Pick a random word to potentially replace
        idx = random.randint(0, len(code) - 1)
        old_word = code[idx]
        
        # Generate a candidate replacement
        # Flip a few bits
        num_flips = random.randint(1, 4)
        flips = random.sample(range(N), num_flips)
        cand = old_word
        for pos in flips:
            cand ^= (1 << pos)
            
        if cand == old_word:
            continue
            
        # Check if cand is valid when old_word is removed
        # We need to check against all other words
        # Use numpy check
        if check_candidate_excluding(cand, code, code_arr, idx):
            # Replace old_word with cand
            code[idx] = cand
            code_arr[idx] = cand
            code_set.discard(old_word)
            code_set.add(cand)
            
            # Now try to add a new word
            # We can try to add a random word
            for _ in range(10):
                new_cand = random.randint(0, 2**N - 1)
                if check_candidate(new_cand, code, code_arr):
                    code.append(new_cand)
                    code_arr = np.append(code_arr, new_cand)
                    code_set.add(new_cand)
                    break
            # Note: The blocked array is now out of sync. 
            # We should probably stop using it or rebuild it if we do many swaps.
            # For simplicity, let's stop using blocked array after first swap.
            # Or just break out of swap loop if we want to be safe.
            # Let's continue swapping but be aware blocked is stale.
            # Actually, if we modify code, blocked is invalid.
            # Let's just use the numpy check for all subsequent operations.
            # But the numpy check is slower.
            # Let's just break after a few successful swaps to avoid complexity.
            if swap_attempts > 10:
                break

    return code
