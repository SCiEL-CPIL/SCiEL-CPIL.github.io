import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 17
    MIN_DIST = 6
    
    def time_left():
        return deadline - time.time()
    
    # Build the [17,8,6] quadratic residue code (256 words) as a strong seed.
    # The [18,9,6] extended QR code for p=17 has generator polynomial related to
    # the QR set. We construct it via the parity-check matrix approach.
    
    # For the [17,8,6] code: it's the shortened [18,9,6] code.
    # The [18,9,6] extended QR(17) code has a known structure.
    # 
    # Alternative: use the fact that the code is cyclic/quasi-cyclic.
    # For p=17, the QR set Q = {1, 2, 4, 8, 13, 15, 16, 9} (squares mod 17)
    # 
    # Let me construct the [17,8,6] code directly using a generator matrix.
    # The code is the set of all codewords c = m * G where G is 8x17.
    #
    # Actually, let me use a known construction: the [17,8,6] code can be built
    # from the [18,9,6] extended code by deleting one coordinate.
    #
    # A simpler approach: use the parity-check matrix. The [17,8,6] code has
    # a 9x17 parity-check matrix H such that c*H^T = 0 for all codewords c.
    #
    # For QR codes, H can be constructed from the QR set. Let me use a different
    # approach: build the code via the generator polynomial.
    #
    # For the binary QR code of length 17:
    # The generator polynomial g(x) of the [17,9,6] QR code divides x^17+1.
    # But we need the [17,8,6] subcode.
    #
    # Let me just use a known generator matrix for the [17,8,6] QR code.
    # I'll construct it using the circulant matrix approach.
    
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # The [17,8,6] code generator matrix can be built as follows:
    # Row i is the cyclic shift of a base vector.
    
    # Let me try a different approach: use the fact that we can construct
    # the code via its defining set. For the [17,8,6] code, the defining set
    # (zeros of the generator polynomial) has size 9.
    
    # Actually, the simplest reliable method: enumerate all 2^8 = 256 possible
    # information vectors and compute codewords using a known generator matrix.
    
    # Generator matrix for [17,8,6] QR code (from literature):
    # Let me construct it properly. The [18,9,6] extended QR(17) code has
    # generator polynomial g(x) = x^9 + x^8 + x^7 + x^5 + x^3 + x^2 + 1 (one possible form)
    # but I need to be careful.
    
    # Let me use a verified approach: build the code from scratch using
    # a systematic construction, then verify distance.
    
    # I'll use the following known generator matrix for the [17,8,6] code:
    # It's a (17,8) code with minimum distance 6.
    
    # Construction via parity check: The code is the null space of a 9x17 matrix.
    # For QR codes, H[i][j] = 1 if (j-i) mod 17 is in Q ∪ {0}, else 0,
    # with appropriate modifications.
    
    # Let me just hardcode a known valid generator matrix.
    # The [17,8,6] QR code generator matrix (rows are codewords for basis vectors):
    
    # Actually, let me construct it algorithmically:
    # The [17,8,6] code is the even-weight subcode of the [17,9,6] QR code? No.
    
    # Let me use a different strategy: start with a good random greedy code
    # but use a much better initialization and search strategy.
    
    # Strategy: 
    # 1. Build a large initial code using greedy with numpy acceleration
    # 2. Use local search with 1-for-1 and 1-for-2 moves
    # 3. Use simulated annealing-style acceptance
    
    rng = random.Random(42)
    
    # Fast distance computation using numpy
    def to_bits(word, n=N):
        return np.array([(word >> j) & 1 for j in range(n)], dtype=np.uint8)
    
    # Build initial code
    code = []
    code_set = set()
    code_arr = None  # (M, 17) uint8 array
    
    def can_add(candidate, code_arr):
        if code_arr is None or len(code_arr) == 0:
            return True
        cand = to_bits(candidate)
        dists = np.sum(code_arr ^ cand, axis=1)
        return bool(dists.min() >= MIN_DIST)
    
    def add_word(candidate, code, code_set, code_arr):
        code.append(candidate)
        code_set.add(candidate)
        if code_arr is None:
            return to_bits(candidate).reshape(1, -1)
        else:
            return np.vstack([code_arr, to_bits(candidate).reshape(1, -1)])
    
    # Greedy construction with structured candidate generation
    # Try all words of specific weights in a smart order
    
    # First, add 0
    code_arr = add_word(0, code, code_set, code_arr)
    
    # Generate candidates by weight, trying to fill in
    # For distance 6, we need weights that don't conflict
    
    # Use a priority-based approach: try words with moderate weight first
    weights_to_try = list(range(3, 15))
    # Shuffle for variety but keep some structure
    for attempt_round in range(200):
        if time_left() < 3.0:
            break
        
        # For each weight, try random candidates
        for w in weights_to_try:
            if time_left() < 3.0:
                break
            # Try up to 5000 candidates of this weight
            for _ in range(5000):
                if time_left() < 3.0:
                    break
                positions = rng.sample(range(N), w)
                candidate = 0
                for p in positions:
                    candidate |= (1 << p)
                if candidate in code_set:
                    continue
                if can_add(candidate, code_arr):
                    code_arr = add_word(candidate, code, code_set, code_arr)
    
    best_code = code[:]
    best_size = len(code)
    
    # Local search phase
    # Move types:
    # 1. Add a word if possible
    # 2. Remove one word, try to add two new words (net gain +1)
    # 3. Swap: remove one, add one different
    
    no_improve = 0
    max_no_improve = 50000
    
    while time_left() > 2.0:
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a word
            weight = rng.randint(3, 14)
            positions = rng.sample(range(N), weight)
            candidate = 0
            for p in positions:
                candidate |= (1 << p)
            if candidate not in code_set and can_add(candidate, code_arr):
                code_arr = add_word(candidate, code, code_set, code_arr)
                no_improve = 0
                if len(code) > best_size:
                    best_size = len(code)
                    best_code = code[:]
        elif move_type < 0.7:
            # 1-for-2 move: remove one word, try to add two
            if len(code) < 3:
                continue
            idx = rng.randint(0, len(code) - 1)
            old = code[idx]
            
            # Temporarily remove
            code.pop(idx)
            code_set.discard(old)
            code_arr = np.delete(code_arr, idx, axis=0)
            
            added_count = 0
            added_words = []
            for _ in range(100):
                weight = rng.randint(3, 14)
                positions = rng.sample(range(N), weight)
                candidate = 0
                for p in positions:
                    candidate |= (1 << p)
                if candidate in code_set:
                    continue
                if can_add(candidate, code_arr):
                    code_arr = add_word(candidate, code, code_set, code_arr)
                    added_words.append(candidate)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count >= 1:
                # Keep the change (net gain >= 0)
                no_improve = 0
                if len(code) > best_size:
                    best_size = len(code)
                    best_code = code[:]
            else:
                # Revert
                code_arr = add_word(old, code, code_set, code_arr)
                no_improve += 1
        else:
            # Swap: remove one, add one different
            if len(code) < 2:
                continue
            idx = rng.randint(0, len(code) - 1)
            old = code[idx]
            
            code.pop(idx)
            code_set.discard(old)
            code_arr = np.delete(code_arr, idx, axis=0)
            
            added = False
            for _ in range(50):
                weight = rng.randint(3, 14)
                positions = rng.sample(range(N), weight)
                candidate = 0
                for p in positions:
                    candidate |= (1 << p)
                if candidate in code_set:
                    continue
                if can_add(candidate, code_arr):
                    code_arr = add_word(candidate, code, code_set, code_arr)
                    added = True
                    break
            
            if not added:
                code_arr = add_word(old, code, code_set, code_arr)
                no_improve += 1
            else:
                no_improve = 0
        
        if no_improve > max_no_improve:
            # Reset stagnation counter and try a different strategy
            no_improve = 0
            # Perturb: randomly remove a few words and re-add
            if len(code) > 10:
                for _ in range(3):
                    idx = rng.randint(0, len(code) - 1)
                    old = code[idx]
                    code.pop(idx)
                    code_set.discard(old)
                    code_arr = np.delete(code_arr, idx, axis=0)
                    # Try to re-add or find alternatives
                    for _ in range(20):
                        weight = rng.randint(3, 14)
                        positions = rng.sample(range(N), weight)
                        candidate = 0
                        for p in positions:
                            candidate |= (1 << p)
                        if candidate not in code_set and can_add(candidate, code_arr):
                            code_arr = add_word(candidate, code, code_set, code_arr)
                            break
    
    # Final verification and return
    if best_code:
        # Verify
        valid = True
        for i in range(len(best_code)):
            if not valid:
                break
            for j in range(i + 1, len(best_code)):
                if (best_code[i] ^ best_code[j]).bit_count() < MIN_DIST:
                    valid = False
                    break
        if valid:
            return best_code
    
    if code:
        return code
    
    return [0]
