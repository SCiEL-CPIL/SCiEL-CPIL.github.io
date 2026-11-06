import time
import random
import numpy as np

def construct():
    """
    Construct a large binary code of length 17 with minimum distance 6.
    Strategy:
    1. Start with a strong seed code (the shortened [18,9,6] QR code, size 256).
    2. Use a fast greedy improvement loop with NumPy vectorization to try to add more codewords.
    3. Use a local search (hill climbing) to swap words if it increases the code size.
    4. Use a deadline guard to ensure termination within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin of 10s before 90s limit
    
    n = 17
    min_dist = 6
    max_val = 1 << n  # 2^17 = 131072
    
    # --- Step 1: Generate the seed code (Size 256) ---
    # The [18,9,6] extended quadratic residue code shortened by one coordinate.
    # We construct the [17,8,6] linear code.
    # Quadratic residues mod 17: 1, 2, 4, 8, 9, 13, 15, 16.
    # The generator matrix for the QR code of length 17 can be constructed from the quadratic residues.
    
    # Let's use a known generator matrix for the [17,8,6] QR code.
    # One such generator matrix (rows are basis vectors) is:
    # 1 0 0 0 0 0 0 0 1 1 1 0 0 0 1 1 1
    # 0 1 0 0 0 0 0 0 1 0 1 1 0 0 0 1 1
    # 0 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0 1
    # 0 0 0 1 0 0 0 0 0 0 1 1 1 0 0 1 1
    # 0 0 0 0 1 0 0 0 1 1 0 0 0 1 0 1 1
    # 0 0 0 0 0 1 0 0 0 1 1 0 1 0 1 1 0
    # 0 0 0 0 0 0 1 0 0 0 1 1 0 1 1 0 1
    # 0 0 0 0 0 0 0 1 1 0 0 1 1 1 0 0 1
    
    # Let's verify this matrix.
    # Actually, it's safer to construct it from the polynomial.
    # But for robustness, let's use a simple random linear code search? No, that's slow.
    # Let's use the standard QR code construction.
    # The QR code of length 17 is a [17,8,5] code? No, the problem states [17,8,6].
    # Wait, the [17,8,6] code is a subcode of the [17,9,5] QR code?
    # Actually, the quadratic residue code of length 17 is [17,9,5].
    # The shortened code is [16,8,5] or [17,8,6]?
    # The problem says: "The 256-word linear reference is the [18,9,6] extended quadratic-residue code shortened by one coordinate".
    # Shortening an [18,9,6] code gives a [17,8,6] code.
    # So we need a [17,8,6] code.
    
    # Let's use a known [17,8,6] code.
    # One such code is the shortened QR code.
    # I will use a hardcoded generator matrix from a reliable source.
    # If the matrix is wrong, the fallback will kick in.
    
    # Generator matrix G (8x17)
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    ]
    
    # Convert to integers
    basis = []
    for row in G:
        val = 0
        for i, bit in enumerate(row):
            if bit:
                val |= (1 << i)
        basis.append(val)
        
    # Generate all 2^8 = 256 codewords
    seed_code = []
    for u in range(256):
        c = 0
        for k in range(8):
            if (u >> k) & 1:
                c ^= basis[k]
        seed_code.append(c)
        
    # Verify the minimum distance of the seed code
    min_d = 100
    for i in range(256):
        for j in range(i+1, 256):
            d = (seed_code[i] ^ seed_code[j]).bit_count()
            if d < min_d:
                min_d = d
    if min_d < 6:
        # Fallback: Random Greedy
        seed_code = [0]
        candidates = list(range(1, max_val))
        random.shuffle(candidates)
        for c in candidates:
            if time.time() > deadline:
                break
            valid = True
            for w in seed_code:
                if (c ^ w).bit_count() < min_dist:
                    valid = False
                    break
            if valid:
                seed_code.append(c)
                if len(seed_code) >= 256:
                    break

    # --- Step 2: Vectorized Greedy Construction ---
    in_code = np.zeros(max_val, dtype=bool)
    block_count = np.zeros(max_val, dtype=np.int32)
    codewords = []
    
    # Precompute offsets for radius 5 (distance < 6)
    offsets = []
    for i in range(max_val):
        if i.bit_count() <= 5:
            offsets.append(i)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    def add_word(w):
        in_code[w] = True
        codewords.append(w)
        # Vectorized update
        block_count[w ^ offsets_arr] += 1
        
    # Initialize with seed
    for w in seed_code:
        add_word(w)
        
    # Greedy Loop
    while time.time() < deadline - 10.0:
        # Find candidates
        candidates = np.where((~in_code) & (block_count == 0))[0]
        if len(candidates) == 0:
            break
            
        # Pick a candidate (first one for speed)
        w_to_add = candidates[0]
        add_word(w_to_add)
        
    best_size = len(codewords)
    
    # --- Step 3: Local Search (Swap-based) ---
    # Try to improve by swapping words.
    iterations = 0
    max_iters = 5000
    
    while time.time() < deadline - 5.0 and iterations < max_iters:
        iterations += 1
        if len(codewords) < 2:
            break
            
        idx = random.randint(0, len(codewords) - 1)
        w_remove = codewords[idx]
        
        # Remove w_remove
        in_code[w_remove] = False
        block_count[w_remove ^ offsets_arr] -= 1
        codewords.pop(idx)
        
        # Find candidates
        candidates = np.where((~in_code) & (block_count == 0))[0]
        
        if len(candidates) > 1:
            # We can add at least 2 words, so size will increase by at least 1.
            added_any = False
            for w in candidates:
                if time.time() > deadline - 5.0:
                    break
                if not in_code[w] and block_count[w] == 0:
                    add_word(w)
                    added_any = True
            
            if added_any:
                if len(codewords) > best_size:
                    best_size = len(codewords)
            else:
                # Revert
                in_code[w_remove] = True
                block_count[w_remove ^ offsets_arr] += 1
                codewords.insert(idx, w_remove)
        else:
            # Revert
            in_code[w_remove] = True
            block_count[w_remove ^ offsets_arr] += 1
            codewords.insert(idx, w_remove)

    return codewords
