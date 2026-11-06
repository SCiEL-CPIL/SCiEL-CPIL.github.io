import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    N = 17
    MIN_DIST = 6

    def time_left():
        return deadline - time.time()

    # GF(2^8) arithmetic
    PRIM_POLY = 0x11D

    def gf_mul(a, b):
        res = 0
        for i in range(8):
            if (b >> i) & 1:
                res ^= a
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return res

    def gf_pow(a, p):
        res = 1
        for _ in range(p):
            res = gf_mul(res, a)
        return res

    # Find alpha (primitive element)
    alpha = None
    for a in range(2, 256):
        if gf_pow(a, 17) == 1 and gf_pow(a, 1) != 1:
            alpha = a
            break

    if alpha is None:
        return greedy_construct(deadline)

    # Quadratic residues mod 17
    Q = {1, 2, 4, 8, 9, 13, 15, 16}

    # Calculate generator polynomial g(x) = prod_{q in Q} (x + alpha^q)
    g_coeffs = [1]

    for q in Q:
        root = gf_pow(alpha, q)
        new_g = [0] * (len(g_coeffs) + 1)
        for i in range(len(g_coeffs)):
            new_g[i] ^= gf_mul(g_coeffs[i], root)
            new_g[i+1] ^= g_coeffs[i]
        g_coeffs = new_g

    # Convert to binary polynomial coefficients
    bin_g = [1 if c == 1 else 0 for c in g_coeffs]
    while len(bin_g) < 9:
        bin_g.append(0)

    # Convert bin_g to integer for fast shifting
    g_int = 0
    for i in range(9):
        if bin_g[i]:
            g_int |= (1 << i)

    # Generate all 512 codewords of the [17,9,6] code
    base_code = []

    for m in range(512):
        c = 0
        for i in range(9):
            if (m >> i) & 1:
                c ^= (g_int << i)

        # Check even weight
        if c.bit_count() % 2 == 0:
            base_code.append(c)

    code = base_code[:]
    code_set = set(code)
    
    # Convert code to numpy array for fast distance checks
    # shape: (num_codewords, N)
    def code_to_arr(codes):
        return np.array([[(c >> i) & 1 for i in range(N)] for c in codes], dtype=np.uint8)

    code_arr = code_to_arr(code)

    rng = random.Random(42)

    def check_compatibility(candidates_arr, code_arr):
        """
        candidates_arr: (M, N) uint8
        code_arr: (K, N) uint8
        Returns: boolean array of shape (M,) where True means candidate is compatible with ALL codewords.
        """
        if len(candidates_arr) == 0:
            return np.array([], dtype=bool)
        
        # Compute distances between each candidate and each codeword
        # We want min distance >= MIN_DIST
        # dist = sum(cand ^ code)
        # Using broadcasting: candidates[:, np.newaxis, :] ^ code[np.newaxis, :, :]
        # This creates (M, K, N) array. For M=1000, K=256, N=17, this is ~4.3M elements. Fine.
        
        # To save memory, we can do it in chunks if M is huge, but 1000-2000 is fine.
        diffs = candidates_arr[:, np.newaxis, :] ^ code_arr[np.newaxis, :, :]
        dists = diffs.sum(axis=2)
        min_dists = dists.min(axis=1)
        return min_dists >= MIN_DIST

    def check_mutual_compatibility(candidates_arr):
        """
        candidates_arr: (M, N) uint8
        Returns: boolean mask of valid candidates in a greedy clique.
        Greedy: iterate, if compatible with all previously selected, select it.
        """
        if len(candidates_arr) == 0:
            return np.array([], dtype=bool)
        
        selected_indices = []
        selected_arr = []
        
        # Shuffle to make it non-deterministic and potentially find larger cliques
        indices = np.arange(len(candidates_arr))
        rng.shuffle(indices)
        
        for idx in indices:
            cand = candidates_arr[idx]
            if not selected_arr:
                selected_indices.append(idx)
                selected_arr.append(cand)
                continue
            
            # Check against all selected
            sel_arr = np.array(selected_arr, dtype=np.uint8)
            diffs = sel_arr ^ cand
            dists = diffs.sum(axis=1)
            if dists.min() >= MIN_DIST:
                selected_indices.append(idx)
                selected_arr.append(cand)
        
        mask = np.zeros(len(candidates_arr), dtype=bool)
        for idx in selected_indices:
            mask[idx] = True
        return mask

    # Phase 1: Batch Greedy Extension
    # We will try to add a large batch of candidates at once.
    
    # Generate a pool of candidates
    # Focus on weights 6-11, as these are likely to be in optimal codes.
    # We generate multiple pools and pick the best extension.
    
    best_extension = []
    best_extension_arr = None
    
    for batch_iter in range(50): # 50 attempts
        if time_left() < 10.0:
            break
            
        # Generate random candidates
        num_cands = 2000
        cands = []
        for _ in range(num_cands):
            w = rng.randint(6, 11)
            positions = rng.sample(range(N), w)
            c = 0
            for p in positions:
                c |= (1 << p)
            if c not in code_set:
                cands.append(c)
        
        if not cands:
            continue
            
        cand_arr = code_to_arr(cands)
        
        # Filter against current code
        compat_mask = check_compatibility(cand_arr, code_arr)
        valid_cands = cand_arr[compat_mask]
        valid_cands_int = [cands[i] for i in np.where(compat_mask)[0]]
        
        if len(valid_cands) == 0:
            continue
            
        # Find greedy clique among valid candidates
        clique_mask = check_mutual_compatibility(valid_cands)
        clique_cands_int = [valid_cands_int[i] for i in np.where(clique_mask)[0]]
        
        if len(clique_cands_int) > len(best_extension):
            best_extension = clique_cands_int
            best_extension_arr = code_to_arr(clique_cands_int)
            
        # If we found a good extension, apply it and restart the batch search?
        # Or just accumulate?
        # Accumulating is risky if the first batch was bad.
        # Let's apply the best found so far at the end of the loop, or if it's very good.
        # Actually, let's apply it immediately if it's > 0, then re-generate pool?
        # No, the code changes, so the compatibility changes.
        # We should probably do one big batch, apply, then do local search.
        
    # Apply the best extension found in Phase 1
    if best_extension:
        for c in best_extension:
            code.append(c)
            code_set.add(c)
        code_arr = code_to_arr(code)

    # Phase 2: Local Search (1-for-2 Swap)
    # Try to remove one word and add two or more.
    # We iterate a few times.
    
    improved = True
    swap_iterations = 0
    while improved and time_left() > 15.0 and swap_iterations < 50:
        improved = False
        swap_iterations += 1
        
        # Try a random subset of codewords as removal candidates to save time
        # If we try all 256+, it's slow.
        # Let's try 20 random codewords per iteration.
        indices_to_try = rng.sample(range(len(code)), min(20, len(code)))
        
        for i in indices_to_try:
            if time_left() < 15.0:
                break
                
            removed = code[i]
            code_without = code[:i] + code[i+1:]
            code_without_set = set(code_without)
            code_without_arr = code_to_arr(code_without)
            
            # Generate candidates biased towards the removed word's neighborhood
            # We want words that are "close" to removed but far from others.
            # Actually, just random candidates from the pool again.
            
            # Let's use the same batch logic but for the "reduced" code.
            # Generate 500 candidates
            cands = []
            for _ in range(500):
                # Mix: some near removed, some random
                if rng.random() < 0.5:
                    # Near removed: flip 1-3 bits
                    flips = rng.randint(1, 3)
                    positions = rng.sample(range(N), flips)
                    cand = removed
                    for p in positions:
                        cand ^= (1 << p)
                else:
                    w = rng.randint(6, 11)
                    positions = rng.sample(range(N), w)
                    cand = 0
                    for p in positions:
                        cand |= (1 << p)
                        
                if cand not in code_without_set:
                    cands.append(cand)
            
            if not cands:
                continue
                
            cand_arr = code_to_arr(cands)
            compat_mask = check_compatibility(cand_arr, code_without_arr)
            valid_cands = cand_arr[compat_mask]
            valid_cands_int = [cands[i] for i in np.where(compat_mask)[0]]
            
            if len(valid_cands) == 0:
                continue
                
            clique_mask = check_mutual_compatibility(valid_cands)
            clique_cands_int = [valid_cands_int[i] for i in np.where(clique_mask)[0]]
            
            # Accept if we can add at least 2 words
            if len(clique_cands_int) >= 2:
                # Apply the swap
                code = code_without
                code_set = code_without_set
                for c in clique_cands_int:
                    code.append(c)
                    code_set.add(c)
                code_arr = code_to_arr(code)
                improved = True
                break # Restart the while loop with new code

    # Phase 3: Final Greedy Polish
    # Try to add any remaining single candidates that might fit.
    # Use a larger pool.
    
    for _ in range(100):
        if time_left() < 5.0:
            break
        
        cands = []
        for _ in range(1000):
            w = rng.randint(6, 12)
            positions = rng.sample(range(N), w)
            cand = 0
            for p in positions:
                cand |= (1 << p)
            if cand not in code_set:
                cands.append(cand)
                
        if not cands:
            continue
            
        cand_arr = code_to_arr(cands)
        compat_mask = check_compatibility(cand_arr, code_arr)
        valid_cands_int = [cands[i] for i in np.where(compat_mask)[0]]
        
        if not valid_cands_int:
            continue
            
        # Greedy add these valid candidates one by one
        # They are already compatible with the current code, but not necessarily with each other.
        # So we do a greedy clique selection again.
        valid_cands_arr = code_to_arr(valid_cands_int)
        clique_mask = check_mutual_compatibility(valid_cands_arr)
        added = [valid_cands_int[i] for i in np.where(clique_mask)[0]]
        
        if added:
            for c in added:
                code.append(c)
                code_set.add(c)
            code_arr = code_to_arr(code)

    return code

def greedy_construct(deadline):
    code = [0]
    code_set = {0}
    rng = random.Random(42)
    while time.time() < deadline:
        w = rng.randint(3, 14)
        positions = rng.sample(range(17), w)
        c = 0
        for p in positions:
            c |= (1 << p)
        if c not in code_set:
            ok = True
            for w2 in code:
                if (c ^ w2).bit_count() < 6:
                    ok = False
                    break
            if ok:
                code.append(c)
                code_set.add(c)
    return code
