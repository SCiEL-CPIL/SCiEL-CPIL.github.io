import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    N = q ** n
    
    if radius != 1:
        raise NotImplementedError("Radius > 1 not optimized")

    sphere_size = 1 + n * (q - 1)
    neighbors = np.empty((N, sphere_size), dtype=int)
    
    # Precompute neighbors
    for i in range(N):
        idx = 0
        neighbors[i, idx] = i
        idx += 1
        val = i
        for pos in range(n):
            orig_val = val % q
            val //= q
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[i, idx] = i + (new_val - orig_val) * (q ** pos)
                idx += 1

    best_C = None
    best_len = N + 1
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Partition-based initialization
    # Split n coordinates into two parts: n1 and n2 such that q**n2 is small (e.g., 3^2=9)
    # For q=3, n=7, we can take n2=2, n1=5.
    # Plane index is determined by first n1 coords.
    # Varying coords are last n2 coords.
    # For each plane, we need a covering set of size K_q(n2, 1).
    # For q=3, n2=2, K=3.
    # We use the "diagonal" set {(0,0), (1,1), (2,2)} shifted by the plane offset?
    # Actually, the set {(x, x) | x in {0,1,2}} covers the 3x3 grid.
    # So for plane with fixed prefix P, the words are P concatenated with (0,0), (1,1), (2,2).
    
    n2 = 2
    n1 = n - n2
    plane_size = q ** n2
    num_planes = q ** n1
    
    initial_C = []
    for p in range(num_planes):
        # Decode plane index p into prefix
        prefix = []
        val = p
        for _ in range(n1):
            prefix.append(val % q)
            val //= q
        prefix.reverse()
        
        # Generate local covering set for the n2 part
        # For q=3, n2=2, use {(0,0), (1,1), (2,2)}
        # This is a valid covering for 3x2 with radius 1.
        local_cover = []
        for x in range(q):
            # Ensure we cover the 3x3 grid. (x,x) works for q=3.
            # For general q, we might need a different set, but task is fixed q=3, n=7.
            local_cover.append([x, x])
            
        for lc in local_cover:
            full_word = prefix + lc
            # Encode full_word to integer
            code = 0
            for i, v in enumerate(full_word):
                code += v * (q ** i)
            initial_C.append(code)
            
    # Deduplicate just in case (though planes are disjoint, so no dups)
    initial_C = list(set(initial_C))
    
    # Local Search: Perturb and Prune
    current_C = initial_C.copy()
    
    def prune(C):
        # Remove redundant codewords
        cover_count = np.zeros(N, dtype=int)
        for c in C:
            cover_count[neighbors[c]] += 1
            
        in_C = np.zeros(N, dtype=bool)
        for c in C:
            in_C[c] = True
            
        changed = True
        while changed:
            changed = False
            for i in range(len(C)):
                c = C[i]
                nb = neighbors[c]
                cover_count[nb] -= 1
                if np.any(cover_count[nb] == 0):
                    cover_count[nb] += 1
                else:
                    C.pop(i)
                    in_C[c] = False
                    changed = True
                    break
        return C

    current_C = prune(current_C)
    
    if len(current_C) < best_len:
        best_len = len(current_C)
        best_C = current_C.copy()
        
    while time.time() - start_time < time_limit:
        # Perturb: Pick a random codeword and replace with a random neighbor or random word
        if not current_C:
            break
            
        idx = random.randint(0, len(current_C) - 1)
        old_c = current_C[idx]
        
        # Try replacing with a random word from the space
        # Or maybe a word that is "close" to improve coverage?
        # Random word is a good basin hopper.
        new_c = random.randint(0, N - 1)
        
        if new_c == old_c:
            continue
            
        current_C[idx] = new_c
        current_C = prune(current_C)
        
        if len(current_C) < best_len:
            best_len = len(current_C)
            best_C = current_C.copy()
            
        # Optional: Try to swap with a word that covers many uncovered/low-count areas?
        # The simple perturb+prune is fast.
        
    # Final optimization: try to remove more codewords
    if best_C:
        best_C = prune(best_C)
        
    result_words = []
    for idx in best_C:
        word = []
        val = idx
        for _ in range(n):
            word.append(val % q)
            val //= q
        result_words.append(word)
        
    return result_words
