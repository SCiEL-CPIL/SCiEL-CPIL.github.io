import itertools
import numpy as np
import time

def construct(q, n, radius):
    # For this specific task, we assume q=3, n=7, radius=1
    # But the function signature is general.
    # If parameters differ, we might need a fallback.
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies we only need to handle the specific case or a general case.
    # Given the time limit and complexity, I will specialize for q=3, n=7, r=1
    # and provide a fallback for other cases (e.g., greedy only).
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Simple Greedy
        return _greedy_construct(q, n, radius)

    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    # Map word index to word tuple for later conversion
    # Index is just the integer representation in base q
    
    # Precompute balls as bitsets
    # Ball of i: i and all words at distance 1
    balls = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    # Find index of new_w
                    # Since we generated in product order, index can be computed
                    # Or use a dictionary
                    pass
        
    # Using dictionary for index lookup is safer/faster for arbitrary q,n
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    balls = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        for pos in range(n):
            for delta in [1, -1]:
                new_val = (w[pos] + delta) % q
                new_w = w.copy()
                new_w[pos] = new_val
                j = word_to_idx[tuple(new_w)]
                mask |= (1 << j)
        balls[i] = mask

    all_mask = (1 << N) - 1
    
    # 1. Greedy Initialization
    uncovered = all_mask
    code = []
    
    # Sort words by some heuristic? Or just iterate.
    # To make greedy deterministic and fast, we can just pick the best one each time.
    # Optimization: Precompute "potential" coverage?
    # Simple greedy:
    while uncovered:
        best_i = -1
        best_cov = 0
        # Iterate all words
        for i in range(N):
            cov = (balls[i] & uncovered).bit_count()
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        uncovered &= ~balls[best_i]
        
    # 2. Local Search
    # Data structures for fast updates
    cover_count = np.zeros(N, dtype=np.int32)
    for i in code:
        # Add ball i to cover_count
        # Iterate bits of balls[i]
        mask = balls[i]
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            mask ^= lsb
            
    mask_count_1 = 0
    mask_count_2 = 0
    for i in range(N):
        if cover_count[i] == 1:
            mask_count_1 |= (1 << i)
        elif cover_count[i] == 2:
            mask_count_2 |= (1 << i)

    improved = True
    while improved:
        improved = False
        
        # Pass 1: Single Removal
        # Check if any codeword is redundant
        # Redundant if B_i & mask_count_1 == 0
        for idx_in_code, i in enumerate(code):
            if (balls[i] & mask_count_1) == 0:
                # Remove i
                code.pop(idx_in_code)
                # Update cover_count
                mask = balls[i]
                while mask:
                    lsb = mask & -mask
                    w_idx = lsb.bit_length() - 1
                    cover_count[w_idx] -= 1
                    mask ^= lsb
                # Update masks
                # Recalculate mask_count_1 and mask_count_2?
                # It's O(N) to recalculate. N=2187.
                # We can update incrementally?
                # Words that changed count are those in B_i.
                # For each w in B_i:
                #   old_count = cover_count[w] + 1
                #   new_count = cover_count[w]
                #   Update masks based on old/new
                mask = balls[i]
                while mask:
                    lsb = mask & -mask
                    w_idx = lsb.bit_length() - 1
                    new_c = cover_count[w_idx]
                    old_c = new_c + 1
                    if old_c == 1:
                        mask_count_1 &= ~(1 << w_idx)
                    elif old_c == 2:
                        mask_count_2 &= ~(1 << w_idx)
                    if new_c == 1:
                        mask_count_1 |= (1 << w_idx)
                    elif new_c == 2:
                        mask_count_2 |= (1 << w_idx)
                    mask ^= lsb
                improved = True
                break # Restart loop after modification
                
        if improved:
            continue

        # Pass 2: Pair Replacement
        # Check pairs (i, j)
        len_code = len(code)
        for i_idx in range(len_code):
            i = code[i_idx]
            for j_idx in range(i_idx + 1, len_code):
                j = code[j_idx]
                
                # Calculate U
                # U = (B_i \ B_j & mask_count_1) | (B_j \ B_i & mask_count_1) | (B_i & B_j & mask_count_2)
                Bi = balls[i]
                Bj = balls[j]
                
                Bi_only = Bi & ~Bj
                Bj_only = Bj & ~Bi
                Bi_Bj = Bi & Bj
                
                U = (Bi_only & mask_count_1) | (Bj_only & mask_count_1) | (Bi_Bj & mask_count_2)
                
                if U == 0:
                    # Both redundant
                    # Remove both
                    # Note: order of removal matters for indices
                    # Remove j first (higher index)
                    code.pop(j_idx)
                    # Update counts for j
                    mask = Bj
                    while mask:
                        lsb = mask & -mask
                        w_idx = lsb.bit_length() - 1
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        mask ^= lsb
                        
                    # Remove i
                    code.pop(i_idx)
                    mask = Bi
                    while mask:
                        lsb = mask & -mask
                        w_idx = lsb.bit_length() - 1
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        mask ^= lsb
                        
                    improved = True
                    break
                    
                # Check if U is covered by some k
                # Pick a word w in U
                lsb = U & -U
                w_idx = lsb.bit_length() - 1
                w = words[w_idx]
                
                # Candidates k are in ball(w)
                # Generate ball(w) indices
                candidates = []
                # w itself
                candidates.append(word_to_idx[tuple(w)])
                for pos in range(n):
                    for delta in [1, -1]:
                        new_val = (w[pos] + delta) % q
                        new_w = w.copy()
                        new_w[pos] = new_val
                        candidates.append(word_to_idx[tuple(new_w)])
                        
                found_k = -1
                for k in candidates:
                    if k == i or k == j:
                        continue
                    if (U & ~balls[k]) == 0:
                        found_k = k
                        break
                        
                if found_k != -1:
                    # Replace i, j with k
                    # Remove j
                    code.pop(j_idx)
                    mask = Bj
                    while mask:
                        lsb = mask & -mask
                        w_idx = lsb.bit_length() - 1
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        mask ^= lsb
                        
                    # Remove i
                    code.pop(i_idx)
                    mask = Bi
                    while mask:
                        lsb = mask & -mask
                        w_idx = lsb.bit_length() - 1
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] -= 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        mask ^= lsb
                        
                    # Add k
                    code.append(found_k)
                    mask = balls[found_k]
                    while mask:
                        lsb = mask & -mask
                        w_idx = lsb.bit_length() - 1
                        old_c = cover_count[w_idx]
                        cover_count[w_idx] += 1
                        new_c = cover_count[w_idx]
                        if old_c == 1: mask_count_1 &= ~(1 << w_idx)
                        elif old_c == 2: mask_count_2 &= ~(1 << w_idx)
                        if new_c == 1: mask_count_1 |= (1 << w_idx)
                        elif new_c == 2: mask_count_2 |= (1 << w_idx)
                        mask ^= lsb
                        
                    improved = True
                    break
                    
            if improved:
                break
                
    return [words[i] for i in code]

def _greedy_construct(q, n, radius):
    # Fallback for other parameters
    N = q ** n
    words = [list(t) for t in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    balls = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        # Generate neighbors within radius
        # For general radius, use BFS or recursive generation
        # Here we assume small radius
        def gen_neighbors(pos, current, dist):
            if dist == 0:
                yield current
                return
            if pos == n:
                return
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                new_val = (current[pos] + delta) % q
                if new_val != current[pos]:
                    new_w = current.copy()
                    new_w[pos] = new_val
                    yield from gen_neighbors(pos + 1, new_w, dist - 1)
        # This generation is inefficient for general radius.
        # For the specific task, we don't use this.
        # For fallback, just use simple greedy without bitsets if N is small, or random.
        pass
        
    # Simple Greedy without bitsets for fallback
    uncovered = set(range(N))
    code = []
    while uncovered:
        best_i = -1
        best_cov = 0
        for i in range(N):
            w = words[i]
            cov = 0
            for u_idx in uncovered:
                u = words[u_idx]
                if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                    cov += 1
            if cov > best_cov:
                best_cov = cov
                best_i = i
        code.append(best_i)
        # Remove covered
        w = words[best_i]
        to_remove = []
        for u_idx in uncovered:
            u = words[u_idx]
            if sum(1 for a, b in zip(w, u) if a != b) <= radius:
                to_remove.append(u_idx)
        uncovered.difference_update(to_remove)
    return [words[i] for i in code]
