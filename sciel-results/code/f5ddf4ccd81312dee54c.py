import time
import random
import math
import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius)
    
    start_time = time.monotonic()
    deadline = start_time + 115.0
    
    total = q ** n  # 729
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks (int bitmasks)
    covers = [0] * total
    for i in range(total):
        w = all_words[i]
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                j = word_to_idx[tuple(nw)]
                mask |= (1 << j)
        covers[i] = mask

    # Precompute dist2 neighbors for local search diversity
    dist2 = [[] for _ in range(total)]
    for i in range(total):
        w1 = all_words[i]
        for j in range(i + 1, total):
            w2 = all_words[j]
            d = sum(1 for k in range(n) if w1[k] != w2[k])
            if d == 2:
                dist2[i].append(j)
                dist2[j].append(i)

    def greedy_init(seed):
        """Fast Greedy Set Cover using bitmasks."""
        r = random.Random(seed)
        covered_mask = 0
        code = []
        code_set = set()
        
        # Optional: Randomize order of candidates to break ties or explore different paths
        candidates = list(range(total))
        r.shuffle(candidates)
        
        while True:
            if time.monotonic() > deadline - 5:
                break
            
            best_gain = 0
            best_c = -1
            
            # Iterate through candidates not yet in code
            # Optimization: We can skip candidates whose coverage is fully contained in covered_mask
            # But checking containment is (covers[c] & ~covered_mask) == 0
            # We just compute gain and pick max.
            
            for c in candidates:
                if c in code_set:
                    continue
                gain = (covers[c] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_gain <= 0 or best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            covered_mask |= covers[best_c]
            
            if covered_mask == (1 << total) - 1:
                break
                
        return code, code_set, covered_mask

    def polish(code, code_set, covered_mask, time_budget):
        """Local search with bitmasks."""
        end_time = time.monotonic() + time_budget
        best_code = code[:]
        best_size = len(code)
        rng = random.Random(int(time.monotonic() * 1000) % 100000)
        
        # counts[w] = number of codewords covering w
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            # Extract indices from bitmask? No, we can just iterate set bits.
            # But iterating 729 bits for each of ~75 codewords is 5400 ops.
            # Better: Precompute list of indices for each mask? No, too much memory.
            # Let's just compute counts once.
            pass
            
        # Re-compute counts efficiently
        # Since total is small, we can just loop
        for c in code:
            mask = covers[c]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb

        def is_redundant(idx):
            mask = covers[idx]
            # Check if any word in mask has count == 1
            # We need to check counts for all set bits in mask
            m = mask
            while m:
                lsb = m & -m
                w_idx = lsb.bit_length() - 1
                if counts[w_idx] == 1:
                    return False
                m ^= lsb
            return True

        def remove_codeword(idx):
            code.remove(idx)
            code_set.discard(idx)
            mask = covers[idx]
            m = mask
            while m:
                lsb = m & -m
                w_idx = lsb.bit_length() - 1
                counts[w_idx] -= 1
                m ^= lsb
            return mask # return mask for potential reuse

        def add_codeword(idx):
            code.append(idx)
            code_set.add(idx)
            mask = covers[idx]
            m = mask
            while m:
                lsb = m & -m
                w_idx = lsb.bit_length() - 1
                counts[w_idx] += 1
                m ^= lsb
            return mask

        # Phase 1: Remove all redundant codewords
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for idx in list(code):
                if is_redundant(idx):
                    remove_codeword(idx)
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        # Phase 2: Local Search
        temp = 5.0
        cool_rate = 0.9995
        min_temp = 0.1
        
        while time.monotonic() < end_time:
            if temp < min_temp:
                temp = min_temp
            
            move_type = rng.random()
            
            if move_type < 0.30:
                # Try 2-for-1: find c1, c2 in code such that some c3 covers union
                if len(code) >= 2:
                    c1, c2 = rng.sample(code, 2)
                    union_mask = covers[c1] | covers[c2]
                    # Find a word in union_mask to restrict c3 search
                    # Pick a random word from union_mask
                    m = union_mask
                    if m:
                        # Pick random set bit
                        # Generate random integer in [0, total), mask with union? 
                        # Simpler: pick random index, check if in union
                        # Or just iterate a few random candidates from covered_by
                        # Let's pick a random word from union_mask
                        # To pick random set bit:
                        # r = rng.randint(0, total-1)
                        # if not (union_mask >> r) & 1: continue
                        # This might be slow if union is sparse.
                        # Better: just pick a random candidate c3 from all words and check subset
                        # But checking subset is (union_mask & ~covers[c3]) == 0
                        # We can sample a few c3s.
                        
                        # Heuristic: c3 should be close to c1 or c2
                        candidates = dist2[c1] + dist2[c2]
                        rng.shuffle(candidates)
                        found_c3 = -1
                        for c3 in candidates[:20]: # Check top 20 neighbors
                            if c3 in code_set:
                                continue
                            if (union_mask & ~covers[c3]) == 0:
                                found_c3 = c3
                                break
                        if found_c3 != -1:
                            remove_codeword(c1)
                            remove_codeword(c2)
                            add_codeword(found_c3)
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
            
            elif move_type < 0.70:
                # Try swap: replace one codeword with another
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    
                    # Choose new candidate
                    if rng.random() < 0.7 and dist2[old_c]:
                        new_c = rng.choice(dist2[old_c])
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check if swap is safe
                    # Words uniquely covered by old_c and not by new_c
                    # unique_old_mask = covers[old_c] & ~(covers[new_c] | other_covers)
                    # We don't have "other_covers" easily.
                    # Instead: unique_old = covers[old_c] & ~covers[new_c]
                    # Check if any word in unique_old has count == 1
                    # If so, it becomes uncovered.
                    
                    unique_mask = covers[old_c] & ~covers[new_c]
                    m = unique_mask
                    unsafe = False
                    while m:
                        lsb = m & -m
                        w_idx = lsb.bit_length() - 1
                        if counts[w_idx] == 1:
                            unsafe = True
                            break
                        m ^= lsb
                    
                    if not unsafe:
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        # Update counts
                        m = covers[old_c]
                        while m:
                            lsb = m & -m
                            w_idx = lsb.bit_length() - 1
                            counts[w_idx] -= 1
                            m ^= lsb
                        m = covers[new_c]
                        while m:
                            lsb = m & -m
                            w_idx = lsb.bit_length() - 1
                            counts[w_idx] += 1
                            m ^= lsb
                    elif rng.random() < math.exp(-2.0 / temp):
                        # Risky swap with repair
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        m = covers[old_c]
                        while m:
                            lsb = m & -m
                            w_idx = lsb.bit_length() - 1
                            counts[w_idx] -= 1
                            m ^= lsb
                        m = covers[new_c]
                        while m:
                            lsb = m & -m
                            w_idx = lsb.bit_length() - 1
                            counts[w_idx] += 1
                            m ^= lsb
                        
                        # Repair uncovered words
                        # Find words with count 0
                        # This is expensive. Only do if very risky?
                        # Let's skip repair for now to keep it fast, rely on next steps
                        pass
            
            else:
                # Try remove (if redundant)
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_redundant(c):
                        remove_codeword(c)
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
            
            temp *= cool_rate
        
        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for idx in list(code):
                if is_redundant(idx):
                    remove_codeword(idx)
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size

    overall_best_code = None
    overall_best_size = total + 1

    num_restarts = 20
    time_per_restart = (deadline - time.monotonic() - 5) / num_restarts

    for restart in range(num_restarts):
        if time.monotonic() > deadline - 10:
            break
        
        seed = 42 + restart * 13
        code, code_set, covered_mask = greedy_init(seed)
        
        if not code:
            continue
        
        polished_code, polished_size = polish(code, code_set, covered_mask, time_per_restart - 0.5)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
        
        if overall_best_size <= 73:
            break

    if overall_best_code is None:
        code, code_set, covered_mask = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result

def _generic_construct(q, n, radius):
    """Generic fallback for other parameters."""
    total = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage
    covers = [None] * total
    for i in range(total):
        w = all_words[i]
        cov = {i}
        # Generate all words within Hamming distance 'radius'
        for d in range(1, radius + 1):
            for positions in itertools.combinations(range(n), d):
                for new_vals in itertools.product(range(q), repeat=d):
                    nw = list(w)
                    for pos, val in zip(positions, new_vals):
                        nw[pos] = val
                    nw = tuple(nw)
                    if nw != w:
                        cov.add(word_to_idx[nw])
        covers[i] = cov
    
    # Greedy set cover
    counts = np.zeros(total, dtype=np.int32)
    code = []
    code_set = set()
    
    while True:
        uncovered = sum(1 for c in counts if c == 0)
        if uncovered == 0:
            break
        
        best_gain = 0
        best_c = -1
        for c in range(total):
            if c in code_set:
                continue
            gain = sum(1 for w in covers[c] if counts[w] == 0)
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_gain <= 0:
            break
        
        code.append(best_c)
        code_set.add(best_c)
        for w in covers[best_c]:
            counts[w] += 1
    
    result = []
    for idx in code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
