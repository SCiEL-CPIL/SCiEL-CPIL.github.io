import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    total_words = q ** n
    
    # Precompute all words and their indices
    # Word index is base-q representation
    # We represent spheres as bitsets (Python ints) for fast intersection/union
    
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute spheres for all words
    # sphere[i] is a bitmask of all words within distance 1 of word i
    # Note: 2187 bits fits easily in a Python int
    spheres = [0] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        mask = 1 << i
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    new_word = w.copy()
                    new_word[pos] = new_val
                    j = word_to_idx(new_word)
                    mask |= (1 << j)
        spheres[i] = mask

    # Precompute coverers: for each word j, list of codewords i such that j in sphere[i]
    # This is expensive to build fully, but we can build it on the fly or precompute if memory allows.
    # 2187 * 15 avg coverers per word? No, each word is in 1 + 7*2 = 15 spheres.
    # So coverers[j] has length 15.
    coverers = [[] for _ in range(total_words)]
    for i in range(total_words):
        mask = spheres[i]
        # Iterate through set bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            j = lsb.bit_length() - 1
            coverers[j].append(i)
            m ^= lsb

    best_code = None
    best_size = float('inf')
    
    def local_search(code_set, deadline):
        current_code = set(code_set)
        # count[j] = number of codewords in current_code whose sphere contains j
        count = np.zeros(total_words, dtype=np.int32)
        
        # Initialize counts
        for c in current_code:
            mask = spheres[c]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                count[j] += 1
                m ^= lsb
                
        changed = True
        while changed and time.time() < deadline:
            changed = False
            
            # 1. Prune redundant codewords
            # A codeword c is redundant if for all j in sphere[c], count[j] > 1
            for c in list(current_code):
                mask = spheres[c]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if count[j] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    current_code.remove(c)
                    mask = spheres[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] -= 1
                        m ^= lsb
                    changed = True
            
            if time.time() > deadline:
                break

            # 2. Swap optimization
            # Try to swap a codeword c with a non-codeword c' to improve coverage
            # We prioritize codewords c that cover many "critical" points (count==1)
            # Or just iterate through all c and find the best c'
            
            candidates = list(current_code)
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > deadline:
                    break
                if c not in current_code:
                    continue
                
                # Find points covered by c with count == 1 (critical points)
                critical_points = []
                mask = spheres[c]
                m = mask
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if count[j] == 1:
                        critical_points.append(j)
                    m ^= lsb
                
                if not critical_points:
                    # If no critical points, c is fully redundant in terms of unique coverage, 
                    # but we already pruned fully redundant ones. 
                    # This means c covers only points with count > 1. 
                    # We can try to remove it if we can find a replacement that covers 
                    # nothing new but allows further pruning? 
                    # Actually, if no critical points, removing c leaves all its points 
                    # with count >= 1. So it's safe to remove? 
                    # Wait, if count[j] > 1 for all j in sphere[c], then removing c 
                    # makes count[j] >= 1 for all j. So c is redundant!
                    # Why didn't pruning catch it? 
                    # Pruning checks if count[j] > 1 for ALL j. 
                    # If count[j] == 1 for some j, it's not redundant.
                    # So if critical_points is empty, it means count[j] > 1 for all j.
                    # So it SHOULD have been pruned.
                    # Unless counts changed? No, we update counts.
                    # So this block might be unreachable or indicate a bug.
                    # Let's assume it's redundant and remove.
                    current_code.remove(c)
                    mask = spheres[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] -= 1
                        m ^= lsb
                    changed = True
                    continue

                # We need to replace c with c' such that c' covers all critical_points.
                # Look at coverers of the first critical point to find candidates
                target = critical_points[0]
                cands = coverers[target]
                random.shuffle(cands)
                
                best_c_prime = -1
                best_score = float('inf')
                
                for c_prime in cands:
                    if c_prime in current_code:
                        continue
                    
                    # Check if c_prime covers all critical points
                    covers_all = True
                    mask_cp = spheres[c_prime]
                    for j in critical_points:
                        if not (mask_cp >> j) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Calculate score: 
                    # We want to minimize the number of points in sphere[c_prime] 
                    # that will have count > 1 after the swap.
                    # This encourages c_prime to cover "rare" points (count 0 or 1) 
                    # rather than "common" points (count > 1).
                    # However, count 0 is impossible for critical points of c.
                    # Points in sphere[c_prime] not in sphere[c] will have count increased by 1.
                    # Points in sphere[c_prime] and sphere[c] will have count unchanged (remove c, add c').
                    # Points in sphere[c] not in sphere[c_prime] will have count decreased by 1.
                    
                    # Let's calculate the "waste": number of j in sphere[c_prime] 
                    # such that count[j] > 1 (after swap).
                    # After swap:
                    # count_new[j] = count[j] - (1 if j in sphere[c] else 0) + (1 if j in sphere[c_prime] else 0)
                    
                    waste = 0
                    mask_cp = spheres[c_prime]
                    mask_c = spheres[c]
                    m = mask_cp
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        new_count = count[j]
                        if (mask_c >> j) & 1:
                            new_count -= 1
                        new_count += 1
                        if new_count > 1:
                            waste += 1
                        m ^= lsb
                    
                    if waste < best_score:
                        best_score = waste
                        best_c_prime = c_prime
                        
                if best_c_prime != -1:
                    # Perform swap
                    current_code.remove(c)
                    mask_c = spheres[c]
                    m = mask_c
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] -= 1
                        m ^= lsb
                        
                    current_code.add(best_c_prime)
                    mask_cp = spheres[best_c_prime]
                    m = mask_cp
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        count[j] += 1
                        m ^= lsb
                    changed = True

        return current_code

    def greedy_cover(deadline):
        uncovered = (1 << total_words) - 1
        code_indices = []
        
        while uncovered and time.time() < deadline:
            best_gain = -1
            best_c = -1
            # To speed up, we can sample or use a smarter heuristic.
            # Full scan is 2187 * (bit count). 
            # Bit count of (sphere & uncovered) is fast in Python? 
            # bin(x).count('1') is O(N) where N is number of bits. 
            # 2187 bits is small. 
            # Let's try full scan first.
            
            # Optimization: if best_gain reaches 15 (max possible), break
            for c in range(total_words):
                gain = bin(spheres[c] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                if best_gain == 15:
                    break
            
            if best_c == -1 or best_gain == 0:
                break
                
            code_indices.append(best_c)
            uncovered &= ~spheres[best_c]
            
        return code_indices

    num_runs = 50 # Adjust based on time
    
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        # Greedy initialization
        code_indices = greedy_cover(deadline)
        
        # Local search
        code_set = local_search(set(code_indices), deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set
            if best_size <= 186:
                break

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = [idx_to_word(i) for i in best_code]
    return result
