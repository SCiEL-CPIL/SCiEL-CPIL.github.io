import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit
    
    # Handle the specific case q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        if q ** n > 100000:
            return []
        # Simple greedy for general case
        total = q ** n
        codewords = []
        uncovered = set(range(total))
        
        def idx_to_digits(idx):
            digits = [0] * n
            for i in range(n - 1, -1, -1):
                digits[i] = idx % q
                idx //= q
            return digits
        
        def digits_to_idx(digits):
            idx = 0
            for d in digits:
                idx = idx * q + d
            return idx
        
        # Precompute neighbors
        neighbors = {}
        for i in range(total):
            digits = idx_to_digits(i)
            nbs = [i]
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    nbs.append(digits_to_idx(new_digits))
            neighbors[i] = nbs
        
        while uncovered and time.time() < deadline:
            # Find codeword covering most uncovered
            best_cw = -1
            best_count = -1
            for cw in uncovered:
                count = sum(1 for nb in neighbors[cw] if nb in uncovered)
                if count > best_count:
                    best_count = count
                    best_cw = cw
            
            if best_cw == -1:
                break
            
            codewords.append(best_cw)
            for nb in neighbors[best_cw]:
                uncovered.discard(nb)
        
        result = [idx_to_digits(cw) for cw in codewords]
        seen = set()
        unique_result = []
        for cw in result:
            t = tuple(cw)
            if t not in seen:
                seen.add(t)
                unique_result.append(cw)
        return unique_result
    
    # Specific case: q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    num_coords = 8
    alphabet = 3
    
    # Precompute balls as bitmasks
    # ball_masks[w] is an integer where bit i is 1 if word i is in the ball of w
    ball_masks = [0] * total_words
    # Also store the list of neighbors for quick iteration if needed
    neighbor_lists = []
    
    weights = [3 ** (num_coords - 1 - i) for i in range(num_coords)]
    
    for w in range(total_words):
        mask = 0
        nbs = [w]
        for pos in range(num_coords):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(alphabet):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                nbs.append(new_w)
                mask |= (1 << new_w)
        ball_masks[w] = mask
        neighbor_lists.append(nbs)
    
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % alphabet
            idx //= alphabet
        return digits
    
    def digits_to_idx(digits):
        idx = 0
        for d in digits:
            idx = idx * alphabet + d
        return idx
    
    def vectorized_greedy(deadline_local):
        """Fast vectorized greedy construction."""
        uncovered_mask = np.ones(total_words, dtype=bool)
        selected = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline_local:
                break
            
            # Vectorized count: for each candidate, count how many uncovered words it covers
            # We need to map word indices to a boolean array
            # balls is (total_words, 17)
            # We can use the precomputed neighbor lists? No, vectorized is faster with numpy array of balls
            # Let's recreate the numpy ball array for speed in greedy
            # Actually, we can just use the bitmasks? No, summing bits is slow in Python.
            # Let's stick to numpy array for greedy.
            
            # Re-create numpy balls array if not cached? It's small enough.
            # balls_np is (6561, 17)
            # We'll use a global cache or just build it once.
            pass 
            
        # Since we need numpy for greedy, let's build it once outside.
        # But for now, let's implement a fast greedy using the neighbor lists if numpy is not convenient for dynamic updates.
        # Actually, the current program's vectorized greedy is good. Let's reuse the logic but ensure it's fast.
        
        # To avoid rebuilding numpy arrays, we can use the fact that the set of neighbors is static.
        # Let's use the precomputed `neighbor_lists` for a simple greedy, it might be fast enough for ~500 iterations.
        # 500 iterations * 6561 candidates * 17 neighbors = ~56M ops. In Python, this is too slow.
        # So we MUST use numpy for greedy.
        
        # Let's create the numpy array once.
        balls_np = np.array(neighbor_lists, dtype=np.int32)
        
        uncovered_mask = np.ones(total_words, dtype=bool)
        selected = []
        
        while np.any(uncovered_mask):
            if time.time() > deadline_local:
                break
            
            counts = uncovered_mask[balls_np].sum(axis=1)
            if np.max(counts) == 0:
                break
            
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = int(candidates[np.random.randint(len(candidates))])
            
            selected.append(best_idx)
            uncovered_mask[balls_np[best_idx]] = False
        
        return set(selected)

    # Precompute numpy balls for greedy
    balls_np = np.array(neighbor_lists, dtype=np.int32)
    
    def local_search_bitmask(code_set, deadline_local):
        """Local search using bitmask operations for swap checks."""
        if not code_set:
            return code_set
            
        code_list = list(code_set)
        cov_count = np.zeros(total_words, dtype=np.int32)
        
        # Initialize coverage
        for c in code_list:
            for nb in neighbor_lists[c]:
                cov_count[nb] += 1
        
        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            code_list = list(code_set)
            random.shuffle(code_list)
            
            # 1. Prune redundant codewords
            for c in code_list:
                if time.time() > deadline_local:
                    break
                if c not in code_set:
                    continue
                
                # Check if all neighbors are covered by > 1 codeword
                redundant = True
                for nb in neighbor_lists[c]:
                    if cov_count[nb] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    code_set.remove(c)
                    for nb in neighbor_lists[c]:
                        cov_count[nb] -= 1
                    improved = True
            
            if not code_set:
                break
                
            code_list = list(code_set)
            random.shuffle(code_list)
            
            # 2. 1-Swaps
            for c in code_list:
                if time.time() > deadline_local:
                    break
                if c not in code_set:
                    continue
                
                # Find critical words
                critical_mask = 0
                for nb in neighbor_lists[c]:
                    if cov_count[nb] == 1:
                        critical_mask |= (1 << nb)
                
                if critical_mask == 0:
                    # Should have been pruned, but handle just in case
                    code_set.remove(c)
                    for nb in neighbor_lists[c]:
                        cov_count[nb] -= 1
                    improved = True
                    continue
                
                # Try to find d that covers all critical words
                # d must be such that ball_masks[d] covers critical_mask
                # i.e., (ball_masks[d] & critical_mask) == critical_mask
                # Also d should not be in code_set
                
                # Optimization: Only try if critical_mask has few bits
                # popcount check
                if critical_mask.bit_count() > 4:
                    continue
                
                found_swap = False
                # Iterate through all possible d
                # We can skip d in code_set
                for d in range(total_words):
                    if d in code_set:
                        continue
                    
                    if (ball_masks[d] & critical_mask) == critical_mask:
                        # Valid swap
                        # Remove c
                        code_set.remove(c)
                        for nb in neighbor_lists[c]:
                            cov_count[nb] -= 1
                        
                        # Add d
                        code_set.add(d)
                        for nb in neighbor_lists[d]:
                            cov_count[nb] += 1
                        
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break # Restart loop with new state
            
            # 3. 2-Swaps (Remove c1, c2, add d)
            if not improved and time.time() < deadline_local:
                code_list = list(code_set)
                random.shuffle(code_list)
                n_cw = len(code_list)
                
                found_2_swap = False
                for i in range(n_cw):
                    if time.time() > deadline_local:
                        break
                    c1 = code_list[i]
                    if c1 not in code_set:
                        continue
                        
                    for j in range(i + 1, n_cw):
                        c2 = code_list[j]
                        if c2 not in code_set:
                            continue
                            
                        # Find critical words for {c1, c2}
                        critical_mask = 0
                        # Words covered only by c1 or c2 (and not others)
                        # cov_count[w] == 1 and w in ball(c1) or ball(c2)
                        # Actually, if w is in ball(c1) and ball(c2), and cov_count[w]==2, removing both makes it 0.
                        # If w is in ball(c1) only, and cov_count[w]==1, removing c1 makes it 0.
                        
                        # Efficient way:
                        # Iterate neighbors of c1
                        for nb in neighbor_lists[c1]:
                            if cov_count[nb] == 1:
                                critical_mask |= (1 << nb)
                            elif cov_count[nb] == 2 and nb in neighbor_lists[c2]:
                                # This word is covered by c1 and c2. If we remove both, it becomes 0.
                                # But wait, if cov_count is 2, it means it's covered by c1 and c2 (since we are removing them).
                                # Are there other codewords covering it? No, count is 2.
                                # So yes, it becomes uncovered.
                                critical_mask |= (1 << nb)
                        
                        # Iterate neighbors of c2 (not in c1)
                        for nb in neighbor_lists[c2]:
                            if nb in neighbor_lists[c1]:
                                continue
                            if cov_count[nb] == 1:
                                critical_mask |= (1 << nb)
                            
                        if critical_mask == 0:
                            # Both redundant
                            code_set.remove(c1)
                            for nb in neighbor_lists[c1]:
                                cov_count[nb] -= 1
                            code_set.remove(c2)
                            for nb in neighbor_lists[c2]:
                                cov_count[nb] -= 1
                            improved = True
                            found_2_swap = True
                            break
                            
                        if critical_mask.bit_count() > 3:
                            continue
                            
                        # Find d
                        for d in range(total_words):
                            if d in code_set:
                                continue
                            
                            if (ball_masks[d] & critical_mask) == critical_mask:
                                # Valid 2-swap
                                code_set.remove(c1)
                                for nb in neighbor_lists[c1]:
                                    cov_count[nb] -= 1
                                code_set.remove(c2)
                                for nb in neighbor_lists[c2]:
                                    cov_count[nb] -= 1
                                
                                code_set.add(d)
                                for nb in neighbor_lists[d]:
                                    cov_count[nb] += 1
                                
                                improved = True
                                found_2_swap = True
                                break
                                
                        if found_2_swap:
                            break
                            
                    if found_2_swap:
                        break

        return code_set

    # Main loop
    best_set = None
    best_size = total_words + 1
    
    final_search_time = 15
    
    while time.time() < deadline - final_search_time:
        init_deadline = min(time.time() + 3.0, deadline - final_search_time)
        code_set = vectorized_greedy(init_deadline)
        
        if not code_set:
            continue
            
        ls_deadline = min(time.time() + 5.0, deadline - final_search_time)
        code_set = local_search_bitmask(code_set, ls_deadline)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_set = set(code_set)
            
        if best_size <= 486:
            break
            
    if best_set is not None and time.time() < deadline - 2:
        best_set = local_search_bitmask(best_set, deadline - 2)
        
    if best_set is None:
        best_set = vectorized_greedy(deadline - 1)
        
    result = [idx_to_digits(c) for c in best_set]
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
    return unique_result
