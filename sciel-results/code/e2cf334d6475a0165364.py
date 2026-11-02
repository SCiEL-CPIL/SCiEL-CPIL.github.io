import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Total words
    total_words = q ** n
    
    # Precompute neighbors and masks
    def word_to_idx(word):
        idx = 0
        for d in word:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    balls = []
    ball_masks = []
    
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors = [idx]
        for pos in range(n):
            original = w[pos]
            for new_val in range(q):
                if new_val == original:
                    continue
                w[pos] = new_val
                n_idx = word_to_idx(w)
                neighbors.append(n_idx)
            w[pos] = original
        
        balls.append(neighbors)
        mask = 0
        for nb in neighbors:
            mask |= (1 << nb)
        ball_masks.append(mask)
        
    all_mask = (1 << total_words) - 1
    
    best_solution = None
    best_size = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        covered_mask = 0
        code = []
        
        while covered_mask != all_mask:
            best_gain = -1
            best_candidates = []
            
            for c in range(total_words):
                gain = (ball_masks[c] & ~covered_mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            chosen = random.choice(best_candidates)
            code.append(chosen)
            covered_mask |= ball_masks[chosen]
            
            if time.time() > deadline:
                return None
        
        return code

    def reduce_cover(code):
        if not code:
            return code
            
        cover_count = [0] * total_words
        
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                is_redundant = True
                for w in balls[c]:
                    if cover_count[w] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for w in balls[c]:
                        cover_count[w] -= 1
                    code.pop(i)
                    changed = True
        
        return code

    def local_search(code):
        """
        Try to reduce size by swapping codewords.
        """
        if not code:
            return code
            
        cover_count = [0] * total_words
        for c in code:
            for w in balls[c]:
                cover_count[w] += 1
                
        in_code = [False] * total_words
        for c in code:
            in_code[c] = True
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove each codeword
            for i in range(len(code)):
                c = code[i]
                # Find loss
                loss_mask = 0
                for w in balls[c]:
                    if cover_count[w] == 1:
                        loss_mask |= (1 << w)
                        
                if loss_mask == 0:
                    # Redundant
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code.pop(i)
                    improved = True
                    break # Restart loop
            
            if improved:
                continue
                
            # Try swaps
            for i in range(len(code)):
                c = code[i]
                loss_mask = 0
                for w in balls[c]:
                    if cover_count[w] == 1:
                        loss_mask |= (1 << w)
                        
                if loss_mask == 0:
                    continue # Should have been caught above, but safety
                    
                # Find a c' not in code that covers loss_mask
                # Optimization: only check c' that have some overlap with loss?
                # Just iterate all
                best_c_prime = -1
                # We want to find ANY c' that covers loss_mask
                # If found, we swap. Then we check if reduction is possible.
                # To avoid O(N^2) if many swaps, we can just try the first one found?
                # Or try a few random ones?
                
                # Let's try to find one that maximizes "overlap" with other codewords' unique words?
                # Simpler: Just find one.
                for c_prime in range(total_words):
                    if in_code[c_prime]:
                        continue
                    if (ball_masks[c_prime] & loss_mask) == loss_mask:
                        best_c_prime = c_prime
                        break
                        
                if best_c_prime != -1:
                    # Perform swap
                    # Remove c
                    for w in balls[c]:
                        cover_count[w] -= 1
                    in_code[c] = False
                    code[i] = best_c_prime
                    
                    # Add c_prime
                    for w in balls[best_c_prime]:
                        cover_count[w] += 1
                    in_code[best_c_prime] = True
                    
                    # Check if this swap enabled a reduction
                    # Run a quick reduction pass
                    reduced = False
                    for j in range(len(code)):
                        cj = code[j]
                        redundant = True
                        for w in balls[cj]:
                            if cover_count[w] == 1:
                                redundant = False
                                break
                        if redundant:
                            # Remove cj
                            for w in balls[cj]:
                                cover_count[w] -= 1
                            in_code[cj] = False
                            code.pop(j)
                            reduced = True
                            break
                            
                    if reduced:
                        improved = True
                        break # Restart main while loop
                    else:
                        # Revert swap
                        # Remove c_prime
                        for w in balls[best_c_prime]:
                            cover_count[w] -= 1
                        in_code[best_c_prime] = False
                        code[i] = c
                        # Add c
                        for w in balls[c]:
                            cover_count[w] += 1
                        in_code[c] = True
                        
        return code

    # Main loop
    initial_code = greedy_cover(0)
    if initial_code:
        reduced_code = reduce_cover(initial_code)
        local_searched_code = local_search(reduced_code)
        best_solution = local_searched_code
        best_size = len(local_searched_code)
    
    seed = 1
    while time.time() < deadline:
        code = greedy_cover(seed)
        if code is None:
            break
        reduced_code = reduce_cover(code)
        local_searched_code = local_search(reduced_code)
        if len(local_searched_code) < best_size:
            best_solution = local_searched_code
            best_size = len(local_searched_code)
        seed += 1
        
    if best_solution is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = []
    for idx in best_solution:
        result.append(idx_to_word(idx))
        
    return result
