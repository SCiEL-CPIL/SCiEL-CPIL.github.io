import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    total_space = q ** n
    all_ones = (1 << total_space) - 1
    
    # Precompute cover masks
    cover_masks = [0] * total_space
    
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for j in range(n):
            w[j] = temp % q
            temp //= q
        return w

    for idx in range(total_space):
        center = idx_to_word(idx)
        mask = 1 << idx
        
        if radius >= 1:
            for pos in range(n):
                orig_val = center[pos]
                weight = q ** pos
                for val in range(q):
                    if val == orig_val:
                        continue
                    delta = (val - orig_val) * weight
                    new_idx = idx + delta
                    mask |= (1 << new_idx)
        
        cover_masks[idx] = mask

    def greedy_cover():
        uncovered = all_ones
        code = []
        
        while uncovered != 0:
            best_gain = 0
            best_idx = -1
            
            for i in range(total_space):
                gain = (uncovered & cover_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_gain == 0:
                break
                
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        return code

    def local_search(code):
        if not code:
            return code
            
        code_set = set(code)
        improved = True
        
        while improved:
            improved = False
            
            # Shuffle to avoid bias in deterministic order? 
            # No, let's keep it deterministic for stability, 
            # but the outer loop will randomize the input.
            
            for i in range(len(code)):
                c = code[i]
                
                # Calculate coverage of all other codewords
                others_mask = 0
                for j in range(len(code)):
                    if i == j:
                        continue
                    others_mask |= cover_masks[code[j]]
                
                unique_to_c = cover_masks[c] & ~others_mask
                
                if unique_to_c == 0:
                    # Redundant
                    code.pop(i)
                    code_set.remove(c)
                    improved = True
                    break
                    
                # Try 1-for-1 swap
                # Find c_prime not in code_set such that cover_masks[c_prime] covers unique_to_c
                # Optimization: Only check c_prime that intersect unique_to_c
                # But we need to check if it covers ALL of unique_to_c.
                
                # Iterate through all possible c_prime
                # We can speed this up by checking if (cover_masks[c_prime] & unique_to_c) == unique_to_c
                # This is 2187 checks per c. 200 * 2187 = 437k checks. Fast.
                
                found_swap = False
                for c_prime in range(total_space):
                    if c_prime in code_set:
                        continue
                    if (cover_masks[c_prime] & unique_to_c) == unique_to_c:
                        # Swap
                        code[i] = c_prime
                        code_set.remove(c)
                        code_set.add(c_prime)
                        improved = True
                        found_swap = True
                        break
                        
                if found_swap:
                    break
        
        return code

    # Initial Greedy
    best_code = greedy_cover()
    best_len = len(best_code)
    
    # Local Search
    best_code = local_search(best_code)
    best_len = len(best_code)
    
    # Randomized Restarts
    while time.time() < deadline:
        # Randomized Greedy
        uncovered = all_ones
        code = []
        while uncovered != 0:
            # To add randomness, we can pick from top K
            # But simple random tie-breaking is enough.
            gains = []
            for i in range(total_space):
                gain = (uncovered & cover_masks[i]).bit_count()
                gains.append(gain)
            
            max_gain = max(gains)
            if max_gain == 0:
                break
                
            candidates = [i for i, g in enumerate(gains) if g == max_gain]
            best_idx = random.choice(candidates)
            
            code.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
            
        code = local_search(code)
        
        if len(code) < best_len:
            best_code = code
            best_len = len(code)
            
    return [idx_to_word(i) for i in best_code]
