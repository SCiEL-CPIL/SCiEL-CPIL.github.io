import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    total_words = q ** n
    
    # Precompute digits for all words as a 2D array
    # digits[i] is the tuple of digits for word i
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors (balls) as bitmasks
    # Each ball is a Python integer where bit j is 1 if word j is in the ball of word i
    # For total_words = 6561, we need integers up to 2^6561. Python handles big ints natively.
    
    # To speed up bit setting, we can't easily use numpy for >64 bit integers in a simple array of ints 
    # that supports bitwise ops efficiently across 6561 bits in a single op without complex packing.
    # However, Python big ints are quite fast for bitwise AND and POPCOUNT (bit_count()).
    
    # Let's generate the balls.
    # For radius 1, neighbors are self + words differing in 1 pos.
    # General radius: combinations of positions.
    
    balls = [0] * total_words
    
    # Precompute bit masks for setting bits? No, just 1 << j.
    # To make it faster, we can precompute the list of indices in the ball for each i first,
    # then convert to bitmask.
    
    # Actually, generating the bitmask directly is fine if we do it carefully.
    
    # Optimization: 
    # If q=3, n=8, radius=1:
    # Ball size = 1 + 8*2 = 17.
    # We can iterate through all words, find their neighbors, and set bits.
    
    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        # Generate neighbors
        # For radius 1:
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    # Calculate index of neighbor
                    # new_d = list(d)
                    # new_d[pos] = new_digit
                    # idx = sum(new_d[j] * q^(n-1-j))
                    # We can compute delta = (new_digit - orig_digit) * q^(n-1-pos)
                    # idx = i + delta
                    delta = (new_digit - orig_digit) * (q ** (n - 1 - pos))
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius (slower, but n=8 is small)
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    orig_digits = [d[p] for p in pos_comb]
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * (q ** (n - 1 - pos))
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    # Greedy Set Cover using Bitmasks
    # uncovered is a bitmask of all words initially
    all_mask = (1 << total_words) - 1
    
    best_code = None
    min_size = float('inf')
    
    # We will run a few restarts with different random tie-breaking seeds
    # Since the space is small, greedy is fast.
    
    # To allow for local search, we need a function to check if a set of codewords covers all.
    # cover_mask = OR of balls[c] for c in code
    # is_covered = (cover_mask == all_mask)
    
    def get_cover_mask(code_indices):
        mask = 0
        for c in code_indices:
            mask |= balls[c]
        return mask
        
    def is_valid(code_indices):
        # Check if union of balls covers all
        # Optimization: if len(code) * max_ball_size < total_words, invalid (loose bound)
        # Just do the OR
        mask = 0
        for c in code_indices:
            mask |= balls[c]
        return mask == all_mask

    # Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        uncovered = all_mask
        code = []
        
        while uncovered:
            if time.time() > deadline - 5.0:
                return None # Timeout safety
            
            best_idx = -1
            max_cover = 0
            candidates = []
            
            # Iterate through all possible codewords
            # Optimization: We can skip indices that are already in code? 
            # No, we need to check marginal gain.
            # marginal_gain = (balls[i] & uncovered).bit_count()
            
            # To speed this up, we can maintain a list of active candidates?
            # With 6561 items, iterating all is ~6500 ops. 
            # bit_count on 6561-bit int is fast in Python 3.10+.
            
            for i in range(total_words):
                # If i is already in code, gain is 0.
                # Checking membership in list is O(N). 
                # Better: maintain a boolean array or set for chosen.
                # But we can just check if (balls[i] & uncovered) is 0? No, that's the gain.
                
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
                    
            if max_cover == 0:
                # Should not happen if we start with full space and balls cover self
                break
                
            if len(candidates) > 1:
                best_idx = random.choice(candidates)
            else:
                best_idx = candidates[0]
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    # Local Search: 1-Swap (Remove one, add one) or 1-Remove
    def local_search(code, max_iterations=500):
        if code is None:
            return None
            
        current_code = code[:]
        current_size = len(current_code)
        
        # Precompute cover mask
        current_cover = get_cover_mask(current_code)
        
        # We want to reduce size.
        # Strategy 1: Try to remove a codeword. If valid, keep it.
        # Strategy 2: Try to swap one codeword with another not in code, if it allows removal of another?
        # Simpler: 1-Remove + 1-Swap (Replace one with another, then try remove another?)
        
        # Let's do:
        # 1. While possible, try to remove any codeword.
        # 2. If stuck, try to swap one codeword with a random other, and see if we can then remove another.
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Remove redundant
            for i in range(len(current_code)):
                c = current_code[i]
                # Check if removing c keeps cover valid
                # new_cover = current_cover & ~balls[c]
                # if new_cover == all_mask: valid
                
                # Optimization: if balls[c] is subset of (current_cover ^ balls[c]), then removing c is valid?
                # No. current_cover is union of all.
                # If (current_cover ^ balls[c]) == all_mask, then removing c is valid.
                # Wait. current_cover includes balls[c].
                # If we remove c, new_cover = current_cover & ~balls[c].
                # We need new_cover == all_mask.
                # This implies current_cover must be all_mask (which it is) AND 
                # all_mask & ~balls[c] == all_mask? No.
                # It means every word in all_mask must be covered by some other codeword.
                # i.e., (current_cover & ~balls[c]) == all_mask.
                
                if (current_cover & ~balls[c]) == all_mask:
                    # Safe to remove
                    current_code.pop(i)
                    current_cover &= ~balls[c]
                    current_size -= 1
                    improved = True
                    break # Restart loop as indices changed
                    
            if not improved:
                # Phase 2: Try swaps
                # Pick a random codeword to remove, and a random other to add?
                # Or: Try to replace codeword A with codeword B, such that we can then remove C?
                # This is complex. 
                # Let's try simple swap: Replace A with B. If valid, do it.
                # But size doesn't change. We only care about size reduction.
                # However, a swap might enable a subsequent removal.
                
                # Heuristic: Try to find a codeword A such that there exists B (not in code)
                # where swapping A->B allows removal of some C.
                
                # Simplified Local Search:
                # 1. Pick random A in code.
                # 2. Pick random B not in code.
                # 3. Tentatively swap A->B.
                # 4. Try to remove any redundant codeword from the new set.
                # 5. If size decreased, accept. Else revert.
                
                if len(current_code) > 1:
                    # To keep it fast, limit attempts
                    for _ in range(100):
                        if time.time() > deadline - 2.0:
                            return current_code
                            
                        idx_a = random.randint(0, len(current_code) - 1)
                        a = current_code[idx_a]
                        
                        # Pick B not in code
                        # Since total_words is 6561, we can just pick random until not in code
                        # Or maintain a set.
                        b = random.randint(0, total_words - 1)
                        if b in current_code:
                            continue
                            
                        # Tentative swap
                        new_code = current_code[:]
                        new_code[idx_a] = b
                        new_cover = get_cover_mask(new_code)
                        
                        # Try to remove redundant from new_code
                        # Check if any codeword in new_code is redundant
                        removed_any = False
                        for j in range(len(new_code)):
                            c = new_code[j]
                            if (new_cover & ~balls[c]) == all_mask:
                                # Remove c
                                new_code.pop(j)
                                new_cover &= ~balls[c]
                                removed_any = True
                                break
                                
                        if removed_any and len(new_code) < current_size:
                            current_code = new_code
                            current_cover = new_cover
                            current_size = len(current_code)
                            improved = True
                            break # Restart main loop
                            
        return current_code

    # Main Loop
    # Run greedy with a few seeds, then local search on the best one
    best_candidate = None
    best_size = float('inf')
    
    seeds = [42, 123, 456, 789]
    
    for seed in seeds:
        if time.time() > deadline - 20.0:
            break
            
        code = greedy_construct(seed)
        if code is None:
            continue
            
        # Local search
        code = local_search(code)
        
        if code is not None:
            if len(code) < best_size:
                best_size = len(code)
                best_candidate = code
                
    if best_candidate is None:
        # Fallback to tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(digits_of[i]) for i in best_candidate]
