import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 17
    MIN_D = 6
    TOTAL = 1 << N
    
    # 1. Vectorized Popcount Calculation
    # Use bitwise operations to compute popcount for all 17-bit integers
    all_words = np.arange(TOTAL, dtype=np.uint32)
    # Popcount via bit tricks
    # x = x - (x >> 1) & 0x55555555
    # x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    # x = (x + (x >> 4)) & 0x0F0F0F0F
    # x = (x * 0x01010101) >> 24
    x = all_words
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    weights = (x * 0x01010101) >> 24
    weights = weights.astype(np.uint8)
    
    # Precompute popcount lookup for 17-bit values (already done above, but let's ensure it's a lookup table)
    # Actually, the vectorized calculation gives us the weights for all possible XORs directly if we just use the same logic on the XOR results.
    # But for random access in a loop, a lookup table is faster if we can't vectorize the whole check.
    # However, we CAN vectorize the check:
    # dists = popcount_lookup[code_np ^ word]
    # So we need a lookup table of size 2**17.
    popcount_lookup = weights
    
    # 2. Generate initial linear code (fast enough with vectorized checks? No, linear code generation is greedy too)
    # Let's just start with a random greedy pass, it's often as good or better and simpler.
    # But the task mentions a 256-word linear code. Let's try to generate it quickly or just use random greedy.
    # Random greedy is simpler and robust.
    
    best_code = []
    best_size = 0
    
    # Helper to check if a word can be added using numpy vectorization
    def can_add_fast(word, code_np):
        """Check if word has min distance >= MIN_D from all words in code_np"""
        if len(code_np) == 0:
            return True
        # Compute XOR distances
        xors = code_np ^ word
        # Compute popcounts using lookup
        dists = popcount_lookup[xors]
        return dists.min() >= MIN_D
    
    # Helper to perform a greedy pass
    def greedy_pass(order):
        code = []
        code_np = np.array([], dtype=np.uint32)
        for w in order:
            if can_add_fast(w, code_np):
                code.append(w)
                code_np = np.append(code_np, w)
        return code
    
    # Local improvement: try to add words to an existing code
    def try_extend(code, remaining_candidates):
        """Try to add words from remaining_candidates to code"""
        code_set = set(code)
        code_np = np.array(code, dtype=np.uint32)
        extended = list(code)
        for w in remaining_candidates:
            if w in code_set:
                continue
            if can_add_fast(w, code_np):
                extended.append(w)
                code_set.add(w)
                code_np = np.append(code_np, w)
        return extended
    
    # Create all_words as a list for shuffling
    all_words_list = list(range(TOTAL))
    
    # Precompute weight groups for weighted ordering
    weight_groups = {}
    for w in range(TOTAL):
        wt = weights[w]
        if wt not in weight_groups:
            weight_groups[wt] = []
        weight_groups[wt].append(w)
    
    pass_count = 0
    while time.time() < deadline - 5:
        pass_count += 1
        
        # Alternate between strategies
        if pass_count % 3 == 0:
            # Weight-based ordering: shuffle within each weight class
            order = []
            for wt in sorted(weight_groups.keys()):
                group = weight_groups[wt].copy()
                random.shuffle(group)
                order.extend(group)
            # Also shuffle the weight order sometimes
            if random.random() < 0.5:
                weight_order = sorted(weight_groups.keys())
                random.shuffle(weight_order)
                order = []
                for wt in weight_order:
                    order.extend(weight_groups[wt])
        elif pass_count % 3 == 1:
            # Uniform random
            random.shuffle(all_words_list)
            order = all_words_list[:]
        else:
            # Start from best code and try to extend
            if len(best_code) > 0:
                # Create candidates not in best_code
                best_set = set(best_code)
                candidates = [w for w in range(TOTAL) if w not in best_set]
                random.shuffle(candidates)
                order = best_code + candidates
            else:
                random.shuffle(all_words_list)
                order = all_words_list[:]
        
        current_code = greedy_pass(order)
        
        if len(current_code) > best_size:
            best_code = current_code
            best_size = len(current_code)
            # best_code_np = np.array(best_code, dtype=np.uint32) # Not strictly needed if we re-create in try_extend
        
        # Try local improvement on best code
        if len(best_code) > 0 and time.time() < deadline - 10:
            # Try to extend with words not in current best
            best_set = set(best_code)
            candidates = [w for w in range(TOTAL) if w not in best_set]
            random.shuffle(candidates)
            # Only try a subset to save time? 
            # With vectorized checks, we can try more.
            # Let's try the first 1000 candidates
            subset = candidates[:1000]
            extended = try_extend(best_code, subset)
            if len(extended) > best_size:
                best_code = extended
                best_size = len(extended)
    
    return best_code
