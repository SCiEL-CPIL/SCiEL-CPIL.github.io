import time
import random
import math

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Heuristic: Try to construct superincreasing-like sequences
    # with random perturbations to avoid divisibility.
    
    # We want to maximize size. Max possible is likely around 7-9.
    # Let's try to find a valid set of size 7, 8, 9...
    
    # Strategy: 
    # 1. Generate random superincreasing sequences within 1..256.
    # 2. Validate.
    # 3. Local search: try swapping elements to fix validity or increase size.
    
    def is_valid_set(s):
        if not s:
            return True
        # Generate all subset sums
        sums = [0]
        for x in s:
            new_sums = [sum_ + x for sum_ in sums]
            sums.extend(new_sums)
        
        # Remove 0 (empty set)
        sums = [x for x in sums if x > 0]
        
        # Check distinctness
        if len(sums) != len(set(sums)):
            return False
        
        sums_set = set(sums)
        
        # Check divisibility
        # For each sum, check if any other sum divides it or is divided by it.
        # Optimization: sort sums. For each sum s, check divisors in sums_set.
        for s in sums:
            # Check divisors of s
            limit = int(math.isqrt(s))
            for d in range(1, limit + 1):
                if s % d == 0:
                    if d in sums_set and d != s:
                        return False
                    other = s // d
                    if other != d and other in sums_set and other != s:
                        return False
        return True

    def generate_superincreasing_sequence(start_val, max_val, target_size, randomness=0.0):
        seq = []
        current_sum = 0
        for _ in range(target_size):
            # Next element must be > current_sum to be superincreasing
            # To be "binary-like", we want it close to current_sum or double it?
            # Superincreasing: x_i > sum_{j<i} x_j
            # If we want smallest integers, we pick x_i = current_sum + 1
            # But we need to fit in max_val.
            
            # Randomize the growth factor
            # x_i = current_sum * (1 + randomness * random()) + 1
            # Ensure x_i > current_sum
            min_val = current_sum + 1
            max_possible = max_val
            if min_val > max_possible:
                break
                
            # Add some randomness to avoid exact divisibility patterns
            # We want x_i to not be a multiple of previous elements if possible
            # But superincreasing ensures distinct sums.
            # Divisibility is the main risk.
            
            # Try a range of values
            # Pick a value in [min_val, max_possible]
            # To maximize size, we want small values.
            # So pick near min_val.
            # But add noise.
            
            # Let's try picking from a window near the minimum
            window = max(1, int((max_possible - min_val) * 0.1))
            if window > 10: window = 10
            
            candidates = range(min_val, min(min_val + window, max_possible + 1))
            if not candidates:
                break
                
            # Shuffle candidates
            cand_list = list(candidates)
            random.shuffle(cand_list)
            
            # Try to find a candidate that doesn't immediately cause divisibility issues
            # This is a pre-check.
            found = False
            for c in cand_list:
                # Simple pre-check: c should not be divisible by any previous element
                # and no previous element should divide c
                ok = True
                for prev in seq:
                    if c % prev == 0:
                        ok = False
                        break
                    if prev % c == 0:
                        ok = False
                        break
                if ok:
                    seq.append(c)
                    current_sum += c
                    found = True
                    break
            
            if not found:
                # If no candidate passed pre-check, try the smallest one anyway
                # or break
                if cand_list:
                    seq.append(cand_list[0])
                    current_sum += cand_list[0]
                else:
                    break
                
        return seq

    # Try different starting values and randomness levels
    for target_size in range(10, 4, -1):
        if time.time() - start_time > 55:
            break
            
        found_valid = False
        for _ in range(1000):
            if time.time() - start_time > 55:
                break
                
            # Random start value between 1 and 50
            start_val = random.randint(1, 50)
            # Randomness in growth
            randomness = random.uniform(0.0, 0.5)
            
            seq = generate_superincreasing_sequence(start_val, 256, target_size, randomness)
            
            if len(seq) < best_size:
                continue
                
            if is_valid_set(seq):
                best_size = len(seq)
                best_set = seq[:]
                found_valid = True
                # If we found a large set, we can break early if size is high
                if best_size >= 9:
                    break
        
        if found_valid and best_size >= 9:
            break

    # If no superincreasing set found, fallback to current greedy approach
    if best_size < 5:
        # Run the original greedy logic as fallback
        candidates_base = list(range(1, 257))
        orders = [sorted(candidates_base, reverse=True), sorted(candidates_base)]
        for _ in range(50):
            random.shuffle(candidates_base)
            orders.append(candidates_base[:])
            
        for order in orders:
            if time.time() - start_time > 55:
                break
            current_set = []
            sums_list = []
            sums_set = set()
            
            for x in order:
                new_sums_list = [s + x for s in sums_list]
                new_sums_list.append(x)
                valid = True
                for s_new in new_sums_list:
                    limit = int(math.isqrt(s_new))
                    for i in range(1, limit + 1):
                        if s_new % i == 0:
                            if i in sums_set and i != s_new:
                                valid = False
                                break
                            other = s_new // i
                            if other != i and other in sums_set and other != s_new:
                                valid = False
                                break
                    if not valid: break
                    for k in range(2, 3840 // s_new + 1):
                        if k * s_new in sums_set:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    # Check within new_sums
                    new_sums_set = set(new_sums_list)
                    for s_new in new_sums_list:
                        limit = int(math.isqrt(s_new))
                        for i in range(1, limit + 1):
                            if s_new % i == 0:
                                if i in new_sums_set and i != s_new:
                                    valid = False
                                    break
                                other = s_new // i
                                if other != i and other in new_sums_set and other != s_new:
                                    valid = False
                                    break
                        if not valid: break
                        for k in range(2, 3840 // s_new + 1):
                            if k * s_new in new_sums_set:
                                valid = False
                                break
                        if not valid: break
                if valid:
                    current_set.append(x)
                    sums_list.extend(new_sums_list)
                    sums_set.update(new_sums_list)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]

    return best_set
