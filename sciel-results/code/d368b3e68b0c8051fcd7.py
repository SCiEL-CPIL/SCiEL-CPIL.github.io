import time
import random

def construct():
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    # Max possible sum for 60 elements of max 256 is 60*256 = 15360
    # We need bitsets up to this size.
    # Python integers are arbitrary precision, so shifting is easy.
    
    best_set = []
    
    def is_valid(A):
        """Verify validity using bitsets."""
        if not A:
            return True
        n = len(A)
        S = [0] * (n + 1)
        S[0] = 1  # bit 0 is set, sum 0
        for x in A:
            for k in range(n, 0, -1):
                S[k] |= (S[k-1] << x)
        # Check disjointness
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if S[i] & S[j]:
                    return False
        return True

    def try_build(order, max_depth=60):
        """
        Backtracking search to build a valid set.
        order: list of candidates to try.
        """
        nonlocal best_set
        current_set = []
        S = [0] * (max_depth + 1)
        S[0] = 1
        
        def backtrack(idx, depth):
            nonlocal best_set
            if time.time() - start_time > deadline:
                return
            
            if depth > len(best_set):
                best_set = list(current_set)
            
            if depth >= max_depth:
                return
            
            # Try candidates from order starting at idx
            # We can restrict idx to avoid permutations, but since we are picking a subset,
            # and order is fixed, we just iterate through remaining candidates.
            # To be efficient, we should skip candidates that are "too small" or "too large" if possible?
            # No, just iterate.
            
            for i in range(idx, len(order)):
                x = order[i]
                
                # Check if x can be added
                # We need to check if adding x causes collisions.
                # New sums for cardinality k are S[k-1] << x.
                # We need to ensure these don't collide with existing S[j] (j != k)
                # or other new sums S[j-1] << x (j != k).
                
                valid = True
                new_bits = [0] * (depth + 2)
                
                for k in range(1, depth + 2):
                    if S[k-1]:
                        new_bits[k] = S[k-1] << x
                        # Check collision with old S[j] for j != k
                        for j in range(depth + 1):
                            if j == k:
                                continue
                            if S[j] & new_bits[k]:
                                valid = False
                                break
                        if not valid:
                            break
                        
                        # Check collision with other new_bits[j] for j != k
                        # Note: new_bits[j] is computed for j < k in the loop?
                        # No, we compute new_bits[k] now. We have new_bits[1..k-1] already.
                        # We also need to check against new_bits[j] for j > k?
                        # Those haven't been computed yet.
                        # So we must check against all j != k.
                        # This requires computing all new_bits first, then checking.
                        # Or checking incrementally is tricky because of future j.
                        
                        # Let's just compute all new_bits first, then check.
                        pass
                
                if not valid:
                    continue
                
                # Recompute properly:
                valid = True
                new_bits = [0] * (depth + 2)
                for k in range(1, depth + 2):
                    if S[k-1]:
                        new_bits[k] = S[k-1] << x
                
                # Check collisions
                for k in range(1, depth + 2):
                    if not new_bits[k]:
                        continue
                    for j in range(depth + 1):
                        if j == k:
                            continue
                        # Collision with old S[j]
                        if S[j] & new_bits[k]:
                            valid = False
                            break
                    if not valid:
                        break
                    for j in range(1, depth + 2):
                        if j == k:
                            continue
                        # Collision with new S[j]
                        if new_bits[j] & new_bits[k]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Add x
                current_set.append(x)
                for k in range(depth + 1, 0, -1):
                    S[k] |= new_bits[k]
                
                backtrack(i + 1, depth + 1)
                
                # Undo
                current_set.pop()
                for k in range(depth + 1, 0, -1):
                    S[k] &= ~new_bits[k]
                
                if time.time() - start_time > deadline:
                    return

        backtrack(0, 0)

    # Try different orderings
    all_nums = list(range(1, 257))
    
    # 1. Ascending
    try_build(all_nums)
    
    # 2. Descending
    try_build(sorted(all_nums, reverse=True))
    
    # 3. Randomized restarts
    for _ in range(200):
        if time.time() - start_time > deadline:
            break
        random.shuffle(all_nums)
        try_build(all_nums)
        
        # Also try biased random
        if random.random() < 0.5:
            # Bias towards larger numbers
            order = sorted(all_nums, key=lambda x: random.random() ** 2)
            try_build(order)
        else:
            # Bias towards smaller numbers
            order = sorted(all_nums, key=lambda x: random.random() ** 0.5)
            try_build(order)

    # Final verification
    if best_set and not is_valid(best_set):
        # Fallback to a known valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return [int(x) for x in best_set]
