import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85

    max_val = 256
    max_size = 60

    def is_valid_exact(s):
        """Exact check using bitsets."""
        n = len(s)
        if n < 2:
            return True
        
        # dp[k] is a bitset where bit sum is 1 if sum is reachable with k elements
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for x in s:
            for k in range(n, 0, -1):
                dp[k] |= (dp[k-1] << x)
                
        for k in range(n + 1):
            for j in range(k + 1, n + 1):
                if dp[k] & dp[j]:
                    return False
        return True

    # Phase 1: Analytical Construction (AP)
    # Find largest k such that AP [257-k, ..., 256] is valid
    best_set = []
    best_k = 0
    
    for k in range(max_size, 0, -1):
        if time.time() > deadline - 10:
            break
        s = list(range(257 - k, 257))
        if is_valid_exact(s):
            best_set = s
            best_k = k
            break
            
    # Phase 2: Targeted Search for best_k + 1
    if best_k < max_size and time.time() < deadline - 5:
        target_k = best_k + 1
        seed_set = best_set[:]
        seed_set_set = set(seed_set)
        
        # Try to add one element to the AP set
        # The AP set is [257-best_k, ..., 256]
        # We can try adding x < 257-best_k
        # Or swapping an element to free up space for two?
        # Simple: Try adding x
        candidates = [x for x in range(1, 257 - best_k) if x not in seed_set_set]
        random.shuffle(candidates)
        
        found = False
        for x in candidates:
            if time.time() > deadline - 5:
                break
            new_s = seed_set + [x]
            if is_valid_exact(new_s):
                best_set = new_s
                best_k = target_k
                found = True
                break
                
        # If simple add failed, try 1-swap + 1-add?
        # This is complex. Let's try a few random swaps in the seed set to see if a new set of size best_k allows an add.
        if not found:
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                s = seed_set[:]
                s_set = set(s)
                # Swap one element
                i = random.randint(0, len(s) - 1)
                old_val = s[i]
                # Pick a new value not in set
                # Bias towards smaller values to create gaps?
                new_val = random.randint(1, 256)
                if new_val in s_set:
                    continue
                s[i] = new_val
                s_set.remove(old_val)
                s_set.add(new_val)
                
                if not is_valid_exact(s):
                    continue
                
                # Try to add a new element
                add_candidates = [x for x in range(1, 257) if x not in s_set]
                random.shuffle(add_candidates)
                for x in add_candidates[:20]: # Limit checks
                    if time.time() > deadline - 5:
                        break
                    new_s = s + [x]
                    if is_valid_exact(new_s):
                        best_set = new_s
                        best_k = target_k
                        found = True
                        break
                if found:
                    break

    if not best_set:
        return [1]
        
    return [int(x) for x in best_set]
