import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

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

    def is_valid_range(s):
        """
        Check if sorted s satisfies Max_k < Min_{k+1} for all k.
        This is a sufficient condition for validity, and very fast.
        """
        n = len(s)
        if n < 2:
            return True
        s_sorted = sorted(s)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + s_sorted[i]
            
        for k in range(1, n):
            # Max sum with k elements is sum of k largest
            max_k = prefix[n] - prefix[n - k]
            # Min sum with k+1 elements is sum of k+1 smallest
            min_k1 = prefix[k + 1]
            if max_k >= min_k1:
                return False
        return True

    best_set = []

    # Phase 1: Binary search / Linear search on set size k using AP construction
    # For a given k, the set [257-k, ..., 256] is the "tightest" AP at the top.
    # We want the largest k such that this set is valid.
    # Since validity is monotonic with respect to removing elements (if a set is valid, any subset is valid),
    # we can search for the max k.
    
    # Start from max_size and go down
    for k in range(max_size, 0, -1):
        if time.time() > deadline - 10.0:
            break
        s = list(range(257 - k, 257))
        if is_valid_exact(s):
            best_set = s
            break
            
    if not best_set:
        return [1]

    current_best_size = len(best_set)
    target_size = current_best_size + 1

    # Phase 2: Try to find a set of size target_size
    # Strategy: Start with the best AP set (size k), and try to modify it to size k+1.
    # Modification 1: Add a smaller element.
    # Modification 2: Swap an element and add a new one.
    # Modification 3: Random local search starting from best_set or random valid sets.

    # Helper to check if we can add x to s
    def can_add(s, x):
        if x in s:
            return False
        return is_valid_exact(s + [x])

    # 2a: Try adding one element to the AP set
    seed_set = best_set[:]
    seed_set_set = set(seed_set)
    candidates_to_add = [x for x in range(1, 257 - current_best_size) if x not in seed_set_set]
    random.shuffle(candidates_to_add)
    
    found = False
    for x in candidates_to_add:
        if time.time() > deadline - 5.0:
            break
        if can_add(seed_set, x):
            best_set = seed_set + [x]
            current_best_size += 1
            found = True
            break

    # 2b: If simple add failed, try 1-swap + 1-add
    if not found and time.time() < deadline - 5.0:
        # Try random swaps in the seed set, then try to add
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            s = seed_set[:]
            s_set = set(s)
            # Swap one element
            i = random.randint(0, len(s) - 1)
            old_val = s[i]
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
            for x in add_candidates[:10]: # Limit checks per swap
                if time.time() > deadline - 5.0:
                    break
                new_s = s + [x]
                if is_valid_exact(new_s):
                    best_set = new_s
                    current_best_size += 1
                    found = True
                    break
            if found:
                break

    # 2c: Local Search / Hill Climbing to potentially find even larger sets
    # Start from current best_set
    current_set = best_set[:]
    current_set_set = set(current_set)
    
    # We will try to improve current_set by adding elements, and if stuck, by swapping.
    # To allow for larger jumps, we might also restart from random valid sets of size ~ current_best_size.
    
    for restart in range(50):
        if time.time() > deadline - 2.0:
            break
        
        if restart == 0:
            s = current_set[:]
            s_set = current_set_set
        else:
            # Generate a random valid set of similar size
            # Try APs with random start
            n_try = random.randint(max(2, current_best_size - 5), max(2, current_best_size))
            a_try = random.randint(1, max(1, 256 - n_try))
            s = list(range(a_try, a_try + n_try))
            if not is_valid_exact(s):
                s = current_set[:]
                s_set = current_set_set
                continue
            s_set = set(s)
            
        improved = True
        while improved:
            if time.time() > deadline - 2.0:
                break
            improved = False
            # Try to add an element
            # Bias towards smaller elements? Or random?
            # Smaller elements increase the chance of collision with existing sums?
            # Actually, adding small elements increases the number of subsets significantly.
            # Let's try all unused elements, prioritizing those that are "safe" (e.g., large)
            # But we want max size, so we just need any valid add.
            
            # To speed up, we can check a few random candidates first, then all.
            unused = [x for x in range(1, 257) if x not in s_set]
            random.shuffle(unused)
            
            # Try a few random ones
            for x in unused[:20]:
                if time.time() > deadline - 2.0:
                    break
                if is_valid_exact(s + [x]):
                    s.append(x)
                    s_set.add(x)
                    improved = True
                    if len(s) > len(best_set):
                        best_set = s[:]
                        current_best_size = len(s)
                    break
            
            if improved:
                continue
                
            # If no add found, try a swap to free up space
            # Swap a random element with a random unused one
            if len(s) < 2:
                break
            i = random.randint(0, len(s) - 1)
            old_val = s[i]
            candidates = [x for x in range(1, 257) if x not in s_set]
            if not candidates:
                break
            new_val = random.choice(candidates)
            
            s[i] = new_val
            s_set.remove(old_val)
            s_set.add(new_val)
            
            if is_valid_exact(s):
                # Check if we can add something now
                # (The while loop will handle the add check)
                pass
            else:
                # Revert
                s[i] = old_val
                s_set.remove(new_val)
                s_set.add(old_val)
                # If we can't even keep the same size, we might need to restart or give up on this branch
                # But we want to maximize size, so if we lose validity, we revert.
                # If we revert and can't add, we are stuck.
                # Let's try a few more swaps
                pass # The loop continues, but we haven't improved.

    # Final verification
    if not is_valid_exact(best_set):
        # Fallback to a known valid small set
        return [1]
        
    return [int(x) for x in best_set]
