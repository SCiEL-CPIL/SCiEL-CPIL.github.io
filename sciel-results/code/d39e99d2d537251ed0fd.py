import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 12
    # Fallback: powers of 2 (valid, largest = 2048)
    best = [1 << i for i in range(n)]
    best_max = max(best)
    
    random.seed(42)
    
    def greedy_construct(randomize_prob=0.0, max_offset=0):
        """
        Greedy construction using bitsets for speed.
        randomize_prob: probability of picking a non-minimal valid candidate.
        max_offset: how far to look for alternative candidates.
        """
        arr = []
        sums_bitset = 1  # Bit 0 is set, representing sum 0
        
        for k in range(n):
            # Find smallest valid candidate
            candidate = 1
            min_valid = None
            valid_candidates = []
            
            while True:
                if time.time() > deadline:
                    return None
                
                if candidate in arr:
                    candidate += 1
                    continue
                
                # Check collision using bitset
                # If (sums_bitset << candidate) & sums_bitset is non-zero, collision exists
                if ((sums_bitset << candidate) & sums_bitset) == 0:
                    min_valid = candidate
                    valid_candidates.append(candidate)
                    
                    # If we are randomizing, collect a few more valid candidates
                    if randomize_prob > 0 and len(valid_candidates) < 10:
                        # Look for a few more to have options
                        for _ in range(5):
                            candidate += 1
                            if candidate in arr:
                                continue
                            if ((sums_bitset << candidate) & sums_bitset) == 0:
                                valid_candidates.append(candidate)
                        break
                    else:
                        break
                else:
                    candidate += 1
                
                # Safety: don't loop forever
                if candidate > 10000000:
                    return None
            
            if min_valid is None:
                return None
            
            # Select candidate
            if randomize_prob > 0 and random.random() < randomize_prob and len(valid_candidates) > 1:
                # Pick a random one from the small pool
                selected = random.choice(valid_candidates)
            else:
                selected = min_valid
            
            arr.append(selected)
            # Update bitset: new sums are old sums + selected
            sums_bitset |= (sums_bitset << selected)
            
        return arr

    # First, try pure greedy (deterministic)
    result = greedy_construct(0.0, 0)
    if result is not None and len(result) == n:
        m = max(result)
        if m < best_max:
            best = result
            best_max = m
    
    # Now try randomized greedy
    # The bitset operation is very fast, so we can do many iterations
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Increase randomization slightly in later attempts
        prob = 0.1 + 0.1 * (attempt % 5) / 5.0
        
        result = greedy_construct(prob, 0)
        if result is not None and len(result) == n:
            m = max(result)
            if m < best_max:
                best = result
                best_max = m
    
    # Local search: try to reduce each element
    # We sort the array to make it easier to think about, but the order doesn't matter for the set
    best_sorted = sorted(best)
    
    # Repeat local search until no improvement or time runs out
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            
            old_val = best_sorted[i]
            # We want to try smaller values
            # To check if a new value 'v' is valid, we remove old_val, compute sums of rest,
            # then check if v causes collision.
            
            # Optimization: We can precompute sums without best_sorted[i]
            # But since n is small (12), recomputing is O(2^11) which is 2048 ops.
            # With bitsets, it's even faster.
            
            # Let's compute the bitset of sums for the set without best_sorted[i]
            sums_rest = 1
            for j in range(n):
                if j == i:
                    continue
                sums_rest |= (sums_rest << best_sorted[j])
            
            # Now try to replace best_sorted[i] with a smaller value
            # We must ensure the new value is not in the set (distinct integers)
            # and doesn't cause collision with sums_rest
            
            # We want to minimize the max element.
            # If we reduce an element, the max might reduce if it was the max.
            # Even if it's not the max, it might allow other elements to be reduced later?
            # Actually, just reducing any element is good for the "minimize largest" objective
            # only if it reduces the current maximum OR enables a configuration with a lower maximum.
            # However, simply finding a valid smaller replacement for the current max is a direct win.
            # Let's prioritize reducing the largest element first.
            
            # If i is not the index of the max, we might still improve the set, but the max stays same.
            # The objective is "Minimize the largest integer".
            # So we should focus on reducing the current maximum.
            
            # Let's identify the index of the maximum
            max_idx = best_sorted.index(max(best_sorted))
            
            if i != max_idx:
                # If we are not looking at the max, we can skip unless we want to explore
                # different local minima. But reducing non-max elements doesn't help the score directly
                # unless it unlocks a smaller max later.
                # Given time constraints, let's just focus on the max first.
                # If we have time, we can do full local search.
                pass 
            else:
                # Try to reduce the max element
                for new_val in range(old_val - 1, 0, -1):
                    if time.time() > deadline:
                        break
                    
                    if new_val in best_sorted:
                        continue
                    
                    # Check if new_val is valid with sums_rest
                    if ((sums_rest << new_val) & sums_rest) == 0:
                        # Valid!
                        best_sorted[i] = new_val
                        best_sorted.sort()
                        best_max = max(best_sorted)
                        improved = True
                        break
                # If we found a new max, the loop will restart with improved=True
                # If not, we continue to next i (which will be skipped if not max)
            
            # Actually, the logic above skips non-max indices.
            # Let's allow reducing non-max indices if we have time, as it might lead to a better global state?
            # No, the objective is strictly min(max). 
            # If I reduce a non-max element, max stays same. Score doesn't improve.
            # So I should ONLY try to reduce the current maximum.
            
            # Wait, what if reducing a non-max element allows the max to be reduced further in a subsequent step?
            # That's possible. But direct reduction of max is most effective.
            # Let's stick to reducing the max for now. If we have time, we can do a full local search.
            
            # Let's refine: 
            # 1. Try to reduce the current max.
            # 2. If successful, repeat.
            # 3. If stuck, try reducing other elements to see if it unblocks the max?
            #    This is complex. Let's just do full local search: try to reduce ANY element.
            #    If we reduce a non-max, the max doesn't change, so `improved` should only be set
            #    if the global max decreases? 
            #    No, the score is based on the returned max. 
            #    If I return a set with max 100, and I find a set with max 100 but smaller sum, score is same.
            #    So I only care about reducing the max.
            
            # Therefore, I should only attempt to replace the element at `max_idx`.
            # The loop `for i in range(n)` is inefficient if I skip non-max.
            # Let's restructure.

    # Restructured Local Search
    while time.time() < deadline:
        max_val = max(best_sorted)
        max_idx = best_sorted.index(max_val)
        
        # Compute sums without the max element
        sums_rest = 1
        for j in range(n):
            if j == max_idx:
                continue
            sums_rest |= (sums_rest << best_sorted[j])
        
        found_replacement = False
        # Try to replace max_val with a smaller value
        for new_val in range(max_val - 1, 0, -1):
            if time.time() > deadline:
                break
            
            if new_val in best_sorted:
                continue
            
            if ((sums_rest << new_val) & sums_rest) == 0:
                # Valid replacement
                best_sorted[max_idx] = new_val
                best_sorted.sort()
                found_replacement = True
                break
        
        if not found_replacement:
            # Try to reduce other elements? 
            # If we reduce a non-max element, the max stays the same.
            # However, it might allow a smaller max later.
            # Let's try to reduce the second largest, third largest, etc.
            # But this is a local search. 
            # Let's try a "shake": replace a random non-max element with a smaller valid one.
            # If it doesn't change the max, do we keep it? 
            # If we keep it, we might be in a different local minimum.
            # But we don't have memory of previous states.
            # Let's just try to reduce any element. If the max doesn't change, we don't count it as "improved" for the outer loop condition, 
            # but we might have changed the set. 
            # To avoid infinite loops, we should only accept a change if it reduces the max.
            # If it doesn't reduce the max, it's a lateral move. 
            # Given the time constraint, lateral moves are risky.
            # Let's stick to only accepting moves that reduce the max.
            
            # If we can't reduce the max, we are stuck in this local minimum.
            # We can try to restart the greedy construction with different random seeds or parameters.
            # But we already did 2000 iterations.
            
            # Let's try a few more random restarts if we have time.
            if time.time() < deadline - 5.0:
                # Do a few more randomized greedies
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    result = greedy_construct(0.3, 0)
                    if result is not None and len(result) == n:
                        m = max(result)
                        if m < best_max:
                            best = result
                            best_max = m
                            best_sorted = sorted(best)
                            # Restart local search
                            break
                else:
                    # No improvement found in restarts, break local search
                    break
            else:
                break

    return best_sorted
