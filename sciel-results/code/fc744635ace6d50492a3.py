import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth.sort()
    n_smooth = len(smooth)

    # 2. Precompute conflict masks
    # conflict_mask[i] has bit j set if smooth[i] and smooth[j] are in conflict (one divides the other)
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                    conflict_mask[i] |= (1 << j)

    # 3. Enumerate all antichains with sum <= 128
    # best_for_sum[sum] = (min_size, mask)
    best_for_sum = [None] * 129
    
    # Optimization: Precompute the minimum possible sum for any antichain of size k? 
    # Not strictly necessary if we prune by sum > 128.
    
    def generate_antichains(start_idx, current_mask, current_sum, current_size):
        # Pruning: if sum exceeds 128, stop
        if current_sum > 128:
            return
        
        # If we have a non-empty set, it's a valid antichain candidate
        if current_size > 0:
            if best_for_sum[current_sum] is None or current_size < best_for_sum[current_sum][0]:
                best_for_sum[current_sum] = (current_size, current_mask)
        
        # If we've used up all numbers or time is up, stop
        if start_idx >= n_smooth:
            return
            
        if time.time() > deadline:
            return

        for i in range(start_idx, n_smooth):
            # Check if smooth[i] conflicts with any currently selected number
            if current_mask & conflict_mask[i]:
                continue
            
            # Additional pruning: 
            # Even if we add smooth[i], the minimum sum of the remaining 
            # required elements (if any) might exceed 128? 
            # Since we are just enumerating all subsets, we don't have a fixed target size k.
            # We just need to ensure we don't exceed 128.
            
            new_mask = current_mask | (1 << i)
            new_sum = current_sum + smooth[i]
            
            # If adding this number already exceeds 128, and since smooth numbers are positive,
            # any further additions will also exceed 128. We can break if sorted ascending?
            # Yes, smooth is sorted. If smooth[i] causes sum > 128, then smooth[j] for j>i will too.
            if new_sum > 128:
                break
                
            generate_antichains(i + 1, new_mask, new_sum, current_size + 1)

    # Start enumeration
    # Note: We start with empty set, sum=0, size=0
    generate_antichains(0, 0, 0, 0)

    # 4. Construct answers
    answers = [None] * 129
    
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            size, mask = best_for_sum[n]
            terms = []
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                idx = lsb.bit_length() - 1
                terms.append(smooth[idx])
                temp_mask ^= lsb
            answers[n] = terms
        else:
            # Fallback: This shouldn't happen for 1..128 if enumeration is complete,
            # but just in case, try a simple search or leave as None (which fails).
            # However, every target 1..128 is representable (e.g., by itself if smooth, or combos).
            # If a target is not found, it means no antichain sums to it.
            # But the problem states "All targets require valid representations".
            # So this branch should ideally not be taken.
            # If it is, we try a brute force DFS for that specific n as a last resort.
            # But given the enumeration, this is highly unlikely.
            pass

    # Verification and fallback for any missing
    for n in range(1, 129):
        if answers[n] is None:
            # Try to find any representation, not necessarily minimal, to avoid score 0
            # We can use a simple DFS that doesn't care about minimality first, 
            # or just accept that enumeration missed it (unlikely).
            # Let's do a quick DFS for minimal k again for this specific n
            found = False
            for k in range(1, 31):
                if time.time() > deadline:
                    break
                memo = {}
                def dfs(remaining, k_left, start_idx, mask):
                    if remaining < 0 or k_left < 0:
                        return None
                    if k_left == 0:
                        return () if remaining == 0 else None
                    if start_idx + k_left > n_smooth:
                        return None
                    # Simple bound: min sum of k_left smallest available
                    # This is an approximation, but helps pruning
                    key = (remaining, k_left, start_idx, mask)
                    if key in memo:
                        return memo[key]
                    
                    for i in range(start_idx, n_smooth):
                        val = smooth[i]
                        if val > remaining:
                            break
                        if mask & conflict_mask[i]:
                            continue
                        res = dfs(remaining - val, k_left - 1, i + 1, mask | (1 << i))
                        if res is not None:
                            memo[key] = (i,) + res
                            return (i,) + res
                    memo[key] = None
                    return None
                
                res_idx = dfs(n, k, 0, 0)
                if res_idx is not None:
                    answers[n] = [smooth[i] for i in res_idx]
                    found = True
                    break
            
            if not found:
                # Last resort: if n is smooth, use [n]
                if n in smooth:
                    answers[n] = [n]
                else:
                    # This should not happen
                    answers[n] = [] # Invalid, but prevents crash

    return [answers[n] for n in range(1, 129)]
