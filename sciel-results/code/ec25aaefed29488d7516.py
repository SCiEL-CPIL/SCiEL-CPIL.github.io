import time

def construct():
    deadline = time.time() + 50
    N = 512
    MAX_SIZE = 48
    
    # Greedy descending: try to add largest numbers first
    # This tends to fill the set with large numbers which are less likely to be means of others.
    # We maintain the DP of the current set.
    
    best_set = []
    
    # We will try a few different strategies for ordering if greedy descending fails to get 48
    # But let's start with pure descending.
    
    def try_greedy(order):
        chosen = []
        # dp[k] = bitset of sums of k elements from chosen
        # Max sum is 48 * 512 = 24576. Bitset size ~25k bits. 
        # Python ints handle this easily.
        dp = [0] * (MAX_SIZE + 1)
        dp[0] = 1
        
        for c in order:
            if len(chosen) >= MAX_SIZE:
                break
            m = len(chosen)
            
            # Check if c can be added
            # 1. c is not the mean of any subset of chosen
            #    sum(T) = c * |T| for T subset of chosen, |T| >= 2
            ok = True
            for k in range(2, m + 1):
                target = c * k
                if (dp[k] >> target) & 1:
                    ok = False
                    break
            
            if ok:
                # 2. No existing a is the mean of a subset of chosen + c
                #    Subsets of chosen + c not containing a.
                #    These are either subsets of chosen (already valid) 
                #    or subsets of chosen + c that include c.
                #    If a is mean of T U {c}, T subset of chosen \ {a}, |T| >= 1
                #    sum(T) + c = a * (|T| + 1)
                #    sum(T) = a * (|T| + 1) - c
                #    Let j = |T|. j ranges from 1 to m-1.
                #    We need to check if sum(T) is achievable by j elements from chosen \ {a}.
                #    This is tricky with a single global dp.
                #    However, we can approximate or use the fact that if m is small, we can check.
                #    For a fast check, we can just do the full verification at the end?
                #    No, we need incremental to prune.
                
                # Let's refine the check for existing a.
                # We can't easily remove 'a' from dp.
                # But we can check: does there exist a subset of chosen (including c) of size k >= 2, 
                # not containing a, summing to a*k?
                # This is equivalent to: 
                # (Subset of chosen of size k, sum a*k, not containing a) OR
                # (Subset of chosen of size k-1, sum a*k - c, not containing a, with c added)
                
                # Since we don't have "dp without a", we can use a necessary condition:
                # If the global dp[k] has bit a*k set, it might be using a.
                # To be safe, we can just perform the check for all a if m is small, 
                # or use a slower check only when m is large.
                
                # Given the time limit, let's assume the "c is mean of chosen" check is the main bottleneck.
                # The "a is mean of chosen+c" check is harder.
                
                # Alternative: Just add c, and if the final set is invalid, we failed.
                # But we want to maximize size.
                
                # Let's try a simpler incremental check:
                # Only check if c is mean of chosen.
                # This allows invalid sets to form where existing a becomes a mean.
                # We will verify at the end. If invalid, we might have a smaller valid subset.
                
                # To be safer, let's check the condition for the NEW element c strictly.
                # And for existing elements, we will do a full verification at the end of the greedy pass.
                
                chosen.append(c)
                # Update dp
                ndp = dp[:]
                for k in range(m, -1, -1):
                    ndp[k + 1] |= dp[k] << c
                dp = ndp
                
        return chosen

    def is_valid(arr):
        m = len(arr)
        if m < 3:
            return True
        pos = {v: i for i, v in enumerate(arr)}
        max_sum = m * N
        dp = [0] * (m + 1)
        dp[0] = 1
        for v in arr:
            for k in range(m - 1, -1, -1):
                dp[k + 1] |= dp[k] << v
        for idx, a in enumerate(arr):
            other = [arr[j] for j in range(m) if j != idx]
            d = [0] * m
            d[0] = 1
            for v in other:
                for k in range(len(other) - 1, -1, -1):
                    d[k + 1] |= d[k] << v
            for k in range(2, m):
                target = a * k
                if target > max_sum:
                    break
                if (d[k] >> target) & 1:
                    return False
        return True

    # Try descending order
    order_desc = list(range(N, 0, -1))
    candidate = try_greedy(order_desc)
    
    # If candidate is valid, return it
    if is_valid(candidate) and len(candidate) == 48:
        return candidate

    # If not, try to find the largest valid subset of candidate
    # Or try other orders.
    
    # Let's try ascending order too
    order_asc = list(range(1, N + 1))
    candidate_asc = try_greedy(order_asc)
    if is_valid(candidate_asc) and len(candidate_asc) > len(candidate):
        candidate = candidate_asc

    # If we still don't have 48, we can try to prune the invalid candidate to make it valid
    # Or just return the best valid one found.
    
    # To ensure we return a valid set, we must verify.
    # The greedy "c is mean of chosen" check is necessary but not sufficient for the whole set.
    # So candidate might be invalid.
    
    # Let's do a quick local search to fix validity if candidate is invalid.
    # Or simply, if candidate is invalid, we return a known valid small set?
    # No, we want to maximize size.
    
    # Let's assume the greedy descending produces a valid set or close to it.
    # If not, we can try to remove elements that cause violations.
    
    if not is_valid(candidate):
        # Remove elements that are means of others
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(candidate) - 1, -1, -1):
                a = candidate[i]
                others = [x for x in candidate if x != a]
                # Check if a is mean of any subset of others
                m_o = len(others)
                dp_o = [0] * (m_o + 1)
                dp_o[0] = 1
                for v in others:
                    for k in range(m_o - 1, -1, -1):
                        dp_o[k + 1] |= dp_o[k] << v
                is_mean = False
                for k in range(2, m_o):
                    if (dp_o[k] >> (a * k)) & 1:
                        is_mean = True
                        break
                if is_mean:
                    candidate.pop(i)
                    changed = True
                    break # Restart check from beginning as set changed
            
            # Also check if removing a makes other elements invalid? 
            # No, removing elements cannot create new violations for remaining elements?
            # Wait. If a is removed, subsets for other elements lose 'a'.
            # So if b was NOT a mean before, it might become a mean? 
            # No, the set of subsets shrinks. If b was not a mean of any subset, 
            # it is still not a mean of any subset of a smaller set.
            # So removing elements preserves validity of the remaining elements.
            # So we just need to remove all elements that are currently means.
    
    return candidate
