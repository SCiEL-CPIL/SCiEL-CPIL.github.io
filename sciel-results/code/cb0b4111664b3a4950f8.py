import time

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Generate all divisors of 5040
    n = 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    total_divs = len(divisors)
    target = 4979
    
    # Precompute suffix sums for pruning: suffix_max[i] = sum of divisors[i:]
    # Actually, for "can we make remainder with exactly k items from divisors[i:]":
    # max possible = sum of largest k from divisors[i:]
    # min possible = sum of smallest k from divisors[i:]
    # We'll compute these on the fly or precompute.
    
    # Precompute: for each index i and each k, the max and min sum achievable
    # with exactly k distinct elements from divisors[i:]
    # divisors is sorted descending, so divisors[i:] is also descending.
    # Max sum with k items from divisors[i:] = sum(divisors[i:i+k])
    # Min sum with k items from divisors[i:] = sum(divisors[total_divs-k:total_divs]) if i+k <= total_divs... 
    # Actually min is the sum of the k smallest from divisors[i:], which are the last k in divisors[i:]
    # i.e., divisors[total_divs-k : total_divs] if total_divs - i >= k
    
    # Let's precompute prefix sums for quick range sums
    # divisors sorted descending: d[0] >= d[1] >= ... >= d[m-1]
    m = total_divs
    # prefix[i] = sum of divisors[0:i]
    prefix = [0] * (m + 1)
    for i in range(m):
        prefix[i + 1] = prefix[i] + divisors[i]
    
    # sum of divisors[i:i+k] = prefix[i+k] - prefix[i]
    # sum of k smallest from divisors[i:] = sum of divisors[m-k:m] if m-i >= k
    # but that's only valid if i <= m-k, i.e., the k smallest overall are available
    # More precisely: k smallest from divisors[i:] are divisors[m-k:m] if i <= m-k
    # Otherwise, it's more complex. Let's just use: the k smallest from divisors[i:] 
    # are the last k elements of divisors[i:], i.e., divisors[m-k:m] if m-i>=k and i<=m-k
    # Actually divisors[i:] has (m-i) elements. The k smallest are the last k: 
    # indices from m-k to m-1, but only if i <= m-k. If i > m-k, then all of divisors[i:] 
    # are among the smallest, but we need exactly k, so it's divisors[m-k:m] only if m-k >= i.
    # Let me think again: divisors[i:] = [d_i, d_{i+1}, ..., d_{m-1}]
    # The k smallest in this subarray are d_{m-k}, ..., d_{m-1} if i <= m-k
    # If i > m-k, then the k smallest are d_i, ..., d_{i+k-1}? No, that's the k LARGEST.
    # The k smallest from a sorted-descending array starting at i are the last k elements:
    # d_{m-k} through d_{m-1}, provided that m-k >= i, i.e., i <= m-k.
    # If i > m-k, then we have fewer than k elements after position m-k, so the k smallest 
    # would be... actually if i > m-k, then m-i < k, meaning we don't have k elements. 
    # But we only call this when m-i >= k.
    # So if m-i >= k, the k smallest are d_{m-k}..d_{m-1} if i <= m-k, 
    # else they are d_i..d_{i+k-1}? No.
    
    # Let me just precompute a 2D table or compute on the fly.
    # For pruning, I'll compute:
    # max_sum(i, k) = sum of k largest from divisors[i:] = sum(divisors[i:i+k]) = prefix[i+k]-prefix[i]
    # min_sum(i, k) = sum of k smallest from divisors[i:] = sum(divisors[m-k:m]) if i <= m-k
    #                = sum(divisors[m-k:m]) is constant! Because the k smallest overall 
    #                are always d_{m-k}..d_{m-1}. But are they all in divisors[i:]? 
    #                Only if i <= m-k. If i > m-k, then some of the k smallest are excluded.
    #                In that case, the k smallest from divisors[i:] are d_i..d_{i+k-1}? 
    #                No, divisors is descending, so d_i > d_{i+1} > ... 
    #                The k smallest from divisors[i:] are the last k: d_{m-k}..d_{m-1} 
    #                BUT only those with index >= i. So if i > m-k, then the k smallest 
    #                from divisors[i:] are d_i, d_{i+1}, ..., d_{i+k-1}? No, those are 
    #                the k LARGEST from divisors[i:].
    
    # I'm overcomplicating. Let me just compute min_sum(i,k) as the sum of the k smallest 
    # elements in divisors[i:]. Since divisors[i:] is sorted descending, the k smallest 
    # are the last k elements: divisors[m-k:m] if m-k >= i, otherwise... 
    # if m-k < i, then divisors[i:] has m-i elements, all >= d_i > d_{m-k}. 
    # The k smallest would be divisors[m-k:m] but that's not in the subarray. 
    # Actually, if i > m-k, then the subarray divisors[i:] starts after position m-k, 
    # so the smallest k elements in divisors[i:] are divisors[m-k:m] ∩ divisors[i:] 
    # = divisors[i:i+k] if i+k <= m? No, those are the LARGEST k.
    
    # OK let me just directly compute: the k smallest from divisors[i:] are 
    # divisors[m-k:m] if i <= m-k, else divisors[i:m-k+1]... this is getting messy.
    
    # Simpler approach: just use a loose bound. 
    # min possible sum with k items from divisors[i:] >= k * divisors[m-1] (the smallest divisor)
    # max possible sum with k items from divisors[i:] <= k * divisors[i] (the largest available)
    # These are loose but valid. For tighter bounds, I can use:
    # max = sum(divisors[i:i+k])
    # min = sum(divisors[m-k:m]) if i <= m-k else sum(divisors[i:i+k]) ... no.
    
    # Let me use: min_sum = sum of k smallest in divisors[i:] = sum(divisors[m-k:m]) if m-k >= i
    # else, if i > m-k, then the k smallest in divisors[i:] are divisors[m-k:m] but clipped... 
    # Actually, if i > m-k, then divisors[i:] = divisors[i:m], and the k smallest are 
    # divisors[m-k:m] only if m-k >= i. If m-k < i, then all of divisors[i:] are > d_{m-k}, 
    # so the k smallest are d_i, d_{i+1}, ..., d_{i+k-1}? No, that's wrong because 
    # divisors is descending, so d_i is the LARGEST in divisors[i:].
    
    # I give up on tight min. Let me use: min_sum(i,k) = k * divisors[-1] (very loose)
    # and max_sum(i,k) = sum(divisors[i:i+k]) (tight)
    # The max bound is the most useful for pruning when remainder is large.
    
    def dfs(rem, k, start):
        """Try to pick exactly k divisors from divisors[start:] summing to rem."""
        if time.time() > deadline:
            return None
        if k == 0:
            return [] if rem == 0 else None
        if rem < 0:
            return None
        # Not enough elements left
        if m - start < k:
            return None
        # Max possible sum with k items from divisors[start:]
        if start + k > m:
            return None
        max_possible = prefix[start + k] - prefix[start]
        if rem > max_possible:
            return None
        # Min possible sum: at least k * smallest divisor
        min_possible = k * divisors[-1]
        if rem < min_possible:
            return None
        # Also, rem must be achievable; tight bound: rem <= max_possible already checked.
        # Try each candidate from divisors[start:]
        for idx in range(start, m - k + 1):
            d = divisors[idx]
            if d > rem:
                continue
            # After picking d, need k-1 more from divisors[idx+1:] summing to rem-d
            sub = dfs(rem - d, k - 1, idx + 1)
            if sub is not None:
                return [d] + sub
        return None
    
    # Iterative deepening on k
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
        result = dfs(target, k, 0)
        if result is not None:
            return result
    
    # Fallback: return any valid solution (should not reach here)
    # Greedy as fallback
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
