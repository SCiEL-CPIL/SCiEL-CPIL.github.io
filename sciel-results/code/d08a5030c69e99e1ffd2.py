import time
import random

def _fast_pair_check(S, x):
    """
    Check if adding x to S creates a pair-violation (k=2).
    1. x is mean of a, b in S: 2x = a + b
    2. y in S is mean of x, z in S: 2y = x + z
    Returns True if violation found (invalid), False otherwise.
    """
    S_set = set(S)
    
    # 1. x is mean of a, b in S
    for a in S:
        b = 2 * x - a
        if b in S_set and b != a:
            return True
            
    # 2. Some a in S is mean of x, b in S
    # 2a = x + b => b = 2a - x
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a and b != x:
            return True
            
    return False

def _is_valid_set(S):
    """
    Full check using bitset DP.
    Returns True if valid.
    """
    if len(S) < 3:
        return True
        
    S = list(S)
    N = len(S)
    
    # For each element, check if it's the mean of any subset of others
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        # DP for subset sums by size
        # dp[k] is an integer where bit s is set if sum s is achievable with k elements
        max_sum = sum(others)
        
        # Initialize dp
        # dp[0] = 1 << 0 (sum 0 with 0 elements)
        dp = [0] * (n_others + 1)
        dp[0] = 1  # bit 0 set
        
        for val in others:
            # Update dp in reverse to avoid reusing same element
            for k in range(n_others - 1, -1, -1):
                if dp[k] == 0:
                    continue
                # Shift by val
                shifted = dp[k] << val
                dp[k+1] |= shifted
                
        # Check if any dp[k] for k >= 2 has a bit set at position k*a
        for k in range(2, n_others + 1):
            if dp[k] == 0:
                continue
            target_sum = k * a
            if target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
            # If target_sum > max_sum, it's impossible for this k and any larger k?
            # No, max_sum is fixed (sum of all others).
            # But for larger k, the sum of k elements is a subset of the sum of all others?
            # Actually, sum of k elements <= sum of all others is NOT always true if we pick the largest k.
            # Wait, sum of ANY k elements <= sum of ALL elements (since all positive).
            # So if k*a > sum(others), it's impossible.
            # And since a is fixed, if k*a > sum(others), then (k+1)*a > sum(others) as well.
            # So we can break.
            else:
                break
                
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Generate multiple orders to try
    orders = []
    
    # 1. Descending (often good for sum constraints)
    orders.append(candidates[::-1])
    
    # 2. Ascending
    orders.append(candidates[:])
    
    # 3. Interleaved (High-Low)
    inter = []
    lo, hi = 1, 512
    while lo <= hi:
        inter.append(hi)
        if lo != hi:
            inter.append(lo)
        lo += 1
        hi -= 1
    orders.append(inter)
    
    # 4. Random orders
    rng = random.Random(42)
    # We can generate more random orders if we have time
    # But let's start with a few deterministic ones
    
    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
            
        current_set = []
        
        for x in order:
            if time.time() - start_time > deadline - 2:
                break
                
            if len(current_set) >= 48:
                break
                
            # Fast pair check
            if _fast_pair_check(current_set, x):
                continue
                
            # Full check on current_set + [x]
            test_set = current_set + [x]
            
            if _is_valid_set(test_set):
                current_set.append(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        # If we found 48, we can stop searching
        if len(best_set) >= 48:
            break
            
    # If we haven't found 48, try more random restarts
    # This is the "refinement machinery" / extra search
    while time.time() - start_time < deadline - 2:
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        
        current_set = []
        
        for x in rand_order:
            if time.time() - start_time > deadline - 2:
                break
            if len(current_set) >= 48:
                break
                
            # Fast pair check
            if _fast_pair_check(current_set, x):
                continue
                
            test_set = current_set + [x]
            
            if _is_valid_set(test_set):
                current_set.append(x)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
        if len(best_set) >= 48:
            break
            
    # Final verification
    if _is_valid_set(best_set):
        return best_set
    else:
        # Fallback to a known small valid set
        # Powers of 2 are always valid?
        # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        # Check: 2 is mean of 1,3 (3 not in set).
        # In powers of 2, can $2^i$ be mean of subset of others?
        # Sum of subset of powers of 2 has unique binary representation.
        # $k \cdot 2^i$ has binary representation of $k$ shifted by $i$.
        # The sum of distinct powers of 2 (excluding $2^i$) will have bits set only where the selected powers are.
        # For the sum to be $k \cdot 2^i$, the binary representation of the sum must match $k \ll i$.
        # This implies that the selected powers must be exactly the bits of $k$ shifted by $i$.
        # But the selected powers are from $\{2^j \mid j \neq i\}$.
        # So the bits of the sum are at positions $j$.
        # The bits of $k \ll i$ are at positions $i + \text{pos}$.
        # So we need $j = i + \text{pos}$ for all set bits in $k$.
        # This means the selected elements are $\{2^{i+p} \mid p \in \text{bits}(k)\}$.
        # These elements must be in the set and distinct from $2^i$.
        # If $k=1$, sum is $2^i$. But we need subset size $\ge 2$. So $k \ge 2$.
        # If $k=2$, sum is $2^{i+1}$. We need to select elements summing to $2^{i+1}$.
        # Only way to sum to $2^{i+1}$ using powers of 2 is to select $2^{i+1}$ itself.
        # But we need $\ge 2$ elements. So impossible.
        # If $k=3$, sum is $3 \cdot 2^i = 2^i + 2^{i+1}$.
        # We need to select $2^i$ and $2^{i+1}$. But $2^i$ is the mean element, so it's excluded.
        # So we can't select $2^i$.
        # Thus, we cannot form the sum $3 \cdot 2^i$ without $2^i$.
        # Generally, the sum of a subset of powers of 2 (excluding $2^i$) cannot be a multiple of $2^i$ with quotient $k$ requiring the bit $i$ to be set in $k$'s binary representation?
        # Actually, if $k$ is odd, $k \cdot 2^i$ has the bit $i$ set.
        # The sum of other powers of 2 (indices $\neq i$) will have bit $i$ unset (since no other power has bit $i$ set, and sums of distinct powers don't carry over to bit $i$ unless we sum $2^i$ itself, which is excluded).
        # Wait, if we sum $2^{i-1} + 2^{i-1}$? No, distinct elements.
        # So bit $i$ in the sum of others is always 0.
        # Therefore, the sum of others can never be equal to $k \cdot 2^i$ if $k$ is odd.
        # If $k$ is even, $k \cdot 2^i = (k/2) \cdot 2^{i+1}$. The bit $i$ is 0.
        # So it's possible?
        # Example: $S = \{1, 2, 4\}$. $x=2$.
        # Others: 1, 4.
        # Subset $\{1, 4\}$ sum 5. $k=2$. $2 \cdot 2 = 4 \neq 5$.
        # Subset $\{1\}$ size 1.
        # Subset $\{4\}$ size 1.
        # So 2 is valid.
        # $x=1$. Others 2, 4.
        # Subset $\{2, 4\}$ sum 6. $k=2$. $2 \cdot 1 = 2 \neq 6$.
        # So powers of 2 are valid.
        
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
