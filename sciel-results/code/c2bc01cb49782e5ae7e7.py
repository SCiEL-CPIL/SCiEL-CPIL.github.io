import math

def construct():
    # 1. Identify primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(256)
    P_small = [p for p in primes if p <= 128]
    P_large = [p for p in primes if p > 128]
    
    # Map small primes to bit indices
    prime_to_idx = {p: i for i, p in enumerate(P_small)}
    num_small_primes = len(P_small)
    
    # 2. Base sum: 1 and all large primes
    base_sum = 1 + sum(P_large)
    base_items = [1] + P_large
    
    # 3. Construct list of remaining numbers R
    # Numbers 2..256 that are not in P_large
    items = []
    for x in range(2, 257):
        if x in P_large:
            continue
        mask = 0
        temp = x
        for p in P_small:
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            # This should not happen for x not in P_large
            # If it does, x has a large prime factor. 
            # As reasoned before, x must be that prime, which is in P_large.
            # So this block is effectively dead code for valid inputs.
            pass
        items.append((x, mask))
        
    # Sort items by value descending
    items.sort(key=lambda t: t[0], reverse=True)
    
    # Remove dominated items
    # If item i is dominated by item j: mask_i is subset of mask_j and val_i <= val_j
    # Then we can remove i.
    n_items = len(items)
    values = [v for v, m in items]
    masks = [m for v, m in items]
    
    kept_indices = [True] * n_items
    for i in range(n_items):
        if not kept_indices[i]:
            continue
        for j in range(n_items):
            if i == j or not kept_indices[j]:
                continue
            # Check if j dominates i
            if (masks[i] & ~masks[j]) == 0 and values[i] <= values[j]:
                # j is a superset of i's mask and has >= value
                # So i is dominated
                kept_indices[i] = False
                break
    
    filtered_items = [(values[i], masks[i]) for i in range(n_items) if kept_indices[i]]
    filtered_items.sort(key=lambda t: t[0], reverse=True)
    
    n_f = len(filtered_items)
    f_values = [v for v, m in filtered_items]
    f_masks = [m for v, m in filtered_items]
    
    # Precompute for each prime, the list of item indices divisible by it
    # This helps in calculating the bound quickly
    prime_to_indices = {p: [] for p in P_small}
    for i in range(n_f):
        m = f_masks[i]
        for p in P_small:
            if m & (1 << prime_to_idx[p]):
                prime_to_indices[p].append(i)
    
    best_sum = 0
    best_selection = []
    
    def dfs(idx, used_mask, current_sum, selected):
        nonlocal best_sum, best_selection
        
        # Pruning
        if current_sum > best_sum:
            best_sum = current_sum
            best_selection = selected.copy()
            
        if idx == n_f:
            return
            
        # Calculate upper bound
        # Sum of max compatible value for each unused prime
        bound = current_sum
        for p in P_small:
            if used_mask & (1 << prime_to_idx[p]):
                continue
            # Find max value item divisible by p, with index >= idx, and compatible with used_mask
            max_val = 0
            for j in prime_to_indices[p]:
                if j < idx:
                    continue
                if f_masks[j] & used_mask:
                    continue
                if f_values[j] > max_val:
                    max_val = f_values[j]
                    # Since items are sorted by value desc, and prime_to_indices[p] 
                    # is built in order of i (which is sorted by value desc),
                    # the first compatible one is the max?
                    # Wait, prime_to_indices[p] contains indices i.
                    # i is sorted by value desc.
                    # So the first i in the list with i >= idx and compatible is the max value.
                    # We can break?
                    # Yes, because f_values is sorted descending.
                    # So the first valid index is the one with the highest value.
                    break
            bound += max_val
            
        if bound <= best_sum:
            return
            
        # Option 1: Take item idx
        if (f_masks[idx] & used_mask) == 0:
            dfs(idx + 1, used_mask | f_masks[idx], current_sum + f_values[idx], selected + [f_values[idx]])
            
        # Option 2: Skip item idx
        dfs(idx + 1, used_mask, current_sum, selected)
        
    dfs(0, 0, 0, [])
    
    # Construct final result
    result = base_items + best_selection
    return result
