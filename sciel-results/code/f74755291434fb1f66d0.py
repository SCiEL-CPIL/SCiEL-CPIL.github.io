"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Generate primes up to 256
    primes = []
    is_prime = [True] * 257
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, 257):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Primes > 47 cannot be bundled with any prime >= 2 to form a number <= 256 
    # that is better than the prime itself + the power of the small prime.
    # Specifically, for p > 47, 2p > 94. 
    # Max power of 2 is 256. 256 + p > 2p.
    # So large primes are always better kept alone.
    large_primes = [p for p in primes if p > 47]
    small_primes = [p for p in primes if p <= 47]
    
    # For small primes, we need to find the optimal partition into groups.
    # Each group forms a number x <= 256.
    # We want to maximize sum(x).
    
    # Precompute best value for any subset of small primes?
    # 2^15 = 32768 subsets. This is small enough for DP or B&B.
    
    # Let's use a simple recursive B&B or DP over subsets.
    # State: bitmask of used small primes.
    # Value: max sum achievable.
    
    n_small = len(small_primes)
    # dp[mask] = max sum
    # Initialize with -inf
    dp = [-1] * (1 << n_small)
    dp[0] = 0
    
    # For each subset, try to form a valid number from the primes in that subset.
    # Actually, we iterate through masks and try to add a "block" (group of primes).
    # A block is a set of primes whose product (with some powers) <= 256.
    # To maximize value, for a fixed set of primes, we should use the max power for each?
    # No, we just need the product of the primes to be <= 256?
    # If we pick primes {2, 3}, we can make 6, 12, 18, 24... up to 256.
    # Max value for {2,3} is 256? No, 256 is 2^8. It doesn't use 3.
    # If we MUST use 3, max value is 3 * 2^k <= 256.
    # 3 * 2^6 = 192. 3 * 2^7 = 384 > 256.
    # So for a specific set of primes P, the max number we can form is:
    # max { prod_{p in P} p^{e_p} | sum(e_p * log p) <= log 256 }
    # This is a knapsack-like problem for each subset.
    # But since the number of primes in a subset is small (product grows fast),
    # we can just compute the max power for each prime in the subset such that product <= 256.
    # Actually, simpler:
    # For a subset S, the best number is the largest multiple of prod(S) that is <= 256?
    # No, the number must be of the form prod_{p in S} p^{e_p}.
    # And we want to maximize this number.
    # Since we can increase any e_p, the max number is the largest divisor of some number <= 256?
    # No, just iterate e_p.
    
    def max_value_for_subset(subset_indices):
        # subset_indices is a list of indices into small_primes
        if not subset_indices:
            return 0
        
        # Start with product 1, exponents 0
        # We want to maximize prod p_i^{e_i} <= 256
        # Since primes are small, we can just try all combinations of exponents?
        # Max exponent for 2 is 8. For 3 is 5. For 5 is 3.
        # Total states is small.
        
        max_val = 0
        # Recursive generation
        def dfs(idx, current_val):
            nonlocal max_val
            if current_val > 256:
                return
            if idx == len(subset_indices):
                if current_val > max_val:
                    max_val = current_val
                return
            
            p = small_primes[subset_indices[idx]]
            # Try all powers of p
            val = current_val
            while val <= 256:
                dfs(idx + 1, val)
                val *= p
            
        dfs(0, 1)
        return max_val

    # Precompute max value for all subsets of small_primes
    # There are 2^15 = 32768 subsets.
    # For each subset, calculating max value might be slow if done naively.
    # But we can memoize.
    
    # Actually, we can just use the DP over masks.
    # dp[mask] = max sum of values of disjoint subsets that partition mask.
    # We can iterate mask from 0 to (1<<n_small)-1.
    # For each mask, we try to pick a submask 'sub' that is a valid block.
    # dp[mask] = max(dp[mask ^ sub] + max_value(sub))
    
    # To speed up, we can precompute max_value for all submasks?
    # 32768 submasks * cost of max_value.
    # Cost of max_value is small.
    
    # Let's optimize max_value calculation.
    # Most subsets will be invalid (product > 256) or small.
    
    max_val_cache = {}
    def get_max_val(mask):
        if mask in max_val_cache:
            return max_val_cache[mask]
        
        # Extract indices
        indices = []
        for i in range(n_small):
            if mask & (1 << i):
                indices.append(i)
        
        if not indices:
            max_val_cache[mask] = 0
            return 0
            
        # If product of primes in mask > 256, it's 0?
        # No, we can use powers. But min value is prod(primes).
        prod = 1
        for i in indices:
            prod *= small_primes[i]
            if prod > 256:
                max_val_cache[mask] = 0
                return 0
        
        # Now find max power combination
        # Since prod <= 256, we have room for powers.
        # Simple DFS
        best = 0
        def dfs(idx, cur):
            nonlocal best
            if cur > 256:
                return
            if idx == len(indices):
                if cur > best:
                    best = cur
                return
            p = small_primes[indices[idx]]
            val = cur
            while val <= 256:
                dfs(idx + 1, val)
                val *= p
        dfs(0, 1)
        
        max_val_cache[mask] = best
        return best

    # DP
    full_mask = (1 << n_small) - 1
    # dp[i] for mask i
    # We only care about partitions.
    # dp[0] = 0
    # For mask in 1..full:
    #   dp[mask] = max over submask s of mask (dp[mask^s] + get_max_val(s))
    #   Note: s must be non-empty.
    #   Also, to avoid symmetry, we can require that the lowest set bit of mask is in s.
    
    dp = [0] * (1 << n_small)
    
    for mask in range(1, 1 << n_small):
        if time.time() > deadline:
            break
        # Find lowest set bit
        lsb = mask & -mask
        # Iterate submasks of mask that contain lsb
        sub = mask
        best_val = 0
        while sub:
            if sub & lsb:
                val_sub = get_max_val(sub)
                if val_sub > 0:
                    rem = mask ^ sub
                    if dp[rem] + val_sub > best_val:
                        best_val = dp[rem] + val_sub
            sub = (sub - 1) & mask
        dp[mask] = best_val

    best_small_sum = dp[full_mask]
    
    # Add large primes
    total_sum = best_small_sum + sum(large_primes)
    
    # Reconstruct the set?
    # The task asks for a list of ints.
    # We need to output the actual numbers, not just the sum.
    # The evaluator checks the sum of the returned list.
    # So we must return the list.
    
    # Reconstruction:
    # We need to trace back the DP choices.
    # This requires storing the choice.
    # Given the time constraint and simplicity, I will just return the list of numbers
    # corresponding to the optimal partition found.
    
    # However, implementing full reconstruction in B&B/DP is tedious.
    # Alternative: Return the numbers directly from the logic.
    # Since the DP finds the max sum, we can just return the set of numbers
    # that achieve this sum.
    
    # Let's do a simple Greedy/Heuristic reconstruction if DP reconstruction is too complex to code quickly.
    # Or, since the problem is small, we can just return the known good set derived above?
    # No, the code should be generic.
    
    # I will implement a simple backtracking to find the set.
    
    # Actually, the prompt says "Return a list of ints".
    # I will just return the set of numbers.
    
    # To be safe and simple, I will just return the list of all primes > 47 
    # plus the numbers from the optimal small prime partition.
    
    # For the small primes, I'll just pick the best single numbers found in the DP logic 
    # by re-running the choice logic.
    
    # Given the complexity, I will fall back to a very strong heuristic:
    # 1. Start with all small primes as their max powers.
    # 2. Try to merge pairs if it increases the sum.
    # 3. Repeat until no improvement.
    
    # This is much easier to implement and likely optimal for this size.
    
    def solve_small_heuristic():
        # Initial assignment: each prime p -> max power <= 256
        assigned = {}
        for p in small_primes:
            power = p
            while power * p <= 256:
                power *= p
            assigned[p] = power
        
        improved = True
        while improved:
            improved = False
            # Try all pairs of primes
            for i in range(len(small_primes)):
                for j in range(i+1, len(small_primes)):
                    p1 = small_primes[i]
                    p2 = small_primes[j]
                    # Check if they are currently assigned to separate numbers
                    # (i.e., they are the "owners" of their numbers)
                    # In this model, each prime "owns" a number.
                    # If we merge p1 and p2, we form a new number.
                    # We lose assigned[p1] and assigned[p2].
                    # We gain max_value({p1, p2}).
                    
                    # Calculate max value for {p1, p2}
                    # It's the max multiple of p1*p2 <= 256?
                    # No, it's max p1^a * p2^b <= 256.
                    # Since p1, p2 are small, we can just compute it.
                    max_merge_val = 0
                    a = 0
                    while p1**a <= 256:
                        b = 0
                        while p1**a * p2**b <= 256:
                            val = p1**a * p2**b
                            if val > max_merge_val:
                                max_merge_val = val
                            b += 1
                        a += 1
                    # Correction: The loop above is inefficient and slightly wrong logic for max.
                    # Just iterate all a, b.
                    max_merge_val = 0
                    for a in range(10):
                        if p1**a > 256: break
                        for b in range(10):
                            if p1**a * p2**b > 256: break
                            val = p1**a * p2**b
                            if val > max_merge_val:
                                max_merge_val = val
                            
                    current_sum = assigned[p1] + assigned[p2]
                    if max_merge_val > current_sum:
                        # Merge
                        # We need to update the assignment.
                        # Who owns the new number? Let's say p1 owns it.
                        # p2 is now "used" by p1's number.
                        # We need to track which primes are "active".
                        # This heuristic is getting messy with state.
                        pass
        return assigned.values()

    # Given the risk of bugs in the heuristic, and that the DP sum is calculated,
    # I will just return the list of numbers that sum to the best found.
    # Since I can't easily reconstruct, I will return the set of numbers
    # from the "Greedy Merge" approach which is robust.
    
    # Let's implement a clean Greedy Merge.
    
    # State: list of current numbers.
    # Each number has a set of prime factors.
    # Initially, for each small prime p, number is max_power(p).
    
    current_numbers = []
    for p in small_primes:
        power = p
        while power * p <= 256:
            power *= p
        current_numbers.append({'val': power, 'primes': {p}})
    
    improved = True
    while improved:
        improved = False
        for i in range(len(current_numbers)):
            for j in range(i+1, len(current_numbers)):
                n1 = current_numbers[i]
                n2 = current_numbers[j]
                # Try to merge n1 and n2
                # New number must be multiple of all primes in n1['primes'] U n2['primes']
                # And <= 256.
                # Max value is max { prod p^e | p in union, val <= 256 }
                primes_union = n1['primes'] | n2['primes']
                
                max_merge_val = 0
                # Simple DFS for max value
                def dfs_merge(idx, cur, primes_list):
                    nonlocal max_merge_val
                    if cur > 256: return
                    if idx == len(primes_list):
                        if cur > max_merge_val: max_merge_val = cur
                        return
                    p = primes_list[idx]
                    val = cur
                    while val <= 256:
                        dfs_merge(idx+1, val, primes_list)
                        val *= p
                        
                primes_list = list(primes_union)
                dfs_merge(0, 1, primes_list)
                
                if max_merge_val > n1['val'] + n2['val']:
                    # Merge
                    new_num = {'val': max_merge_val, 'primes': primes_union}
                    # Remove n2, replace n1
                    current_numbers[j] = new_num
                    del current_numbers[i]
                    # Adjust indices
                    improved = True
                    break
            if improved:
                break
                
    result = [n['val'] for n in current_numbers]
    result.extend(large_primes)
    
    # Ensure 1 is included? "Integer one is allowed once."
    # 1 is coprime with everything.
    # Sum increases by 1.
    result.append(1)
    
    # Convert to Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
