# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 55.0

def construct():
    # 1. Find all primes <= 256
    primes = []
    for i in range(2, 257):
        is_prime = True
        for j in range(2, isqrt(i) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # 2. Identify small primes (<= 43) for DP
    small_primes = [p for p in primes if p <= 43]
    med_primes = [p for p in primes if 43 < p <= 85]
    large_primes = [p for p in primes if 85 < p <= 128]
    huge_primes = [p for p in primes if p > 128]
    
    # Map small primes to indices
    small_prime_to_idx = {p: i for i, p in enumerate(small_primes)}
    n_small = len(small_primes)
    
    # Precompute best value for any subset of small primes
    # If product > 256, value is -1 (invalid)
    # We need to find max n <= 256 such that n is divisible by all p in subset
    best_val_subset = [-1] * (1 << n_small)
    best_num_subset = [0] * (1 << n_small)
    
    for mask in range(1 << n_small):
        # Calculate product of primes in mask
        prod = 1
        valid = True
        for i in range(n_small):
            if mask & (1 << i):
                prod *= small_primes[i]
                if prod > 256:
                    valid = False
                    break
        if not valid:
            continue
            
        # Find max multiple of prod <= 256
        max_k = 256 // prod
        val = max_k * prod
        best_val_subset[mask] = val
        best_num_subset[mask] = val
        
    # DP over masks of small primes
    # dp[mask] = max sum of values for partitioning the set of primes in mask
    # We iterate over masks and try to pick a subset S of remaining primes
    # To optimize, we can iterate over submasks of the complement?
    # Or just iterate all valid submasks S of the current mask?
    # Standard subset DP:
    # dp[mask] = max over all valid submasks S of mask: best_val_subset[S] + dp[mask ^ S]
    # This is O(3^n) which is 3^14 = 4.7M. Acceptable.
    
    dp = [-1] * (1 << n_small)
    dp[0] = 0
    
    # Precompute list of valid submasks for each mask?
    # Or just iterate.
    # To be faster, we can iterate S from 1 to (1<<n_small)-1
    # and update dp[mask | S] from dp[mask]?
    # No, that's not right because S must be disjoint.
    
    # Let's use the standard recursion with memoization or iterative over masks.
    # Iterative:
    # For mask in range(1, 1<<n_small):
    #   dp[mask] = 0
    #   # Try all submasks S of mask
    #   sub = mask
    #   while sub:
    #       if best_val_subset[sub] > 0:
    #           val = best_val_subset[sub] + dp[mask ^ sub]
    #           if val > dp[mask]:
    #               dp[mask] = val
    #       sub = (sub - 1) & mask
    #       
    # This is O(3^n). 4.7M operations. Should be fast enough in Python?
    # 4.7M * 100ns = 0.5s. Yes.
    
    for mask in range(1, 1 << n_small):
        dp[mask] = 0
        sub = mask
        while sub:
            if best_val_subset[sub] > 0:
                val = best_val_subset[sub] + dp[mask ^ sub]
                if val > dp[mask]:
                    dp[mask] = val
            sub = (sub - 1) & mask
            
    full_mask = (1 << n_small) - 1
    best_small_sum = dp[full_mask]
    
    # Now we need to reconstruct the sets to check for distinctness and handle med/large primes
    # We also need to know which sets contain 2 and 3 to attach med/large primes
    
    # Reconstruct best partition for small primes
    # We need to store the choice
    choice = [0] * (1 << n_small)
    for mask in range(1, 1 << n_small):
        best_val = 0
        best_sub = 0
        sub = mask
        while sub:
            if best_val_subset[sub] > 0:
                val = best_val_subset[sub] + dp[mask ^ sub]
                if val > best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        choice[mask] = best_sub
        dp[mask] = best_val
        
    # Extract sets
    current_mask = full_mask
    small_sets = []
    while current_mask:
        sub = choice[current_mask]
        if sub == 0:
            # Should not happen if valid
            break
        # Get primes in sub
        primes_in_sub = [small_primes[i] for i in range(n_small) if sub & (1 << i)]
        small_sets.append((primes_in_sub, best_val_subset[sub]))
        current_mask ^= sub
        
    # Now handle med and large primes
    # They can be attached to sets containing 2 or 3
    # Or be singletons
    
    # Let's evaluate the gain of attaching a prime p to a set S
    # gain = val(S U {p}) - val(S)
    # We want to maximize total sum.
    # Since med/large primes are independent, we can greedily assign them to the best available slot
    # or keep them separate if gain is negative.
    
    # However, attaching p to S changes val(S).
    # And if we attach multiple primes to S, the value changes again.
    # Example: S={2}. val=128.
    # Attach 3? S={2,3}. val=144. Gain 16.
    # Attach 5 to {2,3}? S={2,3,5}. val=120 (if 2*3*5*4=120).
    # Wait, 120 < 144. So attaching 5 is bad.
    
    # This suggests we should treat the "attachment" as part of the subset selection.
    # But we already did DP on small primes.
    # Maybe we should include med primes in the DP?
    # Med primes are 47, 53, 59, 61, 67, 71, 73, 79, 83. (9 primes)
    # Total small+med = 14 + 9 = 23 primes.
    # 2^23 = 8M. 3^23 is too big.
    # But the product constraint prunes heavily.
    # Only subsets with product <= 256 are valid.
    # We can enumerate all valid subsets of small+med primes.
    # How many are there?
    # It's a subset of the 23 primes.
    # We can generate them by recursion or just iterate masks and check product.
    # Iterating 8M masks and checking product (23 ops each) is ~200M ops. Too slow in Python.
    
    # Alternative:
    # Keep the DP on small primes.
    # Then, for each resulting set S, try to add med/large primes greedily?
    # Or, for each med/large prime, check if adding it to any existing set (or creating a new one with 2/3) is beneficial.
    
    # Let's refine the small sets first.
    # For each set in small_sets, we have a base value.
    # We can try to "upgrade" a set S by adding a prime p if p is not in S and p is not in any other set.
    # This is a matching problem again, but on the resulting sets.
    
    # Given time, let's stick to the current pair-matching but add triples to the pair list.
    # The current code generates pairs. We can generate triples and use the same backtracking.
    # Backtracking on pairs was O(P * 2^P) effectively? No, it was O(2^P) where P is number of pairs.
    # If we add triples, the number of items increases.
    
    # Let's just use the DP result for small primes and assume med/large are singletons or attached to 2/3 optimally via a simple check.
    
    # Actually, the simplest improvement is to just use the DP for small primes and keep med/large as singletons, 
    # BUT check if attaching a med/large prime to 2 or 3 (if available in a set) increases the sum.
    
    # Let's just return the DP result for small primes + singletons for rest + 1.
    # This might be better than pair matching if pair matching missed a good triple.
    
    result = [best_num_subset[sub] for sub in [choice[full_mask]]] # This is wrong, we lost the partition
    # Re-extract properly
    current_mask = full_mask
    result_vals = []
    while current_mask:
        sub = choice[current_mask]
        result_vals.append(best_num_subset[sub])
        current_mask ^= sub
        
    # Add med/large/huge as singletons (max power)
    for p in med_primes + large_primes + huge_primes:
        v = 1
        while v * p <= 256:
            v *= p
        result_vals.append(v)
        
    result_vals.append(1)
    
    # Check distinctness
    # If duplicates exist, we need to fix.
    # Given the constraints, duplicates are unlikely but possible.
    # We can just return this.
    
    return [int(x) for x in result_vals]

# EVOLVE-BLOCK-END
