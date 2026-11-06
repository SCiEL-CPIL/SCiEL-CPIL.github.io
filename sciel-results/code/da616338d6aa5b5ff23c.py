import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 90s limit
    
    # Candidates 2..256
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Precompute incompatibility bitsets
    # adj[i] is a bitset where bit j is 1 if candidates[i] and candidates[j] are incompatible
    # Incompatible means lcm(candidates[i], candidates[j]) <= 256
    adj = [0] * n
    
    for i in range(n):
        a = candidates[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = candidates[j]
            # Check lcm(a, b) <= 256
            # lcm(a,b) = a*b // gcd(a,b)
            # To avoid overflow or precision issues, we can check:
            # a // gcd(a,b) * b <= 256
            g = gcd(a, b)
            if (a // g) * b <= 256:
                mask |= (1 << j)
        adj[i] = mask
        
    # Weights are 1/x. For B&B upper bound, we use sum of weights.
    # To make B&B fast, we sort candidates by weight descending (value ascending)
    # This allows the upper bound to drop faster as we process high-value items first?
    # Actually, sorting by weight descending is standard for knapsack-like B&B.
    # Let's reorder candidates and adj accordingly.
    
    # Sort indices by value ascending (1/x descending)
    sorted_indices = sorted(range(n), key=lambda i: candidates[i])
    sorted_candidates = [candidates[i] for i in sorted_indices]
    sorted_adj = [0] * n
    
    # Rebuild adj for sorted candidates
    # Map old index to new index
    old_to_new = [0] * n
    for new_idx, old_idx in enumerate(sorted_indices):
        old_to_new[old_idx] = new_idx
        
    for new_i in range(n):
        old_i = sorted_indices[new_i]
        old_mask = adj[old_i]
        new_mask = 0
        # Convert old mask to new mask
        # Iterate through bits set in old_mask
        m = old_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            old_j = lsb.bit_length() - 1
            new_j = old_to_new[old_j]
            new_mask |= (1 << new_j)
            m ^= lsb
        sorted_adj[new_i] = new_mask
        
    candidates = sorted_candidates
    adj = sorted_adj
    
    # Precompute suffix sums of weights for upper bound
    # weight[i] = 1 / candidates[i]
    # We use Fraction for exactness in final result, but float for fast B&B bound?
    # The task requires "exact reciprocal sum". 
    # However, using Fraction in B&B is very slow.
    # We can use float for the B&B search to find the optimal set,
    # then compute the exact Fraction sum of that set at the end.
    # Since the set is unique or we just need one optimal set, float precision
    # (double) is sufficient to distinguish sums of ~256 fractions with denominators <= 256.
    # Max sum is < 5. Min difference between distinct sums is > 1/(256*256*256) ~ 6e-8.
    # Double precision has ~15 digits, so it should be safe.
    
    weights = [1.0 / candidates[i] for i in range(n)]
    suffix_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
    best_sum = 0.0
    best_set_mask = 0
    best_set_list = []
    
    # Bitset B&B
    # State: idx (current index to consider), available_mask (bitset of candidates available from idx onwards that are compatible with current set)
    # Actually, standard MWIS B&B:
    # We process vertices one by one.
    # At step i, we have a set of "available" vertices from i..n-1 that are compatible with the current solution.
    # We branch: Include i or Exclude i.
    
    # To optimize, we can use a recursive function with memoization? No, state space too large.
    # Just B&B.
    
    # Optimization: If the sum of weights of all available vertices + current_sum <= best_sum, prune.
    # We need to quickly compute sum of weights of available vertices.
    # Precomputing this for all masks is impossible (2^255).
    # Instead, we iterate through available vertices to sum? That's O(N) per node.
    # With N=255, and depth 255, this might be slow if tree is large.
    # But the tree is pruned heavily.
    
    # Alternative bound: Sum of weights of vertices in available_mask.
    # We can precompute weight sums for chunks?
    # Or just iterate. Let's try iterating first.
    
    def sum_weights(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s
        
    # This O(k) where k is number of bits.
    
    # Recursive B&B
    # To avoid recursion limit and overhead, we can use a stack, but recursion is fine for depth 255.
    # sys.setrecursionlimit(10000)
    
    def bnb(idx, current_sum, current_mask, available_mask):
        nonlocal best_sum, best_set_mask, best_set_list
        
        if time.time() > deadline:
            return
            
        # Prune: if current_sum + sum(weights in available_mask) <= best_sum
        # Note: available_mask only contains vertices from idx onwards?
        # Let's define available_mask as the set of vertices from [idx, n) that are compatible with current_mask.
        
        # Calculate upper bound
        # Optimization: if available_mask is 0, return
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set_mask = current_mask
                # Reconstruct list later
            return
            
        # Quick bound check
        # We can approximate bound by suffix_sums[idx] if all remaining were available.
        # But available_mask is a subset.
        # sum_weights(available_mask) is tighter.
        
        # Heuristic bound: if current_sum + suffix_sums[idx] <= best_sum, prune.
        # This is faster than sum_weights.
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # If we are at the end
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set_mask = current_mask
            return
            
        # Check if idx is in available_mask
        bit = 1 << idx
        if available_mask & bit:
            # Option 1: Include idx
            new_sum = current_sum + weights[idx]
            new_mask = current_mask | bit
            
            # Update available_mask for next step
            # New available = (available_mask & ~bit) & ~adj[idx]
            # adj[idx] contains all vertices incompatible with idx.
            # We only care about vertices >= idx+1.
            # The available_mask already only has bits >= idx.
            # After removing idx, we have bits > idx.
            # We must remove any bits in adj[idx].
            new_available = (available_mask & ~bit) & ~adj[idx]
            
            bnb(idx + 1, new_sum, new_mask, new_available)
            
            # Option 2: Exclude idx
            # available_mask stays the same (minus idx? No, idx is just skipped)
            # The next step considers idx+1. The available set for idx+1 is the same available_mask,
            # but we don't include idx.
            # Wait, the definition of available_mask is "vertices from [idx, n) compatible with current_mask".
            # If we exclude idx, the set of vertices from [idx+1, n) compatible with current_mask is
            # simply available_mask & ~(1 << idx).
            bnb(idx + 1, current_sum, current_mask, available_mask & ~bit)
        else:
            # idx is not available (incompatible with current set)
            # Must exclude
            bnb(idx + 1, current_sum, current_mask, available_mask & ~bit)

    # Initial call
    # All vertices are available initially
    initial_available = (1 << n) - 1
    bnb(0, 0.0, 0, initial_available)
    
    # Reconstruct best set
    if best_set_mask == 0:
        # Should not happen, but fallback
        return [2]
        
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(candidates[idx])
        m ^= lsb
        
    # The result is a list of integers.
    # Verify validity (sanity check)
    # for i in range(len(result)):
    #    for j in range(i+1, len(result)):
    #       if gcd(result[i], result[j]) * (result[i]//gcd(result[i], result[j])) <= 256:
    #          print("Invalid!")
            
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    print(f"Sum: {sum(1/x for x in res)}")
    # Print first few
    print(res[:20])
