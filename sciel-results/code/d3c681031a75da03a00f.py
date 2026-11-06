"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # We want to maximize cardinality, up to 48.
    # Strategy: Greedy construction from largest to smallest.
    # This tends to work well because large numbers are less likely to be the average 
    # of a subset of other numbers unless those other numbers are also very large and 
    # specifically arranged. By starting with the largest, we "anchor" the set with 
    # numbers that are hard to be means, and then we fill in with smaller numbers 
    # that are checked against the existing large numbers.
    
    # However, a pure greedy might get stuck. 
    # Let's try a randomized greedy with multiple restarts if time permits.
    
    best_set = []
    
    def _is_valid_set(S):
        """
        Check if S is valid.
        S is a list of distinct integers.
        Condition: No a in S is the mean of any subset of S\{a} of size >= 2.
        """
        n = len(S)
        if n < 2:
            return True
            
        # Pre-sort for potential pruning? Not strictly needed for bitsets.
        # For each a in S, we need to check if k*a is achievable by k elements from S\{a}.
        
        # Optimization: 
        # Let T = S \ {a}.
        # We need to check if there exists k >= 2 such that sum(subset of T, size k) == k*a.
        # This is equivalent to: sum(subset of T, size k) - k*a == 0.
        # Let T' = [x - a for x in T]. We need a subset of T' of size k >= 2 that sums to 0.
        
        # Bitset DP for subset sums with cardinality constraint is expensive if done naively for each a.
        # But N is small (<=48).
        
        # Alternative approach for check:
        # For a fixed a, compute all subset sums of T = S\{a} grouped by cardinality.
        # dp[k] is a bitset of sums achievable with k elements.
        
        # To speed this up, we can precompute for the whole set S?
        # No, removing a changes everything.
        
        # Let's just do the DP for each a.
        # Max sum of T is sum(S) - a <= 48*512 = 24576.
        # Bitset size ~ 24576 bits = 3KB.
        # For each a (48 times), we do DP over 47 elements.
        # 48 * 47 * (bitshift of 3KB int) is very fast in Python.
        
        for a in S:
            T = [x for x in S if x != a]
            n_t = len(T)
            
            # dp[k] is an integer bitset where bit s is 1 if sum s is achievable with k elements.
            # We only care about k >= 2.
            # Initialize
            dp = [0] * (n_t + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            # Process elements in T
            for x in T:
                # Update in reverse order of k
                # We only need k up to n_t
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            # Check for any k >= 2 if k*a is in dp[k]
            # Note: k*a must be achievable.
            # Also k*a must be <= sum(T).
            max_sum_t = sum(T)
            for k in range(2, n_t + 1):
                target = k * a
                if target > max_sum_t:
                    break # Since a > 0, k*a increases with k
                if (dp[k] >> target) & 1:
                    return False
                    
        return True

    def _try_construct_random_order(seed):
        random.seed(seed)
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        S = []
        for x in candidates:
            # Check if adding x keeps S valid
            # We need to check validity of S U {x}
            # Optimization: Instead of full check, we can check:
            # 1. Is x mean of subset of S?
            # 2. Is any y in S mean of subset of S U {x} including x?
            
            # For simplicity and robustness, just do full check on S U {x}
            # But full check is O(N^2 * DP). 
            # With N=48, this is 48 * 48 * 48 * bitops. Might be slow if done 512 times.
            # 512 * 48 * 48 * 48 ~ 5.9M operations. Bitops are fast.
            
            new_S = S + [x]
            if _is_valid_set(new_S):
                S.append(x)
                
            if len(S) == 48:
                break
                
            if time.time() > deadline:
                break
                
        return S

    # Try a few random seeds
    # Also try a deterministic greedy (largest to smallest) as a baseline
    # Largest to smallest often yields larger sets because large numbers are "safer"
    
    # Baseline: Largest to smallest
    S_greedy_large = []
    for x in range(512, 0, -1):
        if time.time() > deadline:
            break
        new_S = S_greedy_large + [x]
        if _is_valid_set(new_S):
            S_greedy_large.append(x)
        if len(S_greedy_large) == 48:
            break
            
    best_set = S_greedy_large
    best_size = len(best_set)
    
    # If greedy didn't reach 48, try random restarts
    if best_size < 48:
        # Try a few random orders
        for i in range(10):
            if time.time() > deadline:
                break
            S_rand = _try_construct_random_order(i)
            if len(S_rand) > best_size:
                best_set = S_rand
                best_size = len(S_rand)
            if best_size == 48:
                break
                
    return best_set

# EVOLVE-BLOCK-END
