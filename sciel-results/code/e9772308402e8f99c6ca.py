"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Strategy:
    # 1. Generate a random candidate set of size M (e.g., 60-80) from [1, 512].
    # 2. Iteratively remove elements that violate the "mean of subset" condition.
    # 3. The violation check for a specific element 'a' in set 'S' can be optimized.
    #    a is the mean of a subset T of S\{a} (|T|>=2) iff sum(T) = k*a.
    #    This is equivalent to: there exists a subset U of S\{a} such that sum(U) = k*a.
    #    Instead of checking all subsets for every element from scratch, we can:
    #    - Precompute subset sums? No, N is too large for full bitset DP on the whole set if we do it repeatedly.
    #    - However, we can check validity of the *current* set S.
    #    - If invalid, identify one violating pair (a, T) and remove 'a'.
    #    - Repeat until valid.
    
    # To make this fast, we need an efficient _is_valid_set and _find_violation.
    # The current seed uses bitset DP for each element. This is O(N^2 * bitops).
    # With N=80, N^2=6400. Bitops on ~24k bits are fast.
    # 6400 * 80 (elements) = 512k bit operations per check.
    # If we do 100 checks, that's 50M ops. Python bit ops are C-speed, so this is feasible.
    
    def _check_violation(S):
        """
        Returns (element, subset) if violation exists, else (None, None).
        S is a list of distinct ints.
        """
        n = len(S)
        if n < 3:
            return None, None
        
        # Sort S to potentially help with pruning? Not strictly needed for bitset.
        # For each a in S:
        #   T = S \ {a}
        #   Check if k*a is a subset sum of T for any k >= 2.
        
        # Optimization: 
        # If we sort S, and a is small, k*a is small.
        # If a is large, k*a is large.
        
        # Let's just iterate.
        for i, a in enumerate(S):
            T = S[:i] + S[i+1:]
            n_t = len(T)
            
            # DP for subset sums of T
            # Max sum is sum(T).
            # We only care about sums that are multiples of a.
            # dp[k] is bitset of sums achievable with k elements.
            
            # To save memory/time, we can limit the bitset size to max_sum.
            # But Python ints are arbitrary precision, so shifting is fast.
            
            dp = [0] * (n_t + 1)
            dp[0] = 1
            
            for x in T:
                for k in range(n_t, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
                        
            # Check if any k*a is in dp[k] for k >= 2
            # k*a <= sum(T)
            max_sum_t = sum(T)
            for k in range(2, n_t + 1):
                target = k * a
                if target > max_sum_t:
                    break
                if (dp[k] >> target) & 1:
                    # Found violation: a is mean of some subset of size k
                    # We don't need to reconstruct the subset, just remove a
                    return a, None
                    
        return None, None

    def _remove_until_valid(S):
        """
        Iteratively removes violating elements until S is valid.
        """
        S = list(S)
        while True:
            a, _ = _check_violation(S)
            if a is None:
                return S
            S.remove(a)
            if len(S) < 2:
                return S

    best_set = []
    
    # Parameters for random search
    # Start with larger candidate sets to have more "room" to remove
    # and hopefully keep more than a greedy approach.
    
    num_attempts = 200
    for i in range(num_attempts):
        if time.time() > deadline:
            break
            
        # Random size between 50 and 80
        # Smaller sets are faster to check, larger sets might yield larger valid cores.
        # Let's try a mix.
        candidate_size = random.randint(55, 75)
        
        candidates = random.sample(range(1, 513), candidate_size)
        
        valid_set = _remove_until_valid(candidates)
        
        if len(valid_set) > len(best_set):
            best_set = valid_set
            # If we hit 48, we can stop early?
            # The task says "Maximize cardinality. At most 48".
            # If we hit 48, that's the max possible.
            if len(best_set) == 48:
                break
                
    # Fallback: If random search didn't do well, try a simple greedy as backup
    # to ensure we return something decent.
    if len(best_set) < 40:
        greedy_set = []
        # Try largest to smallest
        for x in range(512, 0, -1):
            if time.time() > deadline:
                break
            new_set = greedy_set + [x]
            a, _ = _check_violation(new_set)
            if a is None:
                greedy_set.append(x)
            if len(greedy_set) == 48:
                break
        if len(greedy_set) > len(best_set):
            best_set = greedy_set

    return best_set

# EVOLVE-BLOCK-END
