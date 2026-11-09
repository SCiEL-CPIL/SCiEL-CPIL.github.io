"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85
    best_set = []
    
    # Precompute powers of 2 for bitset operations? 
    # Actually, Python integers are arbitrary precision, so 1 << x is fast.
    # We will use bitset DP for subset sum checks.
    
    def can_form_sum(elements, target, k):
        """
        Check if there is a subset of exactly k elements from 'elements' 
        that sums to 'target'.
        Uses bitset DP.
        elements: sorted list of ints
        target: int
        k: int
        """
        if k == 0:
            return target == 0
        if k > len(elements):
            return False
        if target < 0:
            return False
        
        # Optimizations:
        # Min sum of k smallest
        if k <= len(elements):
            min_sum = sum(elements[:k])
            if target < min_sum:
                return False
            max_sum = sum(elements[-k:])
            if target > max_sum:
                return False
        else:
            return False
            
        # Bitset DP
        # dp[j] is a bitset where bit s is 1 if sum s is achievable with j elements
        dp = [0] * (k + 1)
        dp[0] = 1
        mask = (1 << (target + 1)) - 1
        
        for x in elements:
            if x > target:
                continue
            # Iterate downwards to avoid reusing same element
            # We only care up to k
            for j in range(min(k, len(elements)), 0, -1):
                dp[j] |= (dp[j-1] << x) & mask
                # Early exit? If target bit is set, we could return True immediately?
                # Yes, if dp[k] has the target bit set, we are done.
                if (dp[k] >> target) & 1:
                    return True
        return (dp[k] >> target) & 1

    def is_valid_set(S):
        """
        Check if S satisfies the condition: 
        No element in S is the mean of any subset of at least 2 OTHER elements in S.
        S: sorted list
        """
        n = len(S)
        if n < 3:
            return True
            
        for i, x in enumerate(S):
            # Check if x is mean of subset of S \ {x}
            others = S[:i] + S[i+1:]
            # We need subset U of others, size k >= 2, sum(U) = k * x
            # Max size of others is n-1
            for k in range(2, n):
                target = k * x
                if can_form_sum(others, target, k):
                    return False
        return True

    def greedy_construct(order, max_size=48):
        """
        Greedily construct a valid set by iterating through 'order' and adding 
        if valid.
        """
        S = []
        S_set = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if len(S) >= max_size:
                break
                
            # Check if x can be added to S
            # Condition 1: x is not mean of subset of S (size >= 2)
            # Condition 2: For each a in S, a is not mean of subset of S U {x} containing x
            
            valid = True
            
            # Part 1: x vs S
            n_s = len(S)
            for k in range(2, n_s + 1):
                target = k * x
                if can_form_sum(S, target, k):
                    valid = False
                    break
            
            if valid:
                # Part 2: a in S vs S U {x}
                # a is mean of U U {x}, where U subset of S \ {a}, size u >= 1
                # k_total = u + 1
                # sum(U) + x = k_total * a
                # sum(U) = k_total * a - x
                # k_total >= 2 => u >= 1
                # Also U must be distinct from a (which it is, since U subset of S\{a})
                # And U must be subset of S.
                
                for a in S:
                    others_no_a = [y for y in S if y != a]
                    m = len(others_no_a)
                    # u can range from 1 to m
                    for u in range(1, m + 1):
                        target_sum = (u + 1) * a - x
                        if target_sum < 0:
                            continue
                        # Check if there is a subset of others_no_a of size u summing to target_sum
                        if can_form_sum(others_no_a, target_sum, u):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                S.append(x)
                S_set.add(x)
                S.sort() # Keep sorted for can_form_sum efficiency (min/max checks)
                
        return S

    # Strategy: Multiple randomized greedy starts
    # 1. Try standard ascending order first (fast, might work)
    # 2. Try descending order
    # 3. Try random shuffles
    
    # Ascending
    order_asc = list(range(1, 513))
    res_asc = greedy_construct(order_asc)
    if len(res_asc) > len(best_set):
        best_set = res_asc
        if len(best_set) == 48:
            return best_set
            
    # Descending
    order_desc = list(range(512, 0, -1))
    res_desc = greedy_construct(order_desc)
    if len(res_desc) > len(best_set):
        best_set = res_desc
        if len(best_set) == 48:
            return best_set

    # Randomized starts
    # Use a fixed seed for reproducibility in evaluation? 
    # The task says "Maximize cardinality", usually deterministic is preferred unless specified.
    # But randomization helps escape local minima.
    # Let's run a few random permutations if time permits.
    
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        order = list(range(1, 513))
        random.shuffle(order)
        res = greedy_construct(order)
        if len(res) > len(best_set):
            best_set = res
            if len(best_set) == 48:
                break

    return best_set

# EVOLVE-BLOCK-END
