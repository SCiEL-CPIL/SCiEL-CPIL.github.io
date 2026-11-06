import time
import random
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 82.0  # Safety margin under 90s kill
    n = 512
    target = 48

    # Precompute conflict information:
    # For each pair (i, j) with i != j, if we include both, we need to check
    # that no other element is the mean of a subset containing both.
    # But the constraint is: no element x in S is the mean of any subset T of S\{x}, |T|>=2.
    # This means: for all x in S, for all subsets T of S\{x} with |T|>=2, sum(T) != |T|*x.

    # A SAT-like approach: we model this as a maximum independent set problem in a conflict graph.
    # Two elements "conflict" if they cannot both be in a valid set. But the constraint is higher-order.
    # Actually, the constraint is: x is NOT the mean of any subset of OTHER elements.
    # So x conflicts with a set T if x = mean(T) and x not in T.

    # Let's use a different approach: local search / simulated annealing style.
    # Start with a random set, then try to add elements or swap elements to improve.

    # Actually, let's use a smarter greedy with backtracking.
    # The key insight: the constraint is that for each x in S, and for each k>=2,
    # there should be no subset of S\{x} of size k that sums to k*x.

    # Let's precompute for each x, which subsets of other elements sum to k*x.
    # But that's expensive. Instead, let's use a greedy approach with a smart ordering.

    # Strategy: 
    # 1. Start with an empty set.
    # 2. For each candidate x (in a smart order), check if adding x keeps the set valid.
    # 3. Use a fast check: for each existing y, check if x creates a violation for y or for x.

    # The violation conditions when adding x to set C:
    # A) x is the mean of some subset T of C, |T|>=2: sum(T) = |T|*x
    # B) Some y in C becomes the mean of a subset T' of C\{y} ∪ {x} containing x:
    #    i.e., there exists a subset U of C\{y} such that sum(U) + x = (|U|+1)*y
    #    i.e., sum(U) = (|U|+1)*y - x

    # Let's implement a fast check using bitsets for subset sums.

    def can_add(C, x):
        """Check if x can be added to C without violating constraints."""
        m = len(C)
        if m == 0:
            return True
        
        # Check A: x is mean of subset of C
        # For each k from 2 to m, check if there's a k-subset of C summing to k*x
        # Use DP for subset sums of C
        # But we need to do this efficiently. Let's precompute DP for C.
        
        # Actually, let's maintain DP incrementally. But for simplicity in this rewrite,
        # let's just do a direct check with a limited DP.
        
        # For check A: subset sums of C
        # max sum of C is m * n
        # We need to check if k*x is achievable for some k in [2, m]
        
        # Let's compute all subset sums of C using bitset DP
        # dp[s] = bitmask of subset sizes that can achieve sum s
        # But sums can be up to m*n = 48*512 = 24576
        # That's manageable with Python integers as bitsets.
        
        # Actually, let's use a different representation:
        # dp[k] = bitset where bit s is 1 if there's a k-subset of C summing to s
        
        dp = [0] * (m + 1)
        dp[0] = 1  # empty subset, sum 0
        
        for val in C:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        
        # Check A
        for k in range(2, m + 1):
            target_sum = k * x
            if target_sum > m * n:
                continue
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check B: for each y in C, check if y becomes mean of subset containing x
        # For each y in C, we need: exists subset U of C\{y} such that sum(U) = (|U|+1)*y - x
        # Let j = |U|, then sum(U) = (j+1)*y - x, j in [0, m-1]
        # But we need |U|+1 >= 2, so j >= 1.
        
        for y in C:
            # Build DP for C\{y}
            # We can reuse dp by "removing" y, but that's complex. Let's rebuild.
            # C\{y} has m-1 elements.
            dp2 = [0] * (m)  # indices 0..m-1
            dp2[0] = 1
            
            for val in C:
                if val == y:
                    continue
                for k in range(m - 1, 0, -1):
                    dp2[k] |= (dp2[k - 1] << val)
            
            for j in range(1, m):  # j = |U|, U subset of C\{y}
                target_sum = (j + 1) * y - x
                if target_sum < 0:
                    continue
                if target_sum > (m - 1) * n:
                    continue
                if (dp2[j] >> target_sum) & 1:
                    return False
        
        return True

    def is_valid(S):
        """Check if S is a valid set."""
        m = len(S)
        if m <= 1:
            return True
        
        for x in S:
            C = [v for v in S if v != x]
            if not can_add(C, x):
                return False
        return True

    # Greedy with multiple random restarts
    best = []
    rng = random.Random(42)
    
    for trial in range(500):
        if time.time() > deadline - 5.0:
            break
        if len(best) >= target:
            break
        
        # Random order
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        C = []
        for x in order:
            if len(C) >= target:
                break
            if can_add(C, x):
                C.append(x)
        
        if len(C) > len(best):
            best = list(C)
            if len(best) >= target:
                break
    
    # Try to improve best by adding elements
    if len(best) < target:
        # Try adding elements not in best
        remaining = [x for x in range(1, n + 1) if x not in best]
        rng.shuffle(remaining)
        for x in remaining:
            if len(best) >= target:
                break
            if time.time() > deadline - 3.0:
                break
            if can_add(best, x):
                best.append(x)
    
    # Final validation and trimming
    while best and not is_valid(best):
        # Remove the most problematic element
        worst = None
        worst_count = 0
        for x in best:
            C = [v for v in best if v != x]
            # Count how many subsets of C make x the mean
            count = 0
            m = len(C)
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in C:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k - 1] << val)
            for k in range(2, m + 1):
                target_sum = k * x
                if target_sum <= m * n and (dp[k] >> target_sum) & 1:
                    count += 1
            if count > worst_count:
                worst_count = count
                worst = x
        if worst is not None:
            best.remove(worst)
        else:
            break
    
    return best
