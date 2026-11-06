def construct():
    from itertools import combinations
    import random

    # We want the largest subset S of [1,100] such that no subset of S sums to 200.
    # The trivial upper bound: [1,100] itself.
    # We know [66,100] works (size 35).
    # Let's try to find a larger set by greedily adding elements from smaller numbers
    # and checking if the set remains "safe" (no subset sum to 200).
    
    # A set S is safe if there is no subset of S with sum 200.
    # Checking all subsets is exponential. But we can be smarter.
    # If we start with [66,100] and try to add elements from 1..65, we can check
    # if adding an element x creates a subset sum of 200.
    # Actually, it's easier to start with the full set [1,100] and remove elements
    # until the set is safe. But that's also hard.
    
    # Alternative: Greedy addition.
    # Start with an empty set. Add elements from 100 down to 1? Or 1 to 100?
    # If we add large elements first, the subset sums grow faster, so we might
    # detect conflicts earlier? No, we want to MAXIMIZE size.
    
    # Let's try a randomized greedy approach or a specific heuristic.
    # Heuristic: Start with [66,100]. Try to add elements from 1 to 65.
    # For each element e in 1..65, check if S union {e} is safe.
    # How to check if S union {e} is safe?
    # We need to know if there exists a subset T of S such that sum(T) + e = 200,
    # i.e., sum(T) = 200 - e.
    # Since S starts as [66,100], the minimum sum of a non-empty subset is 66.
    # So we only need to check if 200 - e can be formed as a subset sum of S.
    # Note: 200 - e for e in 1..65 ranges from 199 down to 135.
    # The elements in S are all >= 66.
    # Can 135..199 be formed by subsets of [66,100]?
    # Pairs: min 66+67=133, max 99+100=199.
    # Triples: min 66+67+68=201 > 199.
    # So only single elements or pairs matter.
    # Single: Is 200-e in S? Since S subset of [66,100] and 200-e is in [135,199],
    #   it's possible that 200-e is in S. If so, {200-e, e} is a forbidden pair.
    # Pair: Are there a,b in S such that a+b = 200-e?
    #   a,b >= 66. Min sum 132. Max sum 199.
    #   So we need to check if 200-e can be written as a sum of two distinct elements in S.
    
    # Let's refine the "safe" check for S subset of [66,100] plus some small elements.
    # Actually, let's just do a proper subset sum check for the candidate set.
    # The size will be around 35-40. Subset sum check is 2^n which is too big for n=40.
    # But we only care about sum 200.
    # We can use a bitset or a set of reachable sums.
    # Reachable sums up to 200.
    
    def is_safe(S):
        # Returns True if no subset of S sums to 200.
        # S is a list of integers.
        # We track reachable sums <= 200.
        reachable = {0}
        for x in S:
            # new_reachable = {s + x for s in reachable if s + x <= 200}
            # If 200 becomes reachable, return False
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= 200:
                    new_sums.add(ns)
            if 200 in new_sums:
                return False
            reachable.update(new_sums)
        return True

    # Greedy algorithm:
    # Start with the full set [1,100]? No, that's definitely not safe.
    # Start with [66,100] which is safe.
    # Try to add elements from 1 to 65 in some order.
    # Which order? Maybe random? Or descending?
    # Let's try descending: 65, 64, ..., 1.
    # Or ascending: 1, 2, ..., 65.
    # Adding smaller elements first might allow more large elements to stay?
    # Actually, adding any element x requires that 200-x is NOT formable by the current set.
    # Let's try multiple orders and keep the best result.
    
    best_set = list(range(66, 101))
    best_size = len(best_set)
    
    # Try adding elements in different orders
    orders = [
        list(range(65, 0, -1)),  # Descending
        list(range(1, 66)),      # Ascending
        list(range(1, 66, 2)) + list(range(2, 66, 2)), # Even then odd
        list(range(65, 0, -2)) + list(range(64, 0, -2)), # Odd then even
    ]
    
    # Also try random shuffles
    random.seed(42)
    for _ in range(50):
        r = list(range(1, 66))
        random.shuffle(r)
        orders.append(r)
    
    for order in orders:
        S = list(range(66, 101))
        for e in order:
            if e in S:
                continue
            # Try adding e
            S_test = S + [e]
            if is_safe(S_test):
                S = S_test
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    return best_set
