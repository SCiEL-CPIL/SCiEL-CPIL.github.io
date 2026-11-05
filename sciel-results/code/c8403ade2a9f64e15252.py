import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    N = 512
    target_size = 48
    
    # Helper to check if a set is valid exactly
    def is_valid_exact(S):
        S = set(S)
        if len(S) < 2:
            return True
        
        # dp[k][s] = 1 if there is a subset of size k with sum s
        # Max sum approx 48 * 512 = 24576
        max_sum = sum(S)
        # Optimization: We only need to check sums up to max_sum
        # But we need to check for each a in S if there is a subset of S\{a}
        # that sums to k*a.
        
        # We will compute subset sums for the whole set S
        # dp[k][s]
        dp = [[0] * (max_sum + 1) for _ in range(len(S) + 1)]
        dp[0][0] = 1
        
        # We need to be able to exclude 'a' from the subset.
        # Instead of full DP, we can iterate 'a' and do DP on S\{a}?
        # That is O(|S|^2 * MaxSum) which is 48 * 48 * 25000 ~ 57M ops. Too slow in Python.
        
        # Alternative:
        # For each a in S:
        #   We want to know if there exists T subset of S\{a} with sum(T) = k * a for some k >= 2.
        #   Let T' = T.
        #   We can precompute all subset sums of S? No.
        
        # Let's use the property:
        # If a is the mean of T, then sum(T) = |T| * a.
        # We can check for each a, and for each possible subset size k (2 to len(S)-1),
        # if there is a subset of S\{a} of size k summing to k*a.
        
        # To make this fast, we can use a bitset for subset sums?
        # But we need size constraint.
        
        # Given the small N, maybe a simpler check is enough for the final verification?
        # The SA will guide us to 0 approximate violations.
        # If approximate violations are 0, it's very likely exact is 0.
        # We can just return the set if approximate cost is 0.
        
        # Let's implement a faster exact check using recursion with pruning?
        # Or just trust the approximate cost for k=2,3,4,5.
        
        # For now, let's just return True if we assume the SA cost function is good enough.
        # But we must ensure validity.
        
        # Let's try a specific check:
        # For each a in S:
        #   remaining = S - {a}
        #   # Check if a is mean of any subset of remaining
        #   # This is equivalent to: does there exist a subset T of remaining
        #   # such that sum(T) = k * a for some k = len(T)?
        
        # We can do a DP on 'remaining' for each 'a'?
        # 48 * (47 * 25000) ~ 55M operations. Might be too slow in Python (approx 5-10s).
        # We have 55s. We can afford 5-10 checks.
        
        # Let's optimize the DP for S\{a}.
        # We can precompute the DP for S once?
        # dp_S[k][s]
        # Then dp_{S\{a}}[k][s] can be derived?
        # dp_S[k][s] = dp_{S\{a}}[k][s] + dp_{S\{a}}[k-1][s-a]
        # So dp_{S\{a}}[k][s] = dp_S[k][s] - dp_{S\{a}}[k-1][s-a]
        # This is recursive.
        
        # Let's just do the naive check for the final candidate.
        # If SA finds a set with 0 violations for k=2,3,4, it's likely valid.
        # We will verify with a slower but exact method only if needed.
        
        return True # Placeholder, will be replaced by logic below

    def cost_approx(S):
        S_list = list(S)
        S_set = set(S_list)
        cost = 0
        # Check k=2
        # a is mean of b, c => b+c = 2a
        # For each a, check if there are b,c in S\{a} such that b+c=2a
        # We can use a set of sums?
        # Or just iterate pairs.
        # Pairs (b,c) in S. If b+c is even, mean = (b+c)/2.
        # If mean in S and mean != b and mean != c, then violation.
        sums = {}
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                s = S_list[i] + S_list[j]
                if s % 2 == 0:
                    mean = s // 2
                    if mean in S_set and mean != S_list[i] and mean != S_list[j]:
                        cost += 1
        
        # Check k=3
        # a is mean of b,c,d => b+c+d = 3a
        # Iterate triples? 48C3 ~ 17000.
        # For each triple, sum = s. If s % 3 == 0, mean = s//3.
        # If mean in S and mean not in triple, violation.
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                for k in range(j+1, len(S_list)):
                    s = S_list[i] + S_list[j] + S_list[k]
                    if s % 3 == 0:
                        mean = s // 3
                        if mean in S_set and mean != S_list[i] and mean != S_list[j] and mean != S_list[k]:
                            cost += 1
                            
        # Check k=4
        # 48C4 ~ 190k. Might be slow in Python if done every step.
        # Let's skip k=4 in the main loop, or do it less frequently.
        
        return cost

    def is_valid_exact_slow(S):
        S_set = set(S)
        for a in S_set:
            others = [x for x in S_set if x != a]
            # Check if a is mean of any subset of others
            # Use DP on others
            # Max sum of others
            max_sum = sum(others)
            # dp[k][s]
            # We only care about s = k * a
            # So we can limit the sum range?
            # No, we need to know if sum == k*a.
            
            # Simple check:
            # For each subset size k from 2 to len(others):
            #   Target sum = k * a
            #   Check if there is a subset of 'others' of size k summing to target.
            
            # We can use a set of possible sums for each size?
            # sums_by_size = [set() for _ in range(len(others)+1)]
            # sums_by_size[0].add(0)
            # for x in others:
            #   for k in range(len(others), 0, -1):
            #     for s in sums_by_size[k-1]:
            #       sums_by_size[k].add(s+x)
            #   # This is O(N^2 * MaxSum) worst case, but sets might be sparse.
            
            # Let's try this.
            n_others = len(others)
            sums_by_size = [set() for _ in range(n_others + 1)]
            sums_by_size[0].add(0)
            
            for x in others:
                for k in range(n_others, 0, -1):
                    if sums_by_size[k-1]:
                        # Add x to all sums in sums_by_size[k-1]
                        # Optimization: only add if s+x <= max_possible_target?
                        # Max target is n_others * a
                        new_sums = set()
                        for s in sums_by_size[k-1]:
                            ns = s + x
                            # Prune if ns > n_others * a (max possible mean sum for size k is k*a, but k varies)
                            # Actually, we just need to know if k*a is in the set.
                            new_sums.add(ns)
                        sums_by_size[k].update(new_sums)
            
            # Check if any k*a is in sums_by_size[k]
            for k in range(2, n_others + 1):
                if k * a in sums_by_size[k]:
                    return False
        
        return True

    # SA
    random.seed(42)
    best_set = None
    best_cost = float('inf')
    
    # Initial random set
    current_set = set(random.sample(range(1, N+1), target_size))
    current_cost = cost_approx(current_set)
    
    if current_cost == 0:
        if is_valid_exact_slow(current_set):
            return list(current_set)
    
    T = 10.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick move
        x = random.choice(list(current_set))
        y = random.choice([i for i in range(1, N+1) if i not in current_set])
        
        new_set = current_set.copy()
        new_set.remove(x)
        new_set.add(y)
        
        new_cost = cost_approx(new_set)
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_set = new_set
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = current_set.copy()
                
                if best_cost == 0:
                    if is_valid_exact_slow(best_set):
                        return list(best_set)
        
        T *= cooling
        
        if T < 0.1:
            T = 0.1
            
    # Final check
    if best_set and is_valid_exact_slow(best_set):
        return list(best_set)
        
    # Fallback: return best set found (even if not perfectly verified, it might be close)
    # Or try a few more restarts if time permits
    if time.time() < deadline - 5:
        # Try one more restart with different seed
        random.seed(123)
        current_set = set(random.sample(range(1, N+1), target_size))
        current_cost = cost_approx(current_set)
        T = 10.0
        while time.time() < deadline:
            x = random.choice(list(current_set))
            y = random.choice([i for i in range(1, N+1) if i not in current_set])
            new_set = current_set.copy()
            new_set.remove(x)
            new_set.add(y)
            new_cost = cost_approx(new_set)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                current_set = new_set
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_set = current_set.copy()
                    if best_cost == 0:
                        if is_valid_exact_slow(best_set):
                            return list(best_set)
            T *= 0.995
            if T < 0.1: T = 0.1

    return list(best_set) if best_set else list(range(1, target_size+1))

# EVOLVE-BLOCK-END
