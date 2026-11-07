import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    best_set = []
    best_size = 0
    
    # We want to find a large set.
    # A known good strategy is to use numbers that grow fast to ensure unique sums.
    # Then check divisibility.
    
    # Let's try a randomized backtracking approach.
    # We will try to build the set one element at a time.
    
    def is_valid(sums, x):
        """
        Check if adding x to the set of elements (represented by their subset sums 'sums') is valid.
        Returns True if valid, False otherwise.
        Also returns the new set of sums if valid.
        """
        new_sums = set(sums)
        new_sums.add(x)
        
        # Generate new subset sums
        # For each existing sum s, s+x is a new sum
        # We need to check if s+x is in new_sums (collision) or violates divisibility
        # Also check if x itself violates divisibility with existing sums
        
        # 1. Check x against existing sums
        for s in sums:
            if s % x == 0 or x % s == 0:
                return False, None
        
        # 2. Check new sums (s+x) against existing sums and each other
        # We can do this incrementally
        # Actually, it's safer to generate all new sums first, then check all pairs
        # But that's O(N^2).
        
        # Let's generate the list of new sums
        added_sums = []
        for s in sums:
            ns = s + x
            if ns in new_sums:
                # Collision
                return False, None
            new_sums.add(ns)
            added_sums.append(ns)
        
        # Check divisibility among all sums in new_sums
        # We only need to check new sums against old sums, and new sums against each other
        # Old sums were already valid.
        
        # Check new sums against old sums
        for ns in added_sums:
            for s in sums:
                if ns % s == 0 or s % ns == 0:
                    return False, None
        
        # Check new sums against each other
        for i in range(len(added_sums)):
            a = added_sums[i]
            for j in range(i + 1, len(added_sums)):
                b = added_sums[j]
                if a % b == 0 or b % a == 0:
                    return False, None
                    
        return True, new_sums

    def backtrack(current_set, sums, candidates, time_limit):
        nonlocal best_set, best_size
        
        if time.time() > time_limit:
            return
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            
        if not candidates:
            return
            
        # Try each candidate
        # Sort candidates to try larger ones first? Or smaller?
        # Larger numbers grow sums faster, might lead to fewer collisions?
        # But larger numbers are more likely to be divisible by smaller sums?
        # Let's try random order or descending.
        
        # To avoid deep recursion limits, we can use iterative or limit depth.
        # Max size is 30. Recursion depth 30 is fine.
        
        # Heuristic: Try to add candidates that are "safe"
        # We can pre-filter candidates that are divisible by any existing sum or divide any existing sum
        # This is already done in is_valid, but we can do it earlier to skip recursion.
        
        valid_candidates = []
        for x in candidates:
            # Quick check: if x is divisible by any sum in sums, skip
            # Or if any sum in sums is divisible by x, skip
            skip = False
            for s in sums:
                if s % x == 0 or x % s == 0:
                    skip = True
                    break
            if not skip:
                valid_candidates.append(x)
                
        if not valid_candidates:
            return
            
        # Randomize order to escape local optima
        random.shuffle(valid_candidates)
        
        for x in valid_candidates:
            new_sums = set(sums)
            new_sums.add(x)
            # Check full validity
            valid, final_sums = is_valid(sums, x)
            if valid:
                backtrack(current_set + [x], final_sums, [c for c in candidates if c > x], time_limit)
                # Note: We are iterating candidates in some order. 
                # If we use a set for candidates, we need to be careful with removal.
                # Better to pass a list of remaining candidates.
                # The line above `[c for c in candidates if c > x]` assumes we are picking from a sorted list and moving forward?
                # No, backtracking usually involves choosing any remaining element.
                # Let's restructure to pass a list of remaining candidates and remove x from it.
                
                # Actually, the standard backtracking pattern is:
                # for i in range(len(candidates)):
                #    x = candidates[i]
                #    recurse with candidates without x
                
                # My current loop iterates over valid_candidates which is a subset.
                # I need to pass the remaining candidates excluding x.
                
                # Let's refine the recursion call.
                # This is a bit inefficient if done inside the loop like that.
                # Let's stick to a simpler recursive structure.

    # Let's rewrite the backtrack function to be cleaner.
    
    def solve_backtrack(current_set, sums, remaining, time_limit):
        nonlocal best_set, best_size
        
        if time.time() > time_limit:
            return
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            
        if not remaining:
            return
            
        # Try to add each element from remaining
        # To improve performance, we can try to add elements in a specific order.
        # Random order is good for finding diverse solutions.
        
        # We can also try to prioritize elements that are "large" to keep sums distinct quickly.
        # Or "small" to fit more elements?
        
        # Let's shuffle remaining to randomize
        rem_list = list(remaining)
        random.shuffle(rem_list)
        
        for x in rem_list:
            # Check if x can be added
            valid, new_sums = is_valid(sums, x)
            if valid:
                new_remaining = [y for y in remaining if y != x]
                solve_backtrack(current_set + [x], new_sums, new_remaining, time_limit)
                
                # Early termination if we found a very large set?
                # Max possible is 30. If we reach 20+, we can relax time limit.
                # But let's just let it run.

    # Initial candidates
    candidates = list(range(1, 257))
    
    # We can run multiple restarts with different random seeds/orders
    # Or just one long run with random shuffling.
    
    # Let's try a few different strategies:
    # 1. Descending order (greedy-like)
    # 2. Random order
    
    # Strategy 1: Greedy with backtracking?
    # The backtracking above is pure DFS. It might get stuck.
    # We can add a time check inside the loop to switch strategies or restart.
    
    # Let's run the backtracker for a while.
    # To make it faster, we can prune candidates that are multiples of existing sums.
    
    # Actually, the `is_valid` function is O(N^2) where N is number of sums.
    # Number of sums is 2^k. For k=15, N=32768. N^2 is 10^9. Too slow.
    # We need a faster validity check.
    
    # Optimization:
    # Maintain a list of sums.
    # When adding x, new sums are {s+x for s in sums} U {x}.
    # We need to check if any new sum divides or is divisible by any existing sum.
    # And if any new sum divides or is divisible by another new sum.
    
    # Divisibility check:
    # For a new sum ns, check if any s in sums divides ns or is divisible by ns.
    # This is O(N) per new sum. Total O(N^2).
    # Can we do better?
    # If we sort sums, we can use binary search? No, divisibility is not monotonic in that way.
    
    # However, if we choose elements that are powers of 2, sums are distinct and no sum divides another?
    # Sums of powers of 2 are just integers.
    # Example: 1, 2. Sums: 1, 2, 3.
    # 1 divides 2 and 3. Invalid.
    # So powers of 2 don't work directly because 1 divides everything.
    
    # We need to avoid 1? Or avoid sums that are divisors of others.
    # If we start with large numbers, e.g., 100, 200.
    # Sums: 100, 200, 300.
    # 100 divides 200. Invalid.
    
    # It seems very hard to have large sets.
    # Let's check the score 2.3333. That means size 7? (7/3 = 2.33).
    # So the seed program probably found a set of size 7.
    
    # Let's try to find a set of size > 7.
    
    # A known construction for "sum-free" or similar is using primes?
    # If all elements are primes > max_sum/2, then subset sums are sums of distinct primes.
    # But sums can still divide each other.
    
    # Let's rely on the backtracking with a fast check.
    # We can speed up the check by pre-computing divisors of numbers up to max_sum (256*30 = 7680).
    # Precompute a list of divisors for each number up to 7680.
    # Then check if any divisor of ns is in sums.
    
    MAX_SUM = 256 * 30
    divisors = [[] for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1, i):
            divisors[j].append(i)
            
    def is_valid_fast(sums_set, sums_list, x):
        """
        sums_set: set of current sums for O(1) lookup
        sums_list: list of current sums for iteration
        x: new element
        """
        new_sums = [x]
        for s in sums_list:
            ns = s + x
            new_sums.append(ns)
            
        # Check collisions
        for ns in new_sums:
            if ns in sums_set:
                return False, None, None
            # Check if ns is divisible by any existing sum or vice versa
            # We can check divisors of ns
            for d in divisors[ns]:
                if d in sums_set and d != ns:
                    return False, None, None
            # Check if ns divides any existing sum
            # This is harder. We can check multiples of ns.
            # But we can also just iterate sums_list and check if s % ns == 0
            # To avoid O(N^2), we can note that if ns is small, it has many multiples.
            # If ns is large, it has few multiples.
            
        # Check new sums against each other
        # Sort new_sums?
        # For each pair (a, b) in new_sums, check divisibility.
        # This is O(K^2) where K is number of new sums (which is N+1).
        # Still O(N^2).
        
        # Let's combine the checks.
        # 1. Build new_sums list.
        # 2. Check collisions with old sums.
        # 3. Check divisibility with old sums.
        # 4. Check divisibility among new sums.
        
        # Optimization for step 3:
        # For each ns in new_sums:
        #   Check if any s in sums_list is a multiple of ns.
        #   We can iterate sums_list.
        
        # Let's try this.
        
        new_sums_set = set(new_sums)
        # Check collisions
        if new_sums_set.intersection(sums_set):
            return False, None, None
            
        # Check divisibility with old sums
        for ns in new_sums:
            # Check if any old sum divides ns
            for d in divisors[ns]:
                if d in sums_set and d != ns:
                    return False, None, None
            # Check if ns divides any old sum
            # Iterate old sums
            for s in sums_list:
                if s % ns == 0 and s != ns:
                    return False, None, None
                    
        # Check divisibility among new sums
        # We can sort new_sums to break early?
        ns_list = sorted(new_sums)
        for i in range(len(ns_list)):
            a = ns_list[i]
            for j in range(i + 1, len(ns_list)):
                b = ns_list[j]
                if b % a == 0:
                    return False, None, None
                    # break inner loop? Yes, if one pair fails, it's invalid.
                    
        # If we reach here, it's valid
        final_sums_set = sums_set.union(new_sums_set)
        final_sums_list = sums_list + new_sums
        return True, final_sums_set, final_sums_list

    # Restart backtracking with fast check
    def solve_fast(current_set, sums_set, sums_list, remaining, time_limit):
        nonlocal best_set, best_size
        
        if time.time() > time_limit:
            return
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            
        if not remaining:
            return
            
        rem_list = list(remaining)
        random.shuffle(rem_list)
        
        for x in rem_list:
            valid, new_sums_set, new_sums_list = is_valid_fast(sums_set, sums_list, x)
            if valid:
                new_remaining = [y for y in remaining if y != x]
                solve_fast(current_set + [x], new_sums_set, new_sums_list, new_remaining, time_limit)

    # Run multiple restarts
    candidates = list(range(1, 257))
    
    # Try a few different initial orders or just random
    for i in range(20):
        if time.time() > deadline:
            break
            
        # Shuffle candidates for random restart
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        # Initial state
        init_sums_set = set()
        init_sums_list = []
        
        solve_fast([], init_sums_set, init_sums_list, shuffled_candidates, deadline)
        
        # If we found a very large set, we can stop or tighten time
        if best_size > 15:
            break

    return best_set

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res), res)
