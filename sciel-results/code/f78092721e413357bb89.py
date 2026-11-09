import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    n = 192
    max_size = 48
    
    # Precompute triple sums for all possible subsets? No, too many.
    # Instead, we will use a fast validity check.
    
    def is_valid(S):
        """
        Check if S is valid. S is a set of integers.
        Returns True if valid, False otherwise.
        """
        S_list = sorted(S)
        k = len(S_list)
        if k < 4:
            return True
        
        S_set = set(S_list)
        
        for a in S_list:
            # Find b, c, d in S\{a} such that a | (b+c+d)
            # b < c < d
            # Iterate b, c
            # d = m*a - b - c
            # Check if d in S, d > c, d != a
            
            # Optimization:
            # Min possible sum for b,c,d (distinct, != a)
            # Max possible sum
            
            # Let's just iterate pairs (b, c) from S\{a}
            others = [x for x in S_list if x != a]
            if len(others) < 3:
                continue
            
            # For each pair (b, c) with b < c
            # We need d = m*a - b - c
            # d must be in S, d > c, d != a
            # m*a = b + c + d
            # So m*a > b + c + c (since d > c) => m*a > b + 2c
            # And m*a < b + c + max(S)
            
            # Let's just loop m.
            # Range of m:
            # min_sum = others[0] + others[1] + others[2]
            # max_sum = others[-1] + others[-2] + others[-3]
            
            min_sum = others[0] + others[1] + others[2]
            max_sum = others[-1] + others[-2] + others[-3]
            
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            if start_m > end_m:
                continue
            
            # For each m, target = m*a
            # We need b+c+d = target
            # d = target - b - c
            # d > c, d in S, d != a
            
            # To speed up, we can precompute pair sums?
            # But we need to check d in S.
            
            # Let's iterate b, c
            # To avoid O(N^3), we can do:
            # For each b in others:
            #   For each c in others, c > b:
            #     d = target - b - c
            #     if d > c and d in S_set and d != a:
            #        return False
            
            # This is O(N^2 * M).
            # N=48. N^2 ~ 2300. M ~ 10. 23000 ops per a.
            # 48 * 23000 ~ 1M ops.
            # This might be too slow if called frequently.
            
            # Alternative:
            # Precompute all pair sums of others?
            # pairs = [(b, c, b+c) for b, c in combinations(others, 2)]
            # Then for each m, target = m*a
            # We need b+c = target - d.
            # So target - d must be a pair sum.
            # And d > c (where c is the larger of the pair).
            # And d in S, d != a.
            
            # Let's build a dictionary: pair_sum -> list of (b, c)
            # But building this for each a is O(N^2).
            
            # Let's try the direct loop first, it's simpler.
            
            for m in range(start_m, end_m + 1):
                target = m * a
                # Iterate b
                for i in range(len(others)):
                    b = others[i]
                    # Iterate c
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d == a:
                            continue
                        if d in S_set:
                            return False
                            
        return True

    def count_violations(S):
        """
        Count the number of violations in S.
        A violation is a tuple (a, b, c, d) with a | (b+c+d).
        """
        S_list = sorted(S)
        k = len(S_list)
        if k < 4:
            return 0, []
        
        S_set = set(S_list)
        violations = []
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            if len(others) < 3:
                continue
            
            min_sum = others[0] + others[1] + others[2]
            max_sum = others[-1] + others[-2] + others[-3]
            
            start_m = (min_sum + a - 1) // a
            end_m = max_sum // a
            
            if start_m > end_m:
                continue
            
            for m in range(start_m, end_m + 1):
                target = m * a
                for i in range(len(others)):
                    b = others[i]
                    for j in range(i + 1, len(others)):
                        c = others[j]
                        d = target - b - c
                        if d <= c:
                            continue
                        if d == a:
                            continue
                        if d in S_set:
                            violations.append((a, b, c, d))
                            
        return len(violations), violations

    def get_violation_info(S):
        """
        Return list of violations.
        """
        _, violations = count_violations(S)
        return violations

    # Local Search
    def local_search(initial_S, max_iters=10000):
        current_S = set(initial_S)
        current_violations, _ = count_violations(current_S)
        best_S = set(current_S)
        best_violations = current_violations
        
        if current_violations == 0:
            return current_S
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random violation
            if current_violations == 0:
                break
                
            viol = random.choice(get_violation_info(current_S))
            a, b, c, d = viol
            
            # Try removing one of a, b, c, d
            candidates = [a, b, c, d]
            random.shuffle(candidates)
            
            improved = False
            for x in candidates:
                new_S = set(current_S)
                new_S.remove(x)
                new_violations, _ = count_violations(new_S)
                
                if new_violations < current_violations:
                    current_S = new_S
                    current_violations = new_violations
                    if current_violations < best_violations:
                        best_S = set(current_S)
                        best_violations = current_violations
                    improved = True
                    break
                    
            if not improved:
                # Try adding a random element?
                # Or swapping?
                # For now, just break if stuck.
                break
                
        return best_S

    # Greedy Construction
    def greedy_construct():
        S = set()
        # Start with large numbers
        candidates = list(range(n, 0, -1))
        
        for x in candidates:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 10:
                break
                
            # Check if x can be added
            new_S = S | {x}
            if is_valid(new_S):
                S.add(x)
                
        return S

    # Main Logic
    best_S = set()
    
    # 1. Greedy Start
    S = greedy_construct()
    if len(S) > len(best_S):
        best_S = set(S)
        
    if len(best_S) == max_size:
        return list(best_S)
        
    # 2. Local Search to repair/expand
    # We might have a valid set smaller than max_size.
    # We can try to add elements.
    
    # Let's try to add elements one by one to the greedy result.
    current_S = set(S)
    
    # If current_S is not valid (should be, but just in case), repair it.
    while time.time() < deadline - 10:
        if is_valid(current_S):
            if len(current_S) > len(best_S):
                best_S = set(current_S)
                
            if len(current_S) >= max_size:
                break
                
            # Try to add a new element
            added = False
            candidates = [x for x in range(n, 0, -1) if x not in current_S]
            random.shuffle(candidates) # Add some randomness
            
            for x in candidates:
                if time.time() > deadline - 10:
                    break
                new_S = current_S | {x}
                if is_valid(new_S):
                    current_S = new_S
                    added = True
                    break
                    
            if not added:
                # Stuck. Try local search on a slightly larger invalid set?
                # Or just break.
                # Let's try a local search phase to see if we can find a larger valid set.
                break
        else:
            # Repair
            current_S = local_search(current_S)
            if len(current_S) > len(best_S):
                best_S = set(current_S)
                
    # 3. Multiple Restarts with Randomness
    # If we didn't reach max_size, try a few random restarts.
    restarts = 5
    for i in range(restarts):
        if time.time() > deadline - 5:
            break
            
        # Random subset of size max_size?
        # Or random greedy?
        # Let's try a random greedy.
        S_rand = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if len(S_rand) >= max_size:
                break
            if time.time() > deadline - 5:
                break
            new_S = S_rand | {x}
            if is_valid(new_S):
                S_rand.add(x)
                
        if len(S_rand) > len(best_S):
            best_S = set(S_rand)
            
        if len(best_S) == max_size:
            break

    # Final Check
    if is_valid(best_S):
        return list(best_S)
    else:
        # Fallback to a known valid set (e.g. just large numbers)
        # The greedy one should be valid.
        if is_valid(S):
            return list(S)
        return list(range(192 - 47, 193))

