import time
import random
import sys

def construct():
    deadline = time.time() + 140
    prescribed = [1, 2, 5, 11]
    N = 284
    target_size = 20
    sys.setrecursionlimit(100000)
    
    # Initial state
    initial_set = set(prescribed)
    initial_diffs = set()
    for i in range(len(prescribed)):
        for j in range(i + 1, len(prescribed)):
            initial_diffs.add(abs(prescribed[i] - prescribed[j]))
            
    def is_valid_sidon(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def solve_csp(target_k, time_budget):
        start_time = time.time()
        
        # Use a list for the set to maintain order, but a set for O(1) lookup
        # We track: current_elements (list), used_values (set), used_diffs (set)
        
        def backtrack(elements, used_vals, used_diffs):
            if time.time() - start_time > time_budget:
                return None
            
            if len(elements) == target_k:
                return list(elements)
            
            # Generate candidates
            # Optimization: Instead of iterating 1..N, we can be smarter, but N=284 is small.
            candidates = []
            for x in range(1, N + 1):
                if x in used_vals:
                    continue
                
                # Check differences
                valid = True
                # We need to ensure no new difference conflicts with existing ones
                # And no two new differences are equal to each other? 
                # If x1 - y1 = x2 - y2, that's a conflict.
                # But here we are adding ONE x at a time.
                # The new differences are |x - y| for all y in elements.
                # We must ensure:
                # 1. |x - y| not in used_diffs for any y
                # 2. |x - y1| != |x - y2| for any y1, y2 in elements.
                # Condition 2 is automatically true if elements are distinct and x is not in elements?
                # |x - y1| = |x - y2| implies y1 = y2 or y1 + y2 = 2x.
                # If y1 + y2 = 2x, then x is the midpoint.
                # In a Sidon set, a+b = c+d implies {a,b}={c,d}.
                # If x - y1 = y2 - x => 2x = y1 + y2. This is a difference conflict?
                # No, the differences are x-y1 and y2-x.
                # The set of differences is {d | d = |a-b|}.
                # If we add x, new diffs are |x-y| for y in elements.
                # If |x-y1| = |x-y2|, then we have a duplicate difference in the NEW set.
                # So we must check that all |x-y| are distinct.
                
                new_diffs = []
                for y in elements:
                    d = abs(x - y)
                    if d in used_diffs:
                        valid = False
                        break
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return None
            
            # Heuristic: Sort candidates
            # A good heuristic is to pick the candidate that "uses up" the most "valuable" differences?
            # Or simply randomize to explore different branches.
            random.shuffle(candidates)
            
            # Try candidates
            for x, new_diffs in candidates:
                new_elements = elements + [x]
                new_used_vals = used_vals | {x}
                new_used_diffs = used_diffs | set(new_diffs)
                
                result = backtrack(new_elements, new_used_vals, new_used_diffs)
                if result is not None:
                    return result
            
            return None

        # Multiple restarts with different random seeds
        for attempt in range(1000):
            if time.time() - start_time > time_budget:
                break
            random.seed(42 + attempt)
            # Start with prescribed elements
            # Sort prescribed to be deterministic base
            base_elements = sorted(prescribed)
            base_used_vals = set(prescribed)
            base_used_diffs = initial_diffs
            
            res = backtrack(base_elements, base_used_vals, base_used_diffs)
            if res is not None:
                if is_valid_sidon(res):
                    return res
        return None

    best_a = None
    
    # Try to find size 20
    time_budget_20 = min(120, deadline - time.time() - 10)
    res = solve_csp(20, time_budget_20)
    if res:
        best_a = sorted(res)
    
    # If failed, try 19
    if not best_a:
        time_budget_19 = min(20, deadline - time.time() - 5)
        res = solve_csp(19, time_budget_19)
        if res:
            best_a = sorted(res)
            
    # Fallback to greedy if CSP fails (unlikely)
    if not best_a:
        a = sorted(prescribed)
        used = set(a)
        diffs = initial_diffs
        for x in range(1, N + 1):
            if x in used:
                continue
            new_diffs = []
            ok = True
            for y in a:
                d = abs(x - y)
                if d in diffs or d in new_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if ok:
                a.append(x)
                used.add(x)
                diffs.update(new_diffs)
                if len(a) >= 20:
                    break
        best_a = a

    return best_a
