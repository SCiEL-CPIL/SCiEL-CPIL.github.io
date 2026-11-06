import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256

    # Precompute bit masks for quick subset checks?
    # We will represent S as a set for logic, and an int bitmask for fast membership.
    
    def is_valid(S):
        """
        Check if S is a valid set.
        S: set of integers
        """
        if not S:
            return True
        
        # Convert to bitmask for O(1) membership
        # n=256, so we need 257 bits (0 to 256). Python int is arbitrary precision.
        mask = 0
        for x in S:
            mask |= (1 << x)
        
        # Sort S for iteration
        S_list = sorted(S)
        
        for i, a in enumerate(S_list):
            if a == 1:
                # 1 divides everything.
                # Need no pair b, c in S with 1 < b <= c.
                # This means S can have at most one element > 1.
                count_gt_1 = sum(1 for x in S_list[i+1:] if x > 1)
                if count_gt_1 >= 2:
                    return False
                continue
            
            # For this a, check all b in S with b > a
            # b is S_list[j] for j > i
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                # We need c in S such that b <= c <= n and a | (b+c)
                # b + c = k * a
                # c = k * a - b
                # Constraints:
                # 1. c >= b  => k * a - b >= b => k * a >= 2b => k >= ceil(2b / a)
                # 2. c <= n  => k * a - b <= n => k * a <= n + b => k <= floor((n + b) / a)
                
                k_min = (2 * b + a - 1) // a
                k_max = (n + b) // a
                
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    # c is automatically >= b and <= n by k range
                    # Check if c is in S
                    if (mask >> c) & 1:
                        return False
        
        return True

    def get_conflicts(S, x):
        """
        If S U {x} is invalid, find one conflicting triple.
        Returns (a, b, c) or None.
        """
        S_new = S | {x}
        mask = 0
        for y in S_new:
            mask |= (1 << y)
        
        S_list = sorted(S_new)
        
        for i, a in enumerate(S_list):
            if a == 1:
                # Check if 1 is the culprit
                if x == 1:
                    # If x=1, check if there are 2 elements > 1 in S
                    gt_1 = [y for y in S if y > 1]
                    if len(gt_1) >= 2:
                        return (1, gt_1[0], gt_1[1])
                else:
                    # x is > 1. If 1 in S, and there are other >1, it was already invalid?
                    # No, S was valid. So if 1 in S, S had at most 1 element > 1.
                    # If x > 1 is added, and 1 in S, and S had 1 element > 1, now we have 2.
                    if 1 in S:
                        gt_1 = [y for y in S if y > 1]
                        if len(gt_1) >= 1:
                            return (1, gt_1[0], x)
                continue
            
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                k_min = (2 * b + a - 1) // a
                k_max = (n + b) // a
                
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if (mask >> c) & 1:
                        # Found conflict (a, b, c)
                        return (a, b, c)
        
        return None

    # Step 1: Find largest valid interval [L, n]
    best_L = n
    for L in range(1, n + 1):
        if time.time() > deadline - 5:
            break
        S_test = set(range(L, n + 1))
        if is_valid(S_test):
            best_L = L
            break
    
    # Step 2: Multi-start Greedy + Local Search
    best_set = set(range(best_L, n + 1))
    best_size = len(best_set)
    
    # Candidates to add
    candidates = list(range(1, best_L))
    
    def greedy_add(S, candidates, time_limit):
        S = set(S)
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in S:
                continue
            
            test = S | {x}
            if is_valid(test):
                S.add(x)
            else:
                # Try to swap: find conflict, remove one element from conflict, add x
                # This is expensive, so only do it occasionally or if we have time
                if time.time() < time_limit - 2:
                    conflict = get_conflicts(S, x)
                    if conflict:
                        a, b, c = conflict
                        # Try removing b or c (if they are not x) or a
                        # We want to keep size, so remove one, add x.
                        # Prefer removing larger elements? Or random?
                        # Let's try removing the largest element in the conflict that is in S
                        # Actually, removing any of them breaks the specific triple, 
                        # but might break validity elsewhere.
                        # Let's just try removing b, then c, then a.
                        for y in [b, c, a]:
                            if y == x:
                                continue
                            if y in S:
                                test_swap = S | {x}
                                test_swap.remove(y)
                                if is_valid(test_swap):
                                    S = test_swap
                                    break
        return S

    # Run multiple attempts
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        S = set(range(best_L, n + 1))
        S = greedy_add(S, candidates, deadline - 10)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Step 3: Local Search on best_set
    # Try to add any remaining element with swaps
    current = set(best_set)
    remaining = [x for x in range(1, n + 1) if x not in current]
    random.shuffle(remaining)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if x in current:
                continue
            test = current | {x}
            if is_valid(test):
                current.add(x)
                improved = True
                best_size = len(current)
                best_set = current
            else:
                # Try swap
                conflict = get_conflicts(current, x)
                if conflict:
                    a, b, c = conflict
                    for y in [b, c, a]:
                        if y == x or y not in current:
                            continue
                        test_swap = current | {x}
                        test_swap.remove(y)
                        if is_valid(test_swap):
                            current = test_swap
                            improved = True
                            best_size = len(current)
                            best_set = current
                            break

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
