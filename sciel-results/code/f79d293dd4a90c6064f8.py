import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 256
    
    def is_valid_set(S):
        # S is a set of integers
        # Check: for all a < b <= c in S, a does not divide b+c
        S_list = list(S)
        S_list.sort()
        for i, a in enumerate(S_list):
            # b must be > a
            # c must be >= b
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add_x(S, x):
        """Check if x can be added to S without violating constraints.
        x must be smaller than all elements in S for the 'a' role.
        Also need to check if x acts as b or c with existing smaller elements."""
        if x in S:
            return False
        S_list = sorted(S)
        
        # x as 'a': for all b, c in S with b <= c, x must not divide b+c
        for i, b in enumerate(S_list):
            for k in range(i, len(S_list)):
                c = S_list[k]
                if (b + c) % x == 0:
                    return False
        
        # x as 'b' or 'c': for all a in S with a < x, and b, c in S∪{x}
        # where b <= c and at least one of b,c is x
        # Case 1: b=x, c in S with c >= x
        # Case 2: b in S with b >= x, c=x (but b<=c means b<=x, so b=x again)
        # Actually if x is in the set, for a < b <= c:
        # - a < x, b = x, c >= x: check a does not divide x+c
        # - a < x, b < x, c = x: check a does not divide b+x
        # - a < x, b = x, c = x: check a does not divide 2x
        # - a >= x: x is not b or c in a way that creates new constraints beyond what's checked
        
        for a in S_list:
            if a >= x:
                continue
            # a < x, check pairs (b,c) where at least one is x
            # b=x, c=x: a | 2x?
            if (2 * x) % a == 0:
                return False
            # b=x, c in S with c >= x
            for c in S_list:
                if c >= x and (x + c) % a == 0:
                    return False
            # b in S with x <= b < x? No, b must be >= x and b <= c=x, so b=x
            # b in S with b < x, c=x: a | b+x
            for b in S_list:
                if b < x and (b + x) % a == 0:
                    return False
        
        return True
    
    def can_add_y_to_set(S, y):
        """General check: can y be added to S? y could be anywhere in the ordering."""
        if y in S:
            return False
        S_list = sorted(S)
        
        # For each a in S∪{y} with a < b <= c in S∪{y}, check a ∤ (b+c)
        # Only new constraints involve y
        # y as a: for all b,c in S with b<=c, y ∤ (b+c)
        for i, b in enumerate(S_list):
            for k in range(i, len(S_list)):
                c = S_list[k]
                if (b + c) % y == 0:
                    return False
        
        # y as b or c: for all a in S with a < y, and pairs involving y
        for a in S_list:
            if a >= y:
                continue
            # a < y
            # b=y, c=y: a | 2y?
            if (2 * y) % a == 0:
                return False
            # b=y, c in S with c > y: a | (y+c)?
            for c in S_list:
                if c > y and (y + c) % a == 0:
                    return False
            # b in S with y <= b, c=y: but b<=c=y means b<=y, so b=y (already covered)
            # b in S with b < y, c=y: a | (b+y)?
            for b in S_list:
                if b < y and (b + y) % a == 0:
                    return False
        
        return True

    # 1. Find the best interval [L, 256]
    best_L = 1
    best_size = 0
    best_set = set()
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Quick check for a=L
        k_min = (2 * (L + 1) + L - 1) // L
        k_max = (2 * n) // L
        if k_min <= k_max:
            continue
        
        # If L passes the quick check, verify full validity for [L, 256]
        S_test = set(range(L, n + 1))
        if is_valid_set(S_test):
            best_L = L
            best_size = n - L + 1
            best_set = S_test
            break

    # If no interval was found, fallback
    if not best_set:
        return list(range(192, 257))

    # 2. Greedy Expansion
    current_set = set(best_set)
    
    # Try multiple orders for greedy addition
    best_result = set(current_set)
    
    # Order 1: descending from best_L-1
    for x in range(best_L - 1, 0, -1):
        if time.time() > deadline:
            break
        if can_add_y_to_set(current_set, x):
            current_set.add(x)
    
    if len(current_set) > len(best_result):
        best_result = set(current_set)
    
    # 3. Local Search: Try swaps to improve
    # Try to remove one element and add two smaller ones, or swap
    improved = True
    max_iterations = 50
    
    while improved and time.time() < deadline - 3.0:
        improved = False
        current_set = set(best_result)
        
        # Try adding elements that weren't added
        for x in range(1, best_L):
            if time.time() > deadline:
                break
            if x not in current_set and can_add_y_to_set(current_set, x):
                current_set.add(x)
                improved = True
        
        # Try swaps: remove y, add x where x < y
        if not improved:
            S_list = sorted(current_set)
            for i, y in enumerate(S_list):
                if time.time() > deadline:
                    break
                # Try removing y and adding some x < y
                for x in range(1, y):
                    if x in current_set:
                        continue
                    # Tentatively remove y
                    test_set = set(current_set)
                    test_set.remove(y)
                    if can_add_y_to_set(test_set, x):
                        test_set.add(x)
                        # Check if this leads to further improvements
                        # For now, just accept if size doesn't decrease
                        if len(test_set) >= len(current_set):
                            current_set = test_set
                            improved = True
                            break
                if improved:
                    break
        
        if len(current_set) > len(best_result):
            best_result = set(current_set)
            improved = True
    
    # 4. Final polish: try to add any remaining elements
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x not in best_result and can_add_y_to_set(best_result, x):
            best_result.add(x)
    
    return list(best_result)

# EVOLVE-BLOCK-END
