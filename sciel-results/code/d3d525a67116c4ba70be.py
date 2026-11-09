import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def can_add(S_set, x):
        """Check if adding x to S_set keeps it valid."""
        if x in S_set:
            return True
        # Case 1: a=x, b,c in S with x < b <= c, check x ∤ (b+c)
        # Case 2: a in S, a < x <= c, c in S, check a ∤ (x+c)
        # Case 3: a,b in S, a < b <= x, check a ∤ (b+x)
        
        S_list = sorted(S_set)
        
        # Case 1: x is smallest
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for c in S_list[i:]:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        # Case 2: x is middle (a < x <= c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        # Case 3: x is largest (a < b <= x)
        for i, a in enumerate(S_list):
            for b in S_list[i+1:]:
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        return True
    
    def full_validate(S):
        """Full O(k^3) validation."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            for j in range(i+1, k):
                for l in range(j, k):
                    a, b, c = S_sorted[i], S_sorted[j], S_sorted[l]
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True
    
    best_set = set(range(171, 257))  # Known valid, 86 elements
    best_size = len(best_set)
    
    # Try different starting tails
    for L in range(160, 172):
        if time.time() > deadline - 5:
            break
        S = set(range(L, 257))
        if not full_validate(S):
            continue
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Greedy addition: try adding elements from 1 to L-1 in various orders
    for L in range(160, 172):
        if time.time() > deadline - 10:
            break
        base = set(range(L, 257))
        if not full_validate(base):
            continue
        
        # Ascending order
        S = set(base)
        for x in range(1, L):
            if time.time() > deadline - 5:
                break
            if can_add(S, x):
                S.add(x)
        if len(S) > best_size and full_validate(S):
            best_set = set(S)
            best_size = len(S)
        
        # Descending order
        S = set(base)
        for x in range(L-1, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add(S, x):
                S.add(x)
        if len(S) > best_size and full_validate(S):
            best_set = set(S)
            best_size = len(S)
    
    # Try local search: swap elements
    if time.time() < deadline - 15:
        S = set(best_set)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            S_list = sorted(S)
            for i, elem in enumerate(S_list):
                if time.time() > deadline - 10:
                    break
                for x in range(1, n+1):
                    if x in S:
                        continue
                    S.remove(elem)
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
                    S.add(elem)
                if time.time() > deadline - 10:
                    break
    
    return list(best_set)
