import time

def construct():
    start_time = time.time()
    deadline = start_time + 75
    n = 256
    
    def check_addition_fast(candidate, current_set):
        """Check if adding candidate to current_set maintains validity."""
        # For all a < b <= c in new set, a must not divide b+c
        # Only need to check triples involving candidate
        
        # Case 1: candidate is smallest (a), so candidate < b <= c
        # Need: for all b, c in current_set with candidate < b <= c, (b+c) % candidate != 0
        if candidate > 1:
            for b in current_set:
                if b <= candidate:
                    continue
                for c in current_set:
                    if c < b:
                        continue
                    if (b + c) % candidate == 0:
                        return False
        
        # Case 2: candidate is middle (b), so a < candidate <= c
        # Need: for all a, c in current_set with a < candidate <= c, (a+c) % a != 0
        for a in current_set:
            if a >= candidate:
                continue
            for c in current_set:
                if c < candidate:
                    continue
                if (a + c) % a == 0:
                    return False
        
        # Case 3: candidate is largest (c), so a < b <= candidate
        # Need: for all a, b in current_set with a < b <= candidate, (b+candidate) % a != 0
        for a in current_set:
            for b in current_set:
                if b <= a:
                    continue
                if b > candidate:
                    continue
                if (b + candidate) % a == 0:
                    return False
        
        return True
    
    # Start with seed
    best_set = set(range(192, 257))
    
    # Try greedy additions from various orders
    for order in [range(1, n+1), range(n, 0, -1), range(128, 0, -1), range(1, 129)]:
        if time.time() > deadline:
            break
        s = set()
        for val in order:
            if check_addition_fast(val, s):
                s.add(val)
        if len(s) > len(best_set):
            best_set = s.copy()
    
    # Local search: add elements
    if time.time() < deadline:
        s = set(best_set)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for val in range(1, n + 1):
                if val not in s and check_addition_fast(val, s):
                    s.add(val)
                    changed = True
                    if len(s) > len(best_set):
                        best_set = s.copy()
                    break
    
    # Try removing one element and adding others
    if time.time() < deadline - 10:
        s = set(best_set)
        for rem in list(s):
            if time.time() > deadline - 5:
                break
            s2 = s - {rem}
            added_any = False
            for val in range(1, n + 1):
                if val not in s2 and check_addition_fast(val, s2):
                    s2.add(val)
                    added_any = True
            if len(s2) > len(best_set):
                best_set = s2
                s = s2
    
    return list(best_set)
