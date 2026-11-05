import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def check_valid(S_list):
        """Check if S_list (sorted) satisfies the property."""
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def is_valid_add(S_list, m):
        """Check if adding m to S_list (sorted) maintains the property. Returns True if valid."""
        # Find where m would be inserted
        import bisect
        pos = bisect.bisect_left(S_list, m)
        
        # Case 1: m is smallest (a=m), need m ∤ (b+c) for all b,c in S with b > m, c >= b
        for j in range(pos, len(S_list)):
            b = S_list[j]
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % m == 0:
                    return False
        
        # Case 2: m is middle (b=m), need a ∤ (m+c) for all a in S with a < m, c in S with c >= m
        for i in range(0, pos):
            a = S_list[i]
            for k in range(pos, len(S_list)):
                c = S_list[k]
                if (m + c) % a == 0:
                    return False
        
        # Case 3: m is largest (c=m), need a ∤ (b+m) for all a < b in S with b <= m
        for i in range(0, pos):
            a = S_list[i]
            for j in range(i + 1, pos):
                b = S_list[j]
                if (b + m) % a == 0:
                    return False
        
        return True
    
    # Strategy: Start with {171,...,256}, try multiple random orders to add elements
    base_set = list(range(171, n + 1))
    best_S = list(base_set)
    best_size = len(best_S)
    
    # Precompute conflicts for faster checking
    # For each a, precompute which pairs (b,c) with b,c > a have a | (b+c)
    
    # Multiple random restarts with different addition orders
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        S = list(base_set)
        candidates = list(range(1, 171))
        
        if restart % 2 == 0:
            # Ascending order with some randomization
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random())
        else:
            # Descending order with randomization
            random.shuffle(candidates)
            candidates.sort(key=lambda x: -random.random())
        
        for m in candidates:
            if time.time() > deadline:
                break
            if is_valid_add(S, m):
                import bisect
                pos = bisect.bisect_left(S, m)
                S.insert(pos, m)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Local search: try to add more elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S = list(best_S)
        in_S = set(S)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]
        
        for add_m in not_in_S:
            if time.time() > deadline:
                break
            if is_valid_add(S, add_m):
                import bisect
                pos = bisect.bisect_left(S, add_m)
                S.insert(pos, add_m)
                best_S = list(S)
                best_size = len(S)
                improved = True
                in_S = set(S)
                not_in_S = [m for m in range(1, n + 1) if m not in in_S]
                break
    
    # Final verification
    S_final = sorted(best_S)
    if not check_valid(S_final):
        # Fallback to known valid set
        return list(range(171, 257))
    
    return S_final
