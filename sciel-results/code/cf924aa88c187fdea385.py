import random
import time

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin
    
    def can_add(x, S):
        # Check if adding x to S creates a 4-AP
        # S is a set
        # Patterns:
        # 1. x, x+d, x+2d, x+3d
        # 2. x-d, x, x+d, x+2d
        # 3. x-2d, x-d, x, x+d
        # 4. x-3d, x-2d, x-d, x
        
        # Max d for case 1: x+3d <= N => d <= (N-x)//3
        # Max d for case 4: x-3d >= 1 => d <= (x-1)//3
        max_d = (N - x) // 3 if x <= N - 3 else 0
        # Actually, we need to check all d where any of the other terms are in [1, N] and in S.
        
        # Case 1: x is first term
        d = 1
        while x + 3*d <= N:
            if (x+d in S) and (x+2*d in S) and (x+3*d in S):
                return False
            d += 1
            
        # Case 2: x is second term
        d = 1
        while x - d >= 1 and x + 2*d <= N:
            if (x-d in S) and (x+d in S) and (x+2*d in S):
                return False
            d += 1
            
        # Case 3: x is third term
        d = 1
        while x - 2*d >= 1 and x + d <= N:
            if (x-2*d in S) and (x-d in S) and (x+d in S):
                return False
            d += 1
            
        # Case 4: x is fourth term
        d = 1
        while x - 3*d >= 1:
            if (x-3*d in S) and (x-2*d in S) and (x-d in S):
                return False
            d += 1
                
        return True

    def greedy_random():
        S = set()
        candidates = list(range(1, N+1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S

    best_set = set()
    best_size = 0
    
    # Phase 1: Randomized Greedy
    num_greedy = 200
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        S = greedy_random()
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Phase 2: Local Search (Remove-Add)
    # Try to remove one element and add multiple others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S = best_set.copy()
        # Try a limited number of removals
        # Shuffle S to try different elements
        elements = list(S)
        random.shuffle(elements)
        
        # Limit iterations per round to avoid timeout
        max_removals_to_try = 50 
        for u in elements[:max_removals_to_try]:
            if time.time() > deadline:
                break
            S_temp = S.copy()
            S_temp.remove(u)
            added_count = 0
            # Try to add any element not in S_temp
            # Optimization: Only try elements that were not in S or were blocked by u?
            # Simple approach: Try all not in S_temp
            # This is O(N) per removal
            for v in range(1, N+1):
                if v in S_temp:
                    continue
                if can_add(v, S_temp):
                    S_temp.add(v)
                    added_count += 1
                    
            if len(S_temp) > len(S):
                S = S_temp
                best_set = S_temp
                best_size = len(S)
                improved = True
                break # Restart with new best
                
    # Phase 3: Final Polish
    # Try to add any remaining element
    S = best_set
    for v in range(1, N+1):
        if v not in S and can_add(v, S):
            S.add(v)
            
    return sorted(S)
