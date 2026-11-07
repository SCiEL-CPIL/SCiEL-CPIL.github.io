import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def can_add(S, x):
        """Check if S U {x} is valid. O(N^2) worst case."""
        S_list = list(S)
        
        # Precompute: for each b in S, check if x+b is in S
        valid_bs = [b for b in S_list if (x + b) in S]
        
        if len(valid_bs) < 2:
            return True
        
        # For each pair (b, c) in valid_bs, b != c, check if b+c in S or b+c == x
        for i in range(len(valid_bs)):
            b = valid_bs[i]
            for j in range(len(valid_bs)):
                if i == j:
                    continue
                c = valid_bs[j]
                if (b + c) in S or (b + c) == x:
                    return False
        return True

    best_set = set()
    best_size = 0
    
    def try_improve(S):
        """Try to improve set S by adding elements or swapping."""
        nonlocal best_set, best_size
        
        not_in_S = [x for x in range(1, 129) if x not in S]
        
        # Phase 1: Try to add any element that fits
        added = True
        while added:
            added = False
            for x in not_in_S[:]:
                if time.time() > deadline:
                    return S
                if can_add(S, x):
                    S.add(x)
                    not_in_S.remove(x)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
        
        # Phase 2: Local search with swaps
        for _round in range(3):
            if time.time() > deadline:
                break
            
            improved_this_round = False
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if time.time() > deadline:
                    break
                
                if can_add(S, x):
                    S.add(x)
                    not_in_S.remove(x)
                    improved_this_round = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = set(S)
                    continue
                
                S_list = list(S)
                random.shuffle(S_list)
                
                for _ in range(15):
                    if time.time() > deadline:
                        break
                    y = S_list[_ % len(S_list)]
                    
                    S_temp = S - {y}
                    if can_add(S_temp, x):
                        S = S_temp
                        S.add(x)
                        not_in_S.append(y)
                        improved_this_round = True
                        break
                
                if improved_this_round:
                    break
            
            if not improved_this_round:
                break
        
        return S

    random.seed(123)
    
    # Approach 1: Random order greedy
    for restart in range(25):
        if time.time() > deadline:
            break
        
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        S = set()
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        
        S = try_improve(S)
    
    # Approach 2: Descending order greedy
    if time.time() < deadline:
        S = set()
        for x in range(128, 0, -1):
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        S = try_improve(S)
    
    # Approach 3: Ascending order greedy
    if time.time() < deadline:
        S = set()
        for x in range(1, 129):
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        S = try_improve(S)
    
    # Approach 4: Skip-based construction
    if time.time() < deadline:
        S = set()
        for x in range(1, 129, 2):
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        for x in range(2, 129, 2):
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        S = try_improve(S)
    
    # Approach 5: Weighted random (bias toward larger)
    for restart in range(10):
        if time.time() > deadline:
            break
        
        large = [x for x in range(1, 129) if x > 64]
        small = [x for x in range(1, 129) if x <= 64]
        random.shuffle(large)
        random.shuffle(small)
        merged = []
        li, si = 0, 0
        while li < len(large) or si < len(small):
            if li < len(large) and (si >= len(small) or random.random() < 0.5):
                merged.append(large[li])
                li += 1
            elif si < len(small):
                merged.append(small[si])
                si += 1
        
        S = set()
        for x in merged:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.add(x)
        
        S = try_improve(S)
    
    # Final fill
    for x in range(1, 129):
        if x in best_set:
            continue
        if can_add(best_set, x):
            best_set.add(x)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add(best_set, x):
                best_set.add(x)
                changed = True
    
    result = list(best_set)
    return [int(x) for x in result]
