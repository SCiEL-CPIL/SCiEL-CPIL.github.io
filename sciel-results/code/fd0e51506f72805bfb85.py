import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    def is_valid(S):
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i + 1, m):
                b = S_sorted[j]
                if (2 * b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(S, x):
        S_list = sorted(S)
        m = len(S_list)
        
        # Case 1: x is the smallest element 'a'
        for i in range(m):
            b = S_list[i]
            if b <= x:
                continue
            if (2 * b) % x == 0:
                return False
            for j in range(i + 1, m):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element 'b'
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element 'c'
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def greedy_from_largest(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def local_search(S, max_swaps=200):
        S = set(S)
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps:
            improved = False
            swaps_done += 1
            S_list = sorted(S)
            candidates_out = list(S_list)
            random.shuffle(candidates_out)
            candidates_in = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(candidates_in)
            
            for out_elem in candidates_out:
                if time.time() > deadline - 2.0:
                    return S
                S_test = S - {out_elem}
                for in_elem in candidates_in:
                    if can_add(S_test, in_elem):
                        S = S_test | {in_elem}
                        improved = True
                        break
                if improved:
                    break
        return S
    
    best_set = set()
    best_size = 0
    
    # Key structural insight: [171, 256] is always valid (size 86)
    # because for any a >= 171, 3a >= 513 > 511 >= b+c for b,c >= 172
    base_set = set(range(171, n + 1))
    best_set = set(base_set)
    best_size = len(base_set)
    
    # Strategy 1: Start with [171,256] and try to add smaller elements
    # Process smaller elements in various orders
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        S = set(base_set)
        
        # Try adding elements from 1 to 170 in different orders
        candidates = list(range(1, 171))
        
        strategy = restart % 5
        if strategy == 0:
            # Descending
            candidates.sort(reverse=True)
        elif strategy == 1:
            # Ascending
            candidates.sort()
        elif strategy == 2:
            # Random
            rng.shuffle(candidates)
        elif strategy == 3:
            # Prefer elements that are "safe" (large, close to 171)
            candidates.sort(key=lambda x: -x)
            rng.shuffle(candidates[:20])  # Shuffle top 20
        else:
            # Weighted: prefer elements not divisible by small numbers
            def safety_score(x):
                # Count how many a in [1,x-1] have a | (x + something)
                # Simpler: prefer larger x, and x with fewer small divisors
                divisors = sum(1 for d in range(1, x) if x % d == 0)
                return (-divisors, -x)
            candidates.sort(key=safety_score)
        
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Local search to try to swap in more elements
        if len(S) >= best_size - 3:
            S_improved = local_search(S, max_swaps=30)
            S = S_improved
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 2: Start with [171,256], try removing some large elements to add more small ones
    for restart in range(30):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(1000 + restart * 31)
        
        S = set(base_set)
        
        # Try removing 1-3 elements from the base set to allow more additions
        remove_count = rng.randint(0, 3)
        if remove_count > 0:
            to_remove = rng.sample(list(base_set), remove_count)
            S -= set(to_remove)
        
        # Now try adding elements from 1 to 170
        candidates = list(range(1, 171))
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 2.0:
                break
            if can_add(S, x):
                S.add(x)
        
        # Also try re-adding removed elements
        for x in to_remove if remove_count > 0 else []:
            if can_add(S, x):
                S.add(x)
        
        S_improved = local_search(S, max_swaps=20)
        S = S_improved
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Strategy 3: Pure greedy from largest to smallest (original)
    S1 = greedy_from_largest(range(n, 0, -1))
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = set(S1)
    
    # Strategy 4: Greedy with bias toward large elements
    for restart in range(30):
        if time.time() > deadline - 3.0:
            break
        
        rng = random.Random(2000 + restart * 47)
        
        S = set()
        # Process elements in a biased order
        order = list(range(n, 0, -1))
        # Add some randomness to the ordering
        for i in range(len(order) - 1):
            if rng.random() < 0.1:
                j = i + rng.randint(1, min(20, len(order) - i - 1))
                order[i], order[j] = order[j], order[i]
        
        for x in order:
            if time.time() > deadline - 2.0:
                break
            if can_add(S, x):
                S.add(x)
        
        S_improved = local_search(S, max_swaps=20)
        S = S_improved
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final verification
    if not best_set:
        best_set = set(range(171, n + 1))
    
    if not is_valid(best_set):
        # Fallback: rebuild
        best_set = set(range(171, n + 1))
    
    return sorted(best_set)
