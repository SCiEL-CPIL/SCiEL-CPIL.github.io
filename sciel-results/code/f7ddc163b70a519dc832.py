import time
import random

def _can_add(S, v):
    """
    Check if v can be added to set S while maintaining validity.
    S is a set of integers.
    """
    L = len(S)
    if L == 0:
        return True
    if L == 1:
        return True # Any pair of distinct numbers is valid (no triple a<b<=c)
    
    # We need to check triples (a, b, c) with a < b <= c where at least one is v.
    # Since S is a set, v is not in S.
    
    # We can iterate over pairs in S.
    # To optimize, we can sort S.
    S_list = sorted(S)
    
    # Case 1: v is 'a'.
    # Then we need b, c in S such that v < b <= c and v | b+c.
    # Since S_list is sorted, b and c must be > v.
    # We can find the index where elements > v start.
    # Let idx = first index such that S_list[idx] > v.
    # Then b, c must be in S_list[idx:].
    # We check if any pair (b, c) in S_list[idx:] has (b+c) % v == 0.
    
    # Case 2: v is 'b'.
    # Then we need a, c in S such that a < v <= c and a | v+c.
    # a must be < v. c must be >= v.
    # a in S_list[:idx] (where S_list[idx] > v, so a < S_list[idx]. But we need a < v).
    # Actually, let idx_v = first index such that S_list[idx_v] >= v.
    # Then a must be in S_list[:idx_v] (so a < v).
    # c must be in S_list[idx_v:] (so c >= v).
    # Wait, condition is a < v <= c.
    # So a < v and c >= v.
    # We iterate a in S_list where a < v.
    # And c in S_list where c >= v.
    # Check if (v + c) % a == 0.
    
    # Case 3: v is 'c'.
    # Then we need a, b in S such that a < b < v and a | b+v.
    # a, b must be < v.
    # So a, b in S_list where element < v.
    # We iterate pairs (a, b) in S_list[:idx_v] with a < b.
    # Check if (b + v) % a == 0.
    
    # Let's find idx_v: first index where S_list[idx_v] >= v.
    # Python bisect can help, but for small L, linear search or manual is fine.
    # L is at most 256.
    
    # Optimization: Pre-compute or just loop.
    
    # Find indices
    # idx_lt_v: count of elements < v
    # idx_ge_v: index of first element >= v
    
    # Let's just loop.
    
    # Case 1: v is 'a'
    # b, c > v
    # Find first index where S_list[i] > v
    start_gt_v = 0
    while start_gt_v < L and S_list[start_gt_v] <= v:
        start_gt_v += 1
    
    for i in range(start_gt_v, L):
        b = S_list[i]
        # c must be >= b and > v. Since i is start_gt_v, all j >= i have S_list[j] >= b > v.
        for j in range(i, L):
            c = S_list[j]
            if (b + c) % v == 0:
                return False
    
    # Case 2: v is 'b'
    # a < v, c >= v
    # a in S_list[:start_gt_v] ? No.
    # a < v.
    # Let's find end_lt_v: index of first element >= v.
    # start_gt_v was first element > v.
    # Elements equal to v? v is not in S. So start_gt_v is also first element >= v.
    # So a must be in S_list[:start_gt_v].
    # c must be in S_list[start_gt_v:].
    
    for i in range(start_gt_v):
        a = S_list[i]
        # a < v is guaranteed.
        for j in range(start_gt_v, L):
            c = S_list[j]
            # c >= v is guaranteed.
            if (v + c) % a == 0:
                return False
    
    # Case 3: v is 'c'
    # a < b < v
    # a, b in S_list[:start_gt_v].
    for i in range(start_gt_v):
        a = S_list[i]
        for j in range(i + 1, start_gt_v):
            b = S_list[j]
            # a < b < v is guaranteed.
            if (b + v) % a == 0:
                return False
    
    return True

def _greedy_with_order(n, order):
    """
    Runs the greedy algorithm using the specified order of candidates.
    Returns the set of selected numbers.
    """
    S = set()
    S_list = [] # Keep a list for fast access? _can_add sorts S anyway.
    # Actually, _can_add sorts S every time. This is O(K log K) per insertion.
    # Total O(N * K^2 log K) roughly?
    # N=256. K~150.
    # 256 * 150^2 * log(150) ~ 256 * 22500 * 7 ~ 40M ops.
    # This might be too slow for 500 restarts in 50s.
    # 40M ops in Python is ~ 2-4 seconds.
    # 500 restarts * 3s = 1500s. Too slow.
    
    # We need a faster _can_add or a smarter greedy.
    # The current _can_add in the prompt was O(K^2) without sorting overhead if we maintained sorted list?
    # The prompt's _greedy_with_order did O(K^2) checks inline.
    # Let's revert to inline checks but optimized.
    
    # However, the task asks to COMBINE.
    # Inspiration 1 used interval logic.
    # Inspiration 2 used explicit can_add.
    # Current program uses inline checks.
    
    # Let's try to speed up the inline checks.
    # In the prompt's code, it iterated all pairs (b,c) in S.
    # We can maintain S_list sorted.
    # When adding v, we insert into S_list to keep it sorted.
    
    # Let's rewrite _greedy_with_order to maintain sorted S_list.
    
    S_set = set()
    S_list = []
    
    for v in order:
        if v in S_set:
            continue
            
        # Check validity
        valid = True
        L = len(S_list)
        
        # Find position of v in S_list (for bisect-like logic)
        # Since S_list is sorted, we can find idx.
        # But we don't need to insert yet.
        
        # Case 1: v is 'a'. v < b <= c.
        # b, c must be > v.
        # Find first index in S_list where element > v.
        # Since S_list is sorted, we can binary search or linear scan.
        # Linear scan from end? No, from start.
        # Actually, we can just iterate.
        
        # To speed up, we can pre-compute sums? No, v changes.
        
        # Let's use the same logic as _can_add but inline and using sorted S_list.
        
        # Find idx_ge_v: first index where S_list[i] >= v.
        # Since v not in S, this is first where S_list[i] > v.
        idx_ge_v = 0
        while idx_ge_v < L and S_list[idx_ge_v] < v:
            idx_ge_v += 1
            if time.time() > 85: # Safety
                return S_set
        # Now S_list[:idx_ge_v] < v. S_list[idx_ge_v:] > v.
        
        # Case 1: v is 'a'.
        # b, c in S_list[idx_ge_v:].
        for i in range(idx_ge_v, L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                if (b + c) % v == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Case 2: v is 'b'.
            # a < v, c >= v.
            # a in S_list[:idx_ge_v].
            # c in S_list[idx_ge_v:].
            for i in range(idx_ge_v):
                a = S_list[i]
                for j in range(idx_ge_v, L):
                    c = S_list[j]
                    if (v + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Case 3: v is 'c'.
            # a < b < v.
            # a, b in S_list[:idx_ge_v].
            for i in range(idx_ge_v):
                a = S_list[i]
                for j in range(i + 1, idx_ge_v):
                    b = S_list[j]
                    if (b + v) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            S_set.add(v)
            # Insert v into S_list to keep sorted.
            # Binary insertion or linear.
            # Linear insertion is O(L).
            pos = idx_ge_v
            S_list.insert(pos, v)
            
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 40s safety margin
    
    n = 256
    numbers = list(range(1, n + 1))
    
    best_set = set()
    
    # 1. Deterministic greedy (descending)
    # Descending order is good because large numbers are less likely to be 'a' (divisor).
    # If v is large, it's rarely a divisor of b+c.
    order_desc = sorted(numbers, reverse=True)
    S = _greedy_with_order(n, order_desc)
    if len(S) > len(best_set):
        best_set = S
        
    # 2. Random restarts
    attempts = 0
    while time.time() < deadline:
        random.shuffle(numbers)
        S = _greedy_with_order(n, numbers)
        if len(S) > len(best_set):
            best_set = S
        attempts += 1
        if attempts % 10 == 0:
            # Check time
            pass
            
    # 3. Local Search / Polishing
    # Try to add any number that was skipped.
    improved = True
    while improved:
        improved = False
        # Try to add numbers in descending order? Or random?
        # Descending is safer for 'a' role.
        for v in range(n, 0, -1):
            if v in best_set:
                continue
            if _can_add(best_set, v):
                best_set.add(v)
                improved = True
                break # Restart loop as set changed
                
    # 4. Try to swap?
    # If we can't add v, maybe we can remove w and add v?
    # This is complex. Let's stick to addition first.
    
    # 5. Final check and return
    return sorted(best_set)
