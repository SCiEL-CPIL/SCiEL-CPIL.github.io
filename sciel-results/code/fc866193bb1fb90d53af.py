import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute divisibility: div[a] = set of multiples of a up to 2n
    # For checking if a | (b+c), we need b+c to be in multiples of a
    max_sum = 2 * n  # b+c can be up to 2*256 = 512
    multiples = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for m in range(a, max_sum + 1, a):
            multiples[a].add(m)
    
    def can_add_fast(S, x):
        """Check if adding x to set S keeps it valid. S is a set."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Case 1: a=x, need b,c in S with x < b <= c and x | (b+c)
        # b+c must be in multiples[x]
        S_above = [v for v in S_list if v > x]
        for i, b in enumerate(S_above):
            for c in S_above[i:]:
                if (b + c) in multiples[x]:
                    return False
        
        # Case 2: b=x, need a,c in S with a < x <= c and a | (x+c)
        S_below = [v for v in S_list if v < x]
        S_above_or_eq = [v for v in S_list if v >= x]
        for a in S_below:
            for c in S_above_or_eq:
                if (x + c) in multiples[a]:
                    return False
        
        # Case 3: c=x, need a,b in S with a < b <= x and a | (b+x)
        S_up_to_x = [v for v in S_list if v <= x]
        for i, a in enumerate(S_up_to_x):
            for b in S_up_to_x[i+1:]:
                if (b + x) in multiples[a]:
                    return False
        
        return True
    
    def greedy_construct(threshold, order=None):
        """Greedy construction starting from [threshold, 256]."""
        S = set(range(threshold, 257))
        if order is None:
            order = list(range(1, threshold))
        for candidate in order:
            if time.time() > deadline - 5:
                break
            if candidate in S:
                continue
            if can_add_fast(S, candidate):
                S.add(candidate)
        return S
    
    def count_conflicts(x, S):
        """Count how many conflicts x would have with S if added."""
        S_list = sorted(S)
        conflicts = 0
        
        # Case 1: a=x
        S_above = [v for v in S_list if v > x]
        for i, b in enumerate(S_above):
            for c in S_above[i:]:
                if (b + c) in multiples[x]:
                    conflicts += 1
        
        # Case 2: b=x
        S_below = [v for v in S_list if v < x]
        S_above_or_eq = [v for v in S_list if v >= x]
        for a in S_below:
            for c in S_above_or_eq:
                if (x + c) in multiples[a]:
                    conflicts += 1
        
        # Case 3: c=x
        S_up_to_x = [v for v in S_list if v <= x]
        for i, a in enumerate(S_up_to_x):
            for b in S_up_to_x[i+1:]:
                if (b + x) in multiples[a]:
                    conflicts += 1
        
        return conflicts
    
    # Try multiple starting thresholds and orders
    best = set(range(171, 257))
    best_size = len(best)
    
    thresholds = [171, 180, 190, 200, 210, 220]
    
    for threshold in thresholds:
        if time.time() > deadline - 15:
            break
        
        # Strategy 1: Try adding elements in increasing order (smaller first)
        order_increasing = list(range(1, threshold))
        S = greedy_construct(threshold, order_increasing)
        if len(S) > best_size:
            best = S
            best_size = len(S)
        
        # Strategy 2: Try adding elements in decreasing order
        order_decreasing = list(range(threshold - 1, 0, -1))
        S = greedy_construct(threshold, order_decreasing)
        if len(S) > best_size:
            best = S
            best_size = len(S)
        
        # Strategy 3: Weighted greedy - prioritize elements with fewer conflicts
        if time.time() > deadline - 10:
            continue
        
        S = set(range(threshold, 257))
        remaining = list(range(1, threshold))
        
        while remaining and time.time() < deadline - 5:
            # Find the element with fewest conflicts
            best_candidate = None
            min_conflicts = float('inf')
            
            for x in remaining:
                conflicts = count_conflicts(x, S)
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_candidate = x
            
            if best_candidate is None:
                break
            
            if min_conflicts == 0:
                S.add(best_candidate)
                remaining.remove(best_candidate)
            else:
                # No element can be added without conflicts
                break
        
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    return sorted(best)
