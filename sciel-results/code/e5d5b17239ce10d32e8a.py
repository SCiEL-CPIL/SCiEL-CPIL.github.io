import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    def can_add(S, x):
        """Check if adding x to set S keeps it valid. S is a set."""
        if x in S:
            return True
        S_list = sorted(S)
        
        # Case 1: a=x, need b,c in S with x < b <= c and x | (b+c)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: b=x, need a,c in S with a < x <= c and a | (x+c)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: c=x, need a,b in S with a < b <= x and a | (b+x)
        for a in S_list:
            for b in S_list:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def is_valid_set(S):
        """Check if the entire set S is valid."""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if j <= i:
                    continue
                for k, c in enumerate(S_list):
                    if k < j:
                        continue
                    if a < b and b <= c:
                        if (b + c) % a == 0:
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
            if can_add(S, candidate):
                S.add(candidate)
        return S
    
    def local_search(S, max_iterations=100):
        """Try to improve S by swapping elements."""
        S = set(S)
        S_list = sorted(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break
            # Try removing each element and see if we can add a better one
            for i, a in enumerate(S_list):
                if time.time() > deadline - 10:
                    break
                S_without_a = S - {a}
                # Try to add elements not in S that would increase size
                for x in range(1, 257):
                    if x in S:
                        continue
                    if can_add(S_without_a, x):
                        # Check if this swap is beneficial (same size, but maybe enables more)
                        # For now, just accept any valid swap to explore
                        S = S_without_a | {x}
                        S_list = sorted(S)
                        improved = True
                        break
        return S
    
    # Try multiple starting thresholds and orders
    best = set(range(171, 257))
    best_size = len(best)
    
    thresholds = [171, 180, 190, 200, 210, 220]
    orders = [
        list(range(1, 171)),
        list(range(170, 0, -1)),
        sorted(range(1, 171), key=lambda x: -x.bit_count()),  # More bits first
        list(range(2, 171, 2)),  # Even numbers first
        list(range(1, 171, 2)),  # Odd numbers first
    ]
    
    for threshold in thresholds:
        if time.time() > deadline - 15:
            break
        for order in orders:
            if time.time() > deadline - 10:
                break
            # Adjust order to only include elements below threshold
            adjusted_order = [x for x in order if x < threshold]
            S = greedy_construct(threshold, adjusted_order)
            if len(S) > best_size:
                best = S
                best_size = len(S)
    
    # Local search on the best solution
    if time.time() < deadline - 20:
        best = local_search(best, max_iterations=50)
        best_size = len(best)
    
    # Try local search on a few other promising solutions
    if time.time() < deadline - 15:
        for threshold in [171, 180, 190]:
            if time.time() > deadline - 10:
                break
            S = greedy_construct(threshold, list(range(1, threshold)))
            if len(S) >= best_size - 5:
                S = local_search(S, max_iterations=30)
                if len(S) > best_size:
                    best = S
                    best_size = len(S)
    
    return sorted(best)
