"""Improved program: greedy maximize 3-APs + fast local search refinement."""
import random

def construct():
    n = 256
    
    def count_3aps(S):
        """Count distinct 3-term APs with positive difference."""
        count = 0
        S_set = set(S)
        for a in S:
            for d in range(1, n):
                b = a + d
                c = a + 2*d
                if b in S_set and c in S_set:
                    count += 1
        return count
        
    def has_4ap_with(S, x):
        """Check if adding x to S creates any nonconstant 4-term AP."""
        S_set = set(S)
        S_set.add(x)
        for d in range(1, n):
            # x as first: x, x+d, x+2d, x+3d
            if x + 3*d <= n and (x+d) in S_set and (x+2*d) in S_set and (x+3*d) in S_set:
                return True
            # x as second: x-d, x, x+d, x+2d
            if x - d >= 1 and (x-d) in S_set and (x+d) in S_set and (x+2*d) in S_set:
                return True
            # x as third: x-2d, x-d, x, x+d
            if x - 2*d >= 1 and (x-2*d) in S_set and (x-d) in S_set and (x+d) in S_set:
                return True
            # x as fourth: x-3d, x-2d, x-d, x
            if x - 3*d >= 1 and (x-3*d) in S_set and (x-2*d) in S_set and (x-d) in S_set:
                return True
        return False

    def count_3aps_involving(x, S):
        """Count 3-APs in S that contain x."""
        S_set = set(S)
        count = 0
        # x as first: x, x+d, x+2d
        for d in range(1, n):
            if x + 2*d <= n and (x+d) in S_set and (x+2*d) in S_set:
                count += 1
        # x as middle: x-d, x, x+d
        for d in range(1, n):
            if x - d >= 1 and (x-d) in S_set and (x+d) in S_set:
                count += 1
        # x as last: x-2d, x-d, x
        for d in range(1, n):
            if x - 2*d >= 1 and (x-2*d) in S_set and (x-d) in S_set:
                count += 1
        return count

    # Greedy: add elements in order of maximum new 3-APs
    S = set()
    remaining = set(range(1, n+1))
    
    while remaining:
        best_x = None
        best_gain = -1
        for x in remaining:
            if has_4ap_with(S, x):
                continue
            gain = count_3aps_involving(x, S)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None:
            break
        S.add(best_x)
        remaining.remove(best_x)
    
    # Local search: try swaps to improve
    random.seed(42)
    current_S = set(S)
    current_count = count_3aps(current_S)
    
    for iteration in range(100):
        improved = False
        best_gain = 0
        best_a = None
        best_b = None
        
        S_list = list(current_S)
        
        for a in S_list:
            loss = count_3aps_involving(a, current_S)
            
            for b in range(1, n+1):
                if b in current_S:
                    continue
                
                S_temp = current_S - {a}
                if has_4ap_with(S_temp, b):
                    continue
                
                gain = count_3aps_involving(b, S_temp)
                
                net_gain = gain - loss
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_a = a
                    best_b = b
        
        if best_gain > 0 and best_a is not None and best_b is not None:
            current_S.remove(best_a)
            current_S.add(best_b)
            current_count += best_gain
            improved = True
            
        if not improved:
            break
            
    return sorted(current_S)
