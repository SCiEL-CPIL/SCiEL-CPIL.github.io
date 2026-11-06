"""Improved program: bitmask-based greedy + local search refinement."""
import random

def construct():
    n = 256
    
    def popcount(x):
        return bin(x).count('1')
    
    def count_3aps(M):
        """Count distinct 3-term APs with positive difference using bitmask."""
        count = 0
        for d in range(1, n):
            # APs: a, a+d, a+2d
            count += popcount(M & (M >> d) & (M >> (2*d)))
        return count

    def has_4ap_with(M, x):
        """Check if adding x to M creates any nonconstant 4-term AP.
        M is the bitmask of current set (without x). x is 1-indexed.
        """
        for d in range(1, n):
            # 1. x, x+d, x+2d, x+3d
            if x + 3*d <= n:
                if ((M >> (x+d-1)) & 1) and ((M >> (x+2*d-1)) & 1) and ((M >> (x+3*d-1)) & 1):
                    return True
            
            # 2. x-d, x, x+d, x+2d
            if x - d >= 1:
                if ((M >> (x-d-1)) & 1) and ((M >> (x+d-1)) & 1) and ((M >> (x+2*d-1)) & 1):
                    return True
            
            # 3. x-2d, x-d, x, x+d
            if x - 2*d >= 1:
                if ((M >> (x-2*d-1)) & 1) and ((M >> (x-d-1)) & 1) and ((M >> (x+d-1)) & 1):
                    return True
            
            # 4. x-3d, x-2d, x-d, x
            if x - 3*d >= 1:
                if ((M >> (x-3*d-1)) & 1) and ((M >> (x-2*d-1)) & 1) and ((M >> (x-d-1)) & 1):
                    return True
        return False

    def count_new_3aps(M, x):
        """Count how many new 3-APs are formed by adding x to M."""
        count = 0
        for d in range(1, n):
            # 1. x, x+d, x+2d
            if x + 2*d <= n:
                if ((M >> (x+d-1)) & 1) and ((M >> (x+2*d-1)) & 1):
                    count += 1
            # 2. x-d, x, x+d
            if x - d >= 1:
                if ((M >> (x-d-1)) & 1) and ((M >> (x+d-1)) & 1):
                    count += 1
            # 3. x-2d, x-d, x
            if x - 2*d >= 1:
                if ((M >> (x-2*d-1)) & 1) and ((M >> (x-d-1)) & 1):
                    count += 1
        return count

    # Greedy construction with bitmask
    best_M = 0
    best_count = 0
    remaining = list(range(1, n+1))
    random.seed(42)
    
    # Multiple restarts to find a good initial set
    for restart in range(20):
        M_temp = 0
        rem = remaining.copy()
        random.shuffle(rem)
        
        while rem:
            best_x = None
            best_gain = -1
            for x in rem:
                if has_4ap_with(M_temp, x):
                    continue
                gain = count_new_3aps(M_temp, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            M_temp |= (1 << (best_x - 1))
            rem.remove(best_x)
        
        count = count_3aps(M_temp)
        if count > best_count:
            best_count = count
            best_M = M_temp

    # Local search on best_M
    current_M = best_M
    current_count = best_count
    all_elements = list(range(1, n+1))
    
    improved = True
    iterations = 0
    max_iterations = 100
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try removing one and adding one (swap)
        current_elements = [i+1 for i in range(n) if (current_M >> i) & 1]
        random.shuffle(current_elements)
        
        for a in current_elements:
            M_without_a = current_M & ~(1 << (a - 1))
            
            for b in all_elements:
                if (M_without_a >> (b - 1)) & 1:
                    continue
                if has_4ap_with(M_without_a, b):
                    continue
                M_new = M_without_a | (1 << (b - 1))
                new_count = count_3aps(M_new)
                if new_count > current_count:
                    current_M = M_new
                    current_count = new_count
                    improved = True
                    break
            if improved:
                break
        
        # If no swap improved, try adding one
        if not improved:
            for b in all_elements:
                if (current_M >> (b - 1)) & 1:
                    continue
                if has_4ap_with(current_M, b):
                    continue
                M_new = current_M | (1 << (b - 1))
                new_count = count_3aps(M_new)
                if new_count > current_count:
                    current_M = M_new
                    current_count = new_count
                    improved = True
                    break

    result = [i+1 for i in range(n) if (current_M >> i) & 1]
    return sorted(result)
