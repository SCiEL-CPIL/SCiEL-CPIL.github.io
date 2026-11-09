# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 256
    deadline = time.time() + 85.0
    
    def can_add(x, S):
        """Check if adding x to set S maintains validity. O(|S|^2)"""
        if x in S:
            return True
        s = sorted(S)
        # Case 1: x is 'a' (smallest) - need x < b <= c in S, x !| (b+c)
        greater = [v for v in s if v > x]
        for i in range(len(greater)):
            for j in range(i, len(greater)):
                b, c = greater[i], greater[j]
                if (b + c) % x == 0:
                    return False
        # Case 2: x is 'b' (middle) - need a < x <= c in S, a !| (x+c)
        less = [v for v in s if v < x]
        geq = [v for v in s if v >= x]
        for a in less:
            for c in geq:
                if (x + c) % a == 0:
                    return False
        # Case 3: x is 'c' (largest) - need a < b <= x in S, a !| (b+x)
        leq = [v for v in s if v <= x]
        for i in range(len(leq)):
            for j in range(i+1, len(leq)):
                a, b = leq[i], leq[j]
                if (b + x) % a == 0:
                    return False
        return True
    
    best_set = set()
    best_size = 0
    
    # Try different starting tails
    for k in [128, 144, 160, 171, 192, 200]:
        if time.time() > deadline - 15:
            break
        current_set = set(range(k, n + 1))
        
        # Try adding elements in ascending order
        for x in range(1, k):
            if time.time() > deadline - 10:
                break
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
    
    # Multiple random restarts from best set
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        current_set = set(best_set)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        improved = False
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if x not in current_set:
                if can_add(x, current_set):
                    current_set.add(x)
                    improved = True
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            if not improved:
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
