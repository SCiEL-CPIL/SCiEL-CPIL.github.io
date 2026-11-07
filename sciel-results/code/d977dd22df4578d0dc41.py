import time
import random

def can_add(S_set, x):
    # S_set is a set of ints
    # Case 1: x is an element
    U = [u for u in S_set if (x + u) in S_set]
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            if (U[i] + U[j]) in S_set:
                return False

    # Case 2: x is a sum
    for u in S_set:
        v = x - u
        if v in S_set and u < v:
            for w in S_set:
                if w != u and w != v:
                    if (u + w) in S_set and (v + w) in S_set:
                        return False
    return True

def construct():
    deadline = time.time() + 55.0
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    while time.time() < deadline:
        # Random restart
        # Try different initializations
        # 1. Upper half
        # 2. Random subset of size 70
        # 3. Greedy from 1..128
        
        # Let's try greedy from random order
        S = set()
        elems = list(range(1, 129))
        random.shuffle(elems)
        for x in elems:
            if can_add(S, x):
                S.add(x)
        
        # Local search on S
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding
            candidates = [x for x in range(1, 129) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try swap: remove y, add x
                # Only if we can increase size? No, swap is 1-for-1.
                # We need to remove y to allow adding x AND z?
                # Let's just try removing a random y and then adding
                if S:
                    y = random.choice(list(S))
                    S.remove(y)
                    # Try to add 2 elements?
                    # Just loop back to "Try adding"
                    improved = True # Force loop to continue
                    # But we removed 1, so size decreased.
                    # We need to ensure we don't oscillate.
                    # Better: try specific swaps
                    pass
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
    return list(best_set)
