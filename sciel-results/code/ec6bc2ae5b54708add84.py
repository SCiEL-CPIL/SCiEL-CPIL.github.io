import random
import time
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    for i in range(1, N + 1):
        gcd_table[i][i] = i

    # Precompute conflicts
    # conflicts[y] is a set of tuples (a, b) with a < b such that gcd(a,b)==gcd(a,y)==gcd(b,y)
    conflicts = [set() for _ in range(N + 1)]
    for y in range(1, N + 1):
        for a in range(1, N + 1):
            if a == y: continue
            for b in range(a + 1, N + 1):
                if b == y: continue
                if gcd_table[a][b] == gcd_table[a][y] == gcd_table[b][y]:
                    conflicts[y].add((a, b))
                    # Note: (b,a) is not added, we enforce a<b in check

    def is_valid_add(S, y):
        """Check if adding y to S creates a bad triple."""
        # S is a list of elements.
        # We need to check if any pair (a,b) in S is in conflicts[y]
        # To do this efficiently, we can convert S to a set? No, we need to iterate pairs.
        # Optimized: For each a in S, check if any b in S (b != a) is in conflicts with a for y.
        # Since conflicts[y] contains (a,b) with a<b, we can do:
        for a in S:
            for b in S:
                if a == b:
                    continue
                if a < b:
                    if (a, b) in conflicts[y]:
                        return False
                else:
                    if (b, a) in conflicts[y]:
                        return False
        return True

    # Even faster: Precompute for each y, a list of "conflict neighbors"?
    # Actually, the double loop in is_valid_add is O(|S|^2).
    # With set lookup, it's fast.

    # Greedy Initialization
    S = []
    for x in elements:
        if is_valid_add(S, x):
            S.append(x)
    
    best_size = len(S)
    best_S = list(S)
    
    # Local Search
    start_time = time.time()
    deadline = start_time + 55.0
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try to add
        added = False
        candidates_add = [x for x in elements if x not in S]
        random.shuffle(candidates_add)
        
        for y in candidates_add:
            if is_valid_add(S, y):
                S.append(y)
                added = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = list(S)
                break
        
        if not added:
            # Try to swap
            swapped = False
            candidates_remove = list(S)
            random.shuffle(candidates_remove)
            
            for x in candidates_remove:
                candidates_swap = [y for y in elements if y not in S]
                random.shuffle(candidates_swap)
                
                for y in candidates_swap:
                    # Check if S \ {x} U {y} is valid
                    # This is equivalent to checking if y is valid with respect to S \ {x}
                    # We can temporarily remove x, check, and add back if invalid?
                    # Or just construct the list?
                    # S is small, list comprehension is fast.
                    S_temp = [v for v in S if v != x]
                    if is_valid_add(S_temp, y):
                        S.remove(x)
                        S.append(y)
                        swapped = True
                        break
                
                if swapped:
                    break
            
            if not swapped:
                # Kick: remove random, refill
                if S:
                    x_to_remove = random.choice(S)
                    S.remove(x_to_remove)
                    # Greedy fill
                    for y in elements:
                        if y not in S and is_valid_add(S, y):
                            S.append(y)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)

    return best_S
