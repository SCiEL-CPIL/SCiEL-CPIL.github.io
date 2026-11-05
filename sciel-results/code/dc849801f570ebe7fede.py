import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    N = 128
    
    # 1. Precompute GCDs
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # 2. Precompute Conflict Lists
    # Conflict[x] is a list of pairs (a, b) such that (x, a, b) is a bad triple.
    # i.e., gcd(x,a) == gcd(x,b) == gcd(a,b)
    # We store pairs as (min(a,b), max(a,b)) to avoid duplicates if we wanted, 
    # but for checking, order doesn't matter as long as we check both.
    # Actually, for a fixed x, the pair is {a, b}.
    conflict = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd_mat[a][b]
                g_ac = gcd_mat[a][c]
                g_bc = gcd_mat[b][c]
                if g_ab == g_ac and g_ab == g_bc:
                    # Triple (a, b, c) is bad
                    # For x=a, the other two are b, c
                    conflict[a].append((b, c))
                    # For x=b, the other two are a, c
                    conflict[b].append((a, c))
                    # For x=c, the other two are a, b
                    conflict[c].append((a, b))
                    
    # 3. Helper function to check if x can be added to current set
    # in_set is a list of booleans, 1-indexed
    def can_add(x, in_set):
        for a, b in conflict[x]:
            if in_set[a] and in_set[b]:
                return False
        return True

    def greedy_build(order):
        in_set = [False] * (N + 1)
        mask = 0
        for x in order:
            if can_add(x, in_set):
                in_set[x] = True
                mask |= (1 << x)
        return mask, in_set

    def local_search(mask, in_set):
        best_mask = mask
        best_size = bin(mask).count('1')
        
        # To speed up, we can precompute the list of elements in the set
        # But updating it on the fly is fine.
        
        while time.time() < deadline:
            improved = False
            
            # Try to add any element
            for x in range(1, N + 1):
                if not in_set[x]:
                    if can_add(x, in_set):
                        in_set[x] = True
                        mask |= (1 << x)
                        best_mask = mask
                        best_size += 1
                        improved = True
                        break # Restart the loop to try more additions
            
            if improved:
                continue
                
            # If no addition was possible, we are in a local maximum.
            # Perform a "kick": remove a random element to escape.
            # We choose an element that is part of many conflicts? 
            # Or just random?
            # Random is simple and effective for this size.
            
            # Get list of elements in set
            current_elements = [i for i in range(1, N + 1) if in_set[i]]
            if not current_elements:
                break
                
            # Try a few random removals to see if one leads to a better state
            # We will do a small local search after removal
            found_better = False
            for _ in range(10): # Try 10 random kicks
                if time.time() > deadline:
                    break
                u = random.choice(current_elements)
                # Save state
                old_in_set = in_set[:]
                old_mask = mask
                old_size = best_size
                
                # Remove u
                in_set[u] = False
                mask ^= (1 << u)
                
                # Try to add elements
                add_count = 0
                for x in range(1, N + 1):
                    if not in_set[x]:
                        if can_add(x, in_set):
                            in_set[x] = True
                            mask |= (1 << x)
                            add_count += 1
                            
                # If we added more than we removed (1), we improved
                if add_count > 1:
                    best_mask = mask
                    best_size = bin(mask).count('1')
                    found_better = True
                    break
                else:
                    # Revert
                    in_set = old_in_set
                    mask = old_mask
                    best_size = old_size
            
            if not found_better:
                # If we couldn't find an improvement with 10 random kicks,
                # we might be stuck in a deep local minimum.
                # We can try a more systematic kick: remove an element with high degree in conflict graph?
                # Or just accept we are stuck and break?
                # Given the time budget, breaking and restarting might be better.
                # But we are in local_search. Let's try one more systematic approach:
                # Remove the element that blocks the most additions?
                # This is complex. Let's just break to allow outer loop to restart if needed.
                # However, the outer loop is just one call to local_search?
                # No, we can call local_search multiple times with different inits.
                
                # To avoid infinite loop, we break if we can't improve.
                break
                
        return best_mask

    best_mask = 0
    best_size = 0
    
    # Strategy 1: Natural Order
    mask, in_set = greedy_build(range(1, N + 1))
    m = local_search(mask, in_set)
    if bin(m).count('1') > best_size:
        best_mask = m
        best_size = bin(m).count('1')
        
    # Strategy 2: Reverse Order
    if time.time() < deadline:
        mask, in_set = greedy_build(range(N, 0, -1))
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')

    # Strategy 3: Random Restarts
    while time.time() < deadline - 5: # Keep 5s buffer
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        mask, in_set = greedy_build(random_order)
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')
            
    # Convert mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result
