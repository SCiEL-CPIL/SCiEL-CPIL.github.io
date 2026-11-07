import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    idx = {v: i for i, v in enumerate(numbers)}
    
    # Precompute C[u][v] as a bitmask of x such that x conflicts with pair (u,v)
    # Conflict if x | u*v or u | x*v or v | x*u
    # u, v are values. We use indices 0..n-1.
    # C is a 2D list of ints (bitmasks)
    C = [[0] * n for _ in range(n)]
    
    for i in range(n):
        ui = numbers[i]
        for j in range(i + 1, n):
            uj = numbers[j]
            prod = ui * uj
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                uk = numbers[k]
                # Check conditions
                if prod % uk == 0 or (uk * uj) % ui == 0 or (ui * uk) % uj == 0:
                    mask |= (1 << k)
            C[i][j] = mask
            C[j][i] = mask

    # Greedy Initialization
    # Sort by "popularity" (number of conflicts) or just random/greedy
    # Let's try greedy: pick element with fewest conflicts with current set
    S = set()
    remaining = set(range(n))
    
    # To make it faster, we can just pick a random valid set or greedy by value
    # Greedy by value (largest first) is often good for divisibility problems
    for v in sorted(numbers, reverse=True):
        i = idx[v]
        if i in remaining:
            # Check if i can be added to S
            valid = True
            for a in S:
                for b in S:
                    if a == b: continue
                    if (C[a][b] >> i) & 1:
                        valid = False
                        break
                if not valid: break
            if valid:
                S.add(i)
                remaining.remove(i)
    
    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    # We will iterate through potential swaps
    # State: S (set of indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Build list of elements in S and not in S
        S_list = list(S)
        not_S_list = list(remaining)
        
        # 1. Try to add an element (size + 1)
        added = False
        for x in not_S_list:
            if time.time() > deadline: break
            # Check if x is compatible with S
            conflict_pairs = []
            for ai in range(len(S_list)):
                a = S_list[ai]
                for bi in range(ai + 1, len(S_list)):
                    b = S_list[bi]
                    if (C[a][b] >> x) & 1:
                        conflict_pairs.append((a, b))
            
            if not conflict_pairs:
                # Can add x
                S.add(x)
                remaining.remove(x)
                S_list.append(x)
                improved = True
                added = True
                break # Restart loop to update lists
        
        if added:
            continue
            
        # 2. Try 1-swap (remove y, add x)
        # For each x not in S, find conflicts.
        # If conflicts are covered by removing a single y, do it.
        for x in not_S_list:
            if time.time() > deadline: break
            
            conflict_pairs = []
            for ai in range(len(S_list)):
                a = S_list[ai]
                for bi in range(ai + 1, len(S_list)):
                    b = S_list[bi]
                    if (C[a][b] >> x) & 1:
                        conflict_pairs.append((a, b))
            
            if not conflict_pairs:
                continue # Should have been caught above, but just in case
            
            # Check if there is a y in S such that all conflict_pairs contain y
            # This means y is in the intersection of all pairs (as a vertex)
            # i.e., y is a vertex common to all edges in conflict_pairs
            # If conflict_pairs is empty, we handled it.
            # If conflict_pairs is not empty, we need y to be in every pair.
            
            # Optimization: If there are many conflicts, no single y can cover them all
            # unless they all share a common vertex.
            # Let's find the intersection of vertices in all pairs.
            # If intersection has > 1 elements, any of them might work?
            # No, removing y removes all pairs containing y.
            # So we need: for every (a,b) in conflict_pairs, y == a or y == b.
            
            # Let's just iterate y in S and check
            # To speed up, if len(conflict_pairs) > 2, it's unlikely a single y covers all
            # unless they are all (y, something).
            
            for y in S_list:
                if time.time() > deadline: break
                
                # Check if y covers all conflict_pairs
                covers_all = True
                for (a, b) in conflict_pairs:
                    if y != a and y != b:
                        covers_all = False
                        break
                
                if covers_all:
                    # Valid swap
                    S.remove(y)
                    remaining.add(y)
                    S.add(x)
                    remaining.remove(x)
                    # Update S_list? We break and restart anyway.
                    improved = True
                    break
            
            if improved:
                break
        
        # Update best if we found a larger set? 
        # We only add or swap, so size stays same or increases.
        if len(S) > best_size:
            best_S = set(S)
            best_size = len(S)
            
    # Convert back to values
    result = [numbers[i] for i in best_S]
    return sorted(result)
