import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
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
    conflict = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd_mat[a][b]
                g_ac = gcd_mat[a][c]
                g_bc = gcd_mat[b][c]
                if g_ab == g_ac and g_ab == g_bc:
                    conflict[a].append((b, c))
                    conflict[b].append((a, c))
                    conflict[c].append((a, b))
                    
    # 3. Helper function to check if x can be added to current set
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

    def degree_greedy():
        """Greedy that picks the element with fewest conflicts among remaining."""
        in_set = [False] * (N + 1)
        mask = 0
        remaining = set(range(1, N + 1))
        
        while remaining:
            # For each remaining element, count how many conflicts it has
            # with the current set (i.e., how many pairs in conflict[x] are both in_set)
            best_x = None
            best_count = float('inf')
            for x in remaining:
                count = 0
                for a, b in conflict[x]:
                    if in_set[a] and in_set[b]:
                        count += 1
                if count < best_count:
                    best_count = count
                    best_x = x
                if count == 0:
                    break  # Can't do better than 0
            if best_x is None:
                break
            in_set[best_x] = True
            mask |= (1 << best_x)
            remaining.discard(best_x)
            
        return mask, in_set

    def local_search(mask, in_set):
        best_mask = mask
        best_size = bin(mask).count('1')
        
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
                        break
            
            if improved:
                continue
                
            # Get list of elements in set
            current_elements = [i for i in range(1, N + 1) if in_set[i]]
            if not current_elements:
                break
            
            # 1-1 kick: remove one random element, try to add
            found_better = False
            for _ in range(15):
                if time.time() > deadline:
                    break
                u = random.choice(current_elements)
                old_in_set = in_set[:]
                old_mask = mask
                old_size = best_size
                
                in_set[u] = False
                mask ^= (1 << u)
                
                add_count = 0
                for x in range(1, N + 1):
                    if not in_set[x]:
                        if can_add(x, in_set):
                            in_set[x] = True
                            mask |= (1 << x)
                            add_count += 1
                            
                if add_count > 1:
                    best_mask = mask
                    best_size = bin(mask).count('1')
                    found_better = True
                    break
                else:
                    in_set = old_in_set
                    mask = old_mask
                    best_size = old_size
            
            if found_better:
                continue
            
            # 2-2 kick: remove two random elements, try to add
            if len(current_elements) >= 2:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    u, v = random.sample(current_elements, 2)
                    old_in_set = in_set[:]
                    old_mask = mask
                    old_size = best_size
                    
                    in_set[u] = False
                    in_set[v] = False
                    mask ^= (1 << u)
                    mask ^= (1 << v)
                    
                    add_count = 0
                    for x in range(1, N + 1):
                        if not in_set[x]:
                            if can_add(x, in_set):
                                in_set[x] = True
                                mask |= (1 << x)
                                add_count += 1
                                
                    if add_count > 2:
                        best_mask = mask
                        best_size = bin(mask).count('1')
                        found_better = True
                        break
                    else:
                        in_set = old_in_set
                        mask = old_mask
                        best_size = old_size
                
                if found_better:
                    continue
            
            # If we couldn't find an improvement, break
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

    # Strategy 3: Degree-based greedy
    if time.time() < deadline:
        mask, in_set = degree_greedy()
        m = local_search(mask, in_set)
        if bin(m).count('1') > best_size:
            best_mask = m
            best_size = bin(m).count('1')

    # Strategy 4: Random Restarts
    while time.time() < deadline - 5:
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
