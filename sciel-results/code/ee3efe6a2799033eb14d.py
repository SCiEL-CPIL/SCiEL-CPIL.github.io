import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try n=67
    n = 67
    
    # Precomputation
    # 1. Edge indices
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
    
    # 2. All 4-sets
    all_sets = list(itertools.combinations(range(n), 4))
    N4 = len(all_sets)
    
    # 3. sets_with
    sets_with = [[] for _ in range(n)]
    for s_idx, s in enumerate(all_sets):
        for v in s:
            sets_with[v].append(s_idx)
    
    def random_tournament(n):
        return [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        
    def orient_to_masks(orient, n):
        In = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orient[idx] == 0:
                    In[j] |= (1 << i)
                else:
                    In[i] |= (1 << j)
                idx += 1
        return In

    def masks_to_orient(In, n):
        orient = []
        for i in range(n):
            for j in range(i + 1, n):
                if (In[j] >> i) & 1:
                    orient.append(1)
                else:
                    orient.append(0)
        return orient

    best_solution = None
    best_cost = float('inf')
    
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        orient = random_tournament(n)
        In = orient_to_masks(orient, n)
        
        is_bad = [False] * N4
        cost = 0
        for s_idx, s in enumerate(all_sets):
            a, b, c, d = s
            if (In[a] & In[b] & In[c] & In[d]) == 0:
                is_bad[s_idx] = True
                cost += 1
        
        if cost == 0:
            best_solution = orient
            break
        
        if cost < best_cost:
            best_cost = cost
            best_solution = orient[:]
            
        T = 1.0
        while T > 0.01 and time.time() < deadline - 5:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            
            e_idx = edge_idx[i][j]
            
            # Current state
            # If orient[e_idx] == 0, i -> j. In[j] has bit i.
            # If orient[e_idx] == 1, j -> i. In[i] has bit j.
            
            # Flip
            # New state is opposite.
            
            # Affected sets
            # Sets with i not j
            # Sets with j not i
            # Sets with both
            
            # We will calculate delta cost
            delta_cost = 0
            changes = []
            
            # Helper to check and update
            # We need to simulate the new In masks
            In_i_new = In[i]
            In_j_new = In[j]
            
            if orient[e_idx] == 0:
                # i -> j becomes j -> i
                # In[i] gains j
                # In[j] loses i
                In_i_new |= (1 << j)
                In_j_new &= ~(1 << i)
            else:
                # j -> i becomes i -> j
                # In[j] gains i
                # In[i] loses j
                In_j_new |= (1 << i)
                In_i_new &= ~(1 << j)
            
            # Process sets with i
            for s_idx in sets_with[i]:
                s = all_sets[s_idx]
                # Check if j is in s
                if j in s:
                    # Both i and j in s
                    # New mask = In[a] & In[b] & In[c] & In[d] with new In for i,j
                    # We need to map vertices to new masks
                    # s is a tuple of 4
                    mask = 0
                    for v in s:
                        if v == i:
                            mask &= In_i_new
                        elif v == j:
                            mask &= In_j_new
                        else:
                            mask &= In[v]
                        # Optimization: if mask becomes 0, break?
                        # But we need the final mask to check badness.
                    # Actually, just compute fully.
                    
                    new_bad = (mask == 0)
                    old_bad = is_bad[s_idx]
                    if new_bad != old_bad:
                        delta_cost += 1 if new_bad else -1
                        changes.append((s_idx, old_bad, new_bad))
                else:
                    # i in s, j not in s
                    # Only In[i] changes
                    mask = In_i_new
                    for v in s:
                        if v != i:
                            mask &= In[v]
                    
                    new_bad = (mask == 0)
                    old_bad = is_bad[s_idx]
                    if new_bad != old_bad:
                        delta_cost += 1 if new_bad else -1
                        changes.append((s_idx, old_bad, new_bad))
            
            # Process sets with j (but not i, to avoid double counting)
            for s_idx in sets_with[j]:
                s = all_sets[s_idx]
                if i in s:
                    continue # Already processed
                else:
                    # j in s, i not in s
                    mask = In_j_new
                    for v in s:
                        if v != j:
                            mask &= In[v]
                    
                    new_bad = (mask == 0)
                    old_bad = is_bad[s_idx]
                    if new_bad != old_bad:
                        delta_cost += 1 if new_bad else -1
                        changes.append((s_idx, old_bad, new_bad))
            
            new_cost = cost + delta_cost
            
            accept = False
            if new_cost < cost:
                accept = True
            else:
                if delta_cost > 0:
                    # Metropolis
                    if random.random() < math.exp(-delta_cost / T):
                        accept = True
            
            if accept:
                # Apply changes
                for s_idx, old_bad, new_bad in changes:
                    is_bad[s_idx] = new_bad
                cost = new_cost
                In[i] = In_i_new
                In[j] = In_j_new
                orient[e_idx] = 1 - orient[e_idx]
                
                if cost == 0:
                    best_solution = orient
                    break
            # If not accepted, do nothing (is_bad, In, orient remain old)
            
            T *= 0.995
            
        if best_solution:
            break

    if best_solution:
        return {'n': n, 'orientation': best_solution}
    
    # Fallback
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
