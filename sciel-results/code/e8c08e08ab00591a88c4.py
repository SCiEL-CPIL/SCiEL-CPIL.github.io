import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 290

    def time_left():
        return deadline - time.time()

    # 1. Build Paley Graph P(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
                edges.append((i, j))
    
    n = q

    # Precompute K4s and Ind4s
    k4_masks = []
    ind4_masks = []
    
    for i in range(n):
        for j in range(i+1, n):
            if not (adj[i] & (1 << j)): continue
            for k in range(j+1, n):
                if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                for l in range(k+1, n):
                    if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                        k4_masks.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
                    if not (adj[i] & (1 << l)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                        ind4_masks.append((1 << i) | (1 << j) | (1 << k) | (1 << l))
    
    # Build membership lists
    k4_containing = [[] for _ in range(n)]
    ind4_containing = [[] for _ in range(n)]
    
    for idx, mask in enumerate(k4_masks):
        for b in range(n):
            if mask & (1 << b):
                k4_containing[b].append(idx)
                
    for idx, mask in enumerate(ind4_masks):
        for b in range(n):
            if mask & (1 << b):
                ind4_containing[b].append(idx)

    def count_violations(mask):
        k4_viol = 0
        for k4m in k4_masks:
            if (mask & k4m) == k4m:
                k4_viol += 1
        ind4_viol = 0
        for i4m in ind4_masks:
            if (mask & i4m) == 0:
                ind4_viol += 1
        return k4_viol + ind4_viol

    def soft_objective(mask):
        """Compute a smooth surrogate objective using log-sum-exp style penalty."""
        k4_pen = 0.0
        for k4m in k4_masks:
            # Count how many bits of k4m are in mask
            common = mask & k4m
            # Number of missing bits from k4m
            missing = bin(k4m ^ common).count('1')
            # Penalty grows as we get closer to completing the K4
            if missing == 0:
                k4_pen += 100.0
            elif missing == 1:
                k4_pen += 10.0
            elif missing == 2:
                k4_pen += 1.0
            
        ind4_pen = 0.0
        for i4m in ind4_masks:
            # Count how many bits of i4m are NOT in mask (i.e., mask has none of i4m)
            common = mask & i4m
            # Number of bits in i4m that are present in mask
            present = bin(common).count('1')
            # Penalty grows as we get closer to having no connection to i4m
            if present == 0:
                ind4_pen += 100.0
            elif present == 1:
                ind4_pen += 10.0
            elif present == 2:
                ind4_pen += 1.0
        
        return k4_pen + ind4_pen

    def try_add_vertex(seed_offset=0):
        if time_left() < 10:
            return None
            
        random.seed(42 + seed_offset)
        mask = random.getrandbits(n)
        
        # Check if already valid
        if count_violations(mask) == 0:
            return mask

        steps = 0
        max_steps = 30000
        no_improve_count = 0
        best_total = count_violations(mask)
        best_obj = soft_objective(mask)
        best_mask = mask
        
        while steps < max_steps:
            if time_left() < 10:
                if count_violations(best_mask) == 0:
                    return best_mask
                return None
                
            best_delta = 0
            best_bit = -1
            best_is_add = False
            best_new_obj = best_obj
            
            # Try adding bits
            for b in range(n):
                if mask & (1 << b): continue
                
                d_k4 = 0
                for idx in k4_containing[b]:
                    k4m = k4_masks[idx]
                    other_bits = k4m & ~(1 << b)
                    if (mask & other_bits) == other_bits:
                        d_k4 += 1
                        
                d_ind4 = 0
                for idx in ind4_containing[b]:
                    i4m = ind4_masks[idx]
                    other_bits = i4m & ~(1 << b)
                    if (mask & other_bits) == 0:
                        d_ind4 -= 1
                        
                delta = d_k4 + d_ind4
                new_mask = mask | (1 << b)
                new_obj = soft_objective(new_mask)
                
                # Prefer moves that reduce violations, tie-break by soft objective
                if delta < best_delta or (delta == best_delta and new_obj < best_new_obj):
                    best_delta = delta
                    best_bit = b
                    best_is_add = True
                    best_new_obj = new_obj
                    
            # Try removing bits
            for b in range(n):
                if not (mask & (1 << b)): continue
                
                d_k4 = 0
                for idx in k4_containing[b]:
                    k4m = k4_masks[idx]
                    other_bits = k4m & ~(1 << b)
                    if (mask & other_bits) == other_bits:
                        d_k4 -= 1
                
                d_ind4 = 0
                for idx in ind4_containing[b]:
                    i4m = ind4_masks[idx]
                    other_bits = i4m & ~(1 << b)
                    if (mask & other_bits) == 0:
                        d_ind4 += 1
                        
                delta = d_k4 + d_ind4
                new_mask = mask & ~(1 << b)
                new_obj = soft_objective(new_mask)
                
                if delta < best_delta or (delta == best_delta and new_obj < best_new_obj):
                    best_delta = delta
                    best_bit = b
                    best_is_add = False
                    best_new_obj = new_obj
            
            if best_bit != -1 and best_delta < 0:
                if best_is_add:
                    mask |= (1 << best_bit)
                else:
                    mask &= ~(1 << best_bit)
                
                total_viol = count_violations(mask)
                current_obj = soft_objective(mask)
                if total_viol < best_total or (total_viol == best_total and current_obj < best_obj):
                    best_total = total_viol
                    best_obj = current_obj
                    best_mask = mask
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if total_viol == 0:
                    return mask
            else:
                no_improve_count += 1
                if no_improve_count > 30:
                    # Random restart with different seed
                    random.seed(1000 + steps)
                    mask = random.getrandbits(n)
                    total_viol = count_violations(mask)
                    current_obj = soft_objective(mask)
                    if total_viol < best_total or (total_viol == best_total and current_obj < best_obj):
                        best_total = total_viol
                        best_obj = current_obj
                        best_mask = mask
                    no_improve_count = 0
                    if total_viol == 0:
                        return mask

            steps += 1
             
        if count_violations(best_mask) == 0:
            return best_mask
        return None

    added = 0
    while added < 10:
        if time_left() < 30:
            break
            
        # Try multiple seeds to find a valid mask
        mask = None
        for attempt in range(5):
            if time_left() < 20:
                break
            mask = try_add_vertex(attempt)
            if mask is not None:
                break
        
        if mask is None:
            break
        
        new_adj = [0] * (n + 1)
        for i in range(n):
            new_adj[i] = adj[i]
            if mask & (1 << i):
                new_adj[i] |= 1 << n
                new_adj[n] |= 1 << i
                edges.append((i, n))
        
        adj = new_adj
        old_n = n
        n += 1
        added += 1
        
        # Extend membership lists
        k4_containing.append([])
        ind4_containing.append([])
        
        # Update K4s and Ind4s involving the new vertex
        new_v = n - 1
        for i in range(old_n):
            for j in range(i+1, old_n):
                for k in range(j+1, old_n):
                    # Check if i,j,k form a triangle
                    is_triangle = (adj[i] & (1 << j)) and (adj[i] & (1 << k)) and (adj[j] & (1 << k))
                    is_ind_triangle = not (adj[i] & (1 << j)) and not (adj[i] & (1 << k)) and not (adj[j] & (1 << k))
                    
                    if is_triangle:
                        if (adj[i] & (1 << new_v)) and (adj[j] & (1 << new_v)) and (adj[k] & (1 << new_v)):
                            new_mask = (1 << i) | (1 << j) | (1 << k) | (1 << new_v)
                            k4_masks.append(new_mask)
                            k4_idx = len(k4_masks) - 1
                            k4_containing[i].append(k4_idx)
                            k4_containing[j].append(k4_idx)
                            k4_containing[k].append(k4_idx)
                            k4_containing[new_v].append(k4_idx)
                            
                    if is_ind_triangle:
                        if not (adj[i] & (1 << new_v)) and not (adj[j] & (1 << new_v)) and not (adj[k] & (1 << new_v)):
                            new_mask = (1 << i) | (1 << j) | (1 << k) | (1 << new_v)
                            ind4_masks.append(new_mask)
                            ind4_idx = len(ind4_masks) - 1
                            ind4_containing[i].append(ind4_idx)
                            ind4_containing[j].append(ind4_idx)
                            ind4_containing[k].append(ind4_idx)
                            ind4_containing[new_v].append(ind4_idx)
                            
    return n, edges
