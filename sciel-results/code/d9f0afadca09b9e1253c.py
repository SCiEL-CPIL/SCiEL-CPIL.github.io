import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 512
    # Precompute compatibility bitsets
    can = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                mask |= (1 << (j - 1))
                continue
            g = gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << (j - 1))
        can[i] = mask

    def is_compatible(x, S_mask):
        return (can[x] & S_mask) == S_mask

    def greedy_construct(order):
        S_mask = 0
        S_list = []
        for x in order:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        return S_list

    def local_search(S_list):
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << (x - 1))
        
        improved = True
        max_iter = 200
        iter_count = 0
        
        while improved and iter_count < max_iter:
            iter_count += 1
            improved = False
            
            # 1. Try to add any x (prioritize those with highest degree in complement)
            best_add = -1
            best_add_deg = -1
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        # Count how many other non-S elements are compatible with x and S
                        deg = 0
                        for y in range(1, N + 1):
                            if not (S_mask & (1 << (y - 1))) and y != x:
                                if (can[x] >> (y - 1)) & 1:
                                    deg += 1
                        if deg > best_add_deg:
                            best_add_deg = deg
                            best_add = x
            if best_add > 0:
                S_mask |= (1 << (best_add - 1))
                S_list.append(best_add)
                improved = True
                continue

            # 2. Try to swap (remove 1, add 1)
            swapped = False
            u_indices = list(range(len(S_list)))
            random.shuffle(u_indices)
            for idx in u_indices:
                u = S_list[idx]
                S_mask_without_u = S_mask ^ (1 << (u - 1))
                best_v = -1
                best_v_deg = -1
                for v in range(1, N + 1):
                    if not (S_mask & (1 << (v - 1))):
                        if is_compatible(v, S_mask_without_u):
                            deg = 0
                            for y in range(1, N + 1):
                                if not (S_mask_without_u & (1 << (y - 1))) and y != v:
                                    if (can[v] >> (y - 1)) & 1:
                                        deg += 1
                            if deg > best_v_deg:
                                best_v_deg = deg
                                best_v = v
                if best_v > 0:
                    S_mask = S_mask_without_u | (1 << (best_v - 1))
                    S_list[idx] = best_v
                    swapped = True
                    break
            
            if swapped:
                improved = True
                continue

            # 3. Try to remove 1, add 2 (increase size by 1)
            found_triple = False
            u_indices = list(range(len(S_list)))
            random.shuffle(u_indices)
            
            for idx in u_indices:
                u = S_list[idx]
                S_mask_without_u = S_mask ^ (1 << (u - 1))
                
                candidates = []
                for v in range(1, N + 1):
                    if not (S_mask & (1 << (v - 1))):
                        if is_compatible(v, S_mask_without_u):
                            candidates.append(v)
                
                if len(candidates) < 2:
                    continue
                
                # Sort candidates by degree to find good pairs faster
                candidates.sort(key=lambda x: -bin(can[x]).count('1'))
                
                for i in range(len(candidates)):
                    v = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        w = candidates[j]
                        if (can[v] >> (w - 1)) & 1:
                            S_mask = S_mask_without_u | (1 << (v - 1)) | (1 << (w - 1))
                            S_list.pop(idx)
                            S_list.append(v)
                            S_list.append(w)
                            found_triple = True
                            break
                    if found_triple:
                        break
                if found_triple:
                    break
            
            if found_triple:
                improved = True
                continue

            # 4. Try to remove 2, add 3 (increase size by 1, larger jump)
            found_quad = False
            if len(S_list) >= 2:
                u_indices = list(range(len(S_list)))
                random.shuffle(u_indices)
                
                for i1 in u_indices:
                    if time.time() > deadline:
                        break
                    u1 = S_list[i1]
                    remaining_indices = [i for i in u_indices if i != i1]
                    for i2 in remaining_indices:
                        u2 = S_list[i2]
                        S_mask_without_u1u2 = S_mask ^ (1 << (u1 - 1)) ^ (1 << (u2 - 1))
                        
                        candidates = []
                        for v in range(1, N + 1):
                            if not (S_mask & (1 << (v - 1))):
                                if is_compatible(v, S_mask_without_u1u2):
                                    candidates.append(v)
                        
                        if len(candidates) < 3:
                            continue
                        
                        candidates.sort(key=lambda x: -bin(can[x]).count('1'))
                        
                        # Find 3 mutually compatible candidates
                        found = False
                        for a in range(len(candidates)):
                            ca = candidates[a]
                            for b in range(a + 1, len(candidates)):
                                cb = candidates[b]
                                if not ((can[ca] >> (cb - 1)) & 1):
                                    continue
                                for c in range(b + 1, len(candidates)):
                                    cc = candidates[c]
                                    if ((can[ca] >> (cc - 1)) & 1) and ((can[cb] >> (cc - 1)) & 1):
                                        # Valid: remove u1,u2 add ca,cb,cc
                                        S_mask = S_mask_without_u1u2 | (1 << (ca - 1)) | (1 << (cb - 1)) | (1 << (cc - 1))
                                        # Remove u1 and u2 from list
                                        # Need to be careful with indices
                                        if i1 < i2:
                                            S_list.pop(i2)
                                            S_list.pop(i1)
                                        else:
                                            S_list.pop(i1)
                                            S_list.pop(i2)
                                        S_list.append(ca)
                                        S_list.append(cb)
                                        S_list.append(cc)
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            found_quad = True
                            break
                    if found_quad:
                        break
            
            if found_quad:
                improved = True
                continue

        return S_list

    best_size = 0
    best_set = []
    
    # 1. Structural seeds: Divisors of highly composite numbers
    for M in [504, 420, 360, 480, 336, 240, 210, 180, 168, 120, 108, 96, 72, 60, 48, 36, 24, 12, 6, 4, 2, 1]:
        if time.time() > deadline:
            break
        divs = [i for i in range(1, N + 1) if M % i == 0]
        s = local_search(divs)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # 2. Greedy with different orderings
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(can[i]).count('1')
    
    order_asc = sorted(range(1, N + 1), key=lambda x: degrees[x])
    order_desc = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    
    if time.time() < deadline:
        s = greedy_construct(order_asc)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    if time.time() < deadline:
        s = greedy_construct(order_desc)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # 3. Random restarts
    random.seed(42)
    while time.time() < deadline:
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_construct(order)
        s = local_search(s)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
        
        if best_set and random.random() < 0.1:
             s_perturbed = best_set[:]
             if s_perturbed:
                 idx = random.randint(0, len(s_perturbed) - 1)
                 s_perturbed.pop(idx)
                 s = local_search(s_perturbed)
                 if len(s) > best_size:
                     best_size = len(s)
                     best_set = s[:]

    return best_set
