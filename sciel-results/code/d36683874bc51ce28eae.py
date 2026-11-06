import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute conflicts for each x
    # conflicts[x] is a list of pairs (u, v) such that {u, v, x} is a bad triple.
    # We also precompute for each x and u, the bitmask of v that conflict with x given u is present.
    # conflict_mask[x][u] is a bitmask of v such that lcm(u,v)==lcm(u,x)==lcm(v,x)
    
    # First, find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_triples.append((a, b, c))
                    
    # Build conflict structures
    # For each x, we want to know which pairs (u, v) in the rest of the set would block x.
    # conflict_pairs[x] = set of (u, v) with u < v, u != x, v != x, such that {u, v, x} is bad.
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute conflict bitmasks for faster checking
    # conflict_mask[x] = list of (u, v_mask) where v_mask is bitmask of elements that conflict with x when u is present
    # Actually, let's precompute: for each x, a dict mapping u -> bitmask of v such that (u,v) in conflict_pairs[x]
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))
    
    def can_add(x, mask):
        # Check if x can be added to the set represented by mask
        # x conflicts with pair (u,v) if both u and v are in mask
        cd = conflict_dict[x]
        for u, v_mask in cd.items():
            if mask & (1 << (u - 1)):
                if mask & v_mask:
                    return False
        return True

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def popcount(x):
        return bin(x).count('1')
    
    def local_search(mask, max_iters=50):
        improved = True
        iters = 0
        
        while improved and time.time() < deadline - 2 and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            found_swap = False
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
            for x in s_list:
                if time.time() > deadline - 2: break
                mask_no_x = mask & ~(1 << (x - 1))
                # Try to add two elements
                added_count = 0
                new_mask = mask_no_x
                for y in range(1, n + 1):
                    if y == x: continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    if popcount(new_mask) > popcount(mask):
                        mask = new_mask
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            # This is more expensive but can escape deeper local minima
            if time.time() < deadline - 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                # Try a few random pairs to remove
                random.shuffle(s_list)
                for i in range(min(len(s_list), 20)):
                    if time.time() > deadline - 3: break
                    x = s_list[i]
                    for j in range(i+1, min(len(s_list), i+5)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        # Try to add three elements
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y: continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 5:
                # Remove 2 random elements and try to rebuild
                if len(s_list) > 4:
                    remove_idx = random.sample(range(len(s_list)), 2)
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    # Greedily add back
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        improved = True
                        continue

            if not improved:
                break
                
        return mask

    best_mask = 0
    best_size = 0
    
    # Precompute degrees for sorting
    degrees = [len(conflict_pairs[k]) for k in range(n+1)]
    
    # Run multiple randomized greedy passes
    num_restarts = 1000 # High, but each restart is fast
    
    for i in range(num_restarts):
        if time.time() > deadline - 5: # Leave time for final polish
            break
            
        # Different ordering strategies
        strategy = i % 4
        if strategy == 0:
            # Low degree first
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random()*0.1)
        elif strategy == 1:
            # High degree first
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random()*0.1)
        elif strategy == 2:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
        else:
            # Reverse order
            order = list(range(n, 0, -1))
            random.shuffle(order)
            
        mask = greedy(order)
        mask = local_search(mask)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Final polish with more iterations
    best_mask = local_search(best_mask, max_iters=100)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# Helper for popcount if not using bin (bin is fast enough for 128 bits)
