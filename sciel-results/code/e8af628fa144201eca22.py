import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets for fast delta
    comp_bits = [0] * num_masks
    for a in range(num_masks):
        for b in range(num_masks):
            if a == b:
                continue
            if (a & b) == a or (a & b) == b:
                comp_bits[a] |= (1 << b)
    
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    s += 1
        return s
    
    # Greedy initialization
    def greedy_init():
        selected = []
        used = set()
        first = random.randint(0, full_mask)
        selected.append(first)
        used.add(first)
        
        for _ in range(m - 1):
            best_mask = -1
            best_score = -1
            for candidate in range(num_masks):
                if candidate in used:
                    continue
                score = 0
                for s in selected:
                    if is_comparable(candidate, s):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = candidate
            if best_mask >= 0:
                selected.append(best_mask)
                used.add(best_mask)
            else:
                while True:
                    r = random.randint(0, full_mask)
                    if r not in used:
                        break
                selected.append(r)
                used.add(r)
        return selected
    
    random.seed(42)
    
    best_global_score = -1
    best_global_solution = None
    
    # Greedy init
    greedy_solution = greedy_init()
    greedy_score = calc_score(greedy_solution)
    best_global_score = greedy_score
    best_global_solution = list(greedy_solution)
    
    num_restarts = 8
    
    for r in range(num_restarts):
        if time.time() > deadline - 8.0:
            break
            
        current_masks = random.sample(range(num_masks), m)
        used_masks = set(current_masks)
        cur_score = calc_score(current_masks)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
            
        stuck = 0
        max_steps = 30000
        
        for step in range(max_steps):
            if time.time() > deadline - 4.0:
                break
                
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            new_mask = old_mask
            
            op = random.random()
            if op < 0.3:
                if old_mask != full_mask:
                    zeros = [i for i in range(n) if not ((old_mask >> i) & 1)]
                    if zeros:
                        bit = random.choice(zeros)
                        new_mask = old_mask | (1 << bit)
            elif op < 0.6:
                if old_mask != 0:
                    ones = [i for i in range(n) if (old_mask >> i) & 1]
                    if ones:
                        bit = random.choice(ones)
                        new_mask = old_mask ^ (1 << bit)
            elif op < 0.8:
                ones = [i for i in range(n) if (old_mask >> i) & 1]
                zeros = [i for i in range(n) if not ((old_mask >> i) & 1)]
                if ones and zeros:
                    bo = random.choice(ones)
                    bz = random.choice(zeros)
                    new_mask = (old_mask ^ (1 << bo)) | (1 << bz)
            else:
                new_mask = random.randint(0, full_mask)
                
            if new_mask == old_mask or new_mask in used_masks:
                continue
                
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mask_j = current_masks[j]
                old_comp = 1 if (comp_bits[old_mask] >> mask_j) & 1 else 0
                new_comp = 1 if (comp_bits[new_mask] >> mask_j) & 1 else 0
                delta += new_comp - old_comp
                
            if delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                cur_score += delta
                stuck = 0
                if cur_score > best_global_score:
                    best_global_score = cur_score
                    best_global_solution = list(current_masks)
            else:
                stuck += 1
                if stuck > 80:
                    idx = random.randint(0, m - 1)
                    old_mask = current_masks[idx]
                    while True:
                        new_mask = random.randint(0, full_mask)
                        if new_mask not in used_masks:
                            break
                    used_masks.remove(old_mask)
                    used_masks.add(new_mask)
                    current_masks[idx] = new_mask
                    cur_score = calc_score(current_masks)
                    stuck = 0
    
    # Quick polish pass
    if best_global_solution is not None and time.time() < deadline - 1.5:
        polish_masks = list(best_global_solution)
        polish_used = set(polish_masks)
        polish_score = calc_score(polish_masks)
        
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            improved = False
            for idx in range(m):
                old_mask = polish_masks[idx]
                for bit in range(n):
                    if (old_mask >> bit) & 1:
                        candidate = old_mask ^ (1 << bit)
                    else:
                        candidate = old_mask | (1 << bit)
                    if candidate in polish_used:
                        continue
                    delta = 0
                    for k in range(m):
                        if k == idx:
                            continue
                        mask_k = polish_masks[k]
                        old_comp = 1 if (comp_bits[old_mask] >> mask_k) & 1 else 0
                        new_comp = 1 if (comp_bits[candidate] >> mask_k) & 1 else 0
                        delta += new_comp - old_comp
                    if delta > 0:
                        polish_used.remove(old_mask)
                        polish_used.add(candidate)
                        polish_masks[idx] = candidate
                        polish_score += delta
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if polish_score > best_global_score:
            best_global_score = polish_score
            best_global_solution = polish_masks
    
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
