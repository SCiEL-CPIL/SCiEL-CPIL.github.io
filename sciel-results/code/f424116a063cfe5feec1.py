import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    total_masks = 1 << n  # 4096
    
    # Precompute comparability using a flat array for speed
    # comp[a * total_masks + b] = 1 if a and b are comparable
    comp = bytearray(total_masks * total_masks)
    for a in range(total_masks):
        base_a = a * total_masks
        for b in range(total_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[base_a + b] = 1
    
    def calc_score(masks):
        s = 0
        for i in range(m):
            base_i = masks[i] * total_masks
            for j in range(i + 1, m):
                s += comp[base_i + masks[j]]
        return s
    
    def greedy_init(seed_val):
        """Better initialization: start with a chain, then add sets greedily."""
        rng = random.Random(seed_val)
        
        # Create a random chain
        elem_order = list(range(n))
        rng.shuffle(elem_order)
        chain = [0]
        cur = 0
        for e in elem_order:
            cur |= (1 << e)
            chain.append(cur)
        
        current_masks = list(chain)
        used_masks = set(current_masks)
        
        # Greedily add remaining sets
        for _ in range(m - len(current_masks)):
            best_mask = -1
            best_score = -1
            for mask in range(total_masks):
                if mask in used_masks:
                    continue
                base_mask = mask * total_masks
                score = 0
                for cm in current_masks:
                    score += comp[base_mask + cm]
                if score > best_score:
                    best_score = score
                    best_mask = mask
            current_masks.append(best_mask)
            used_masks.add(best_mask)
        
        return current_masks
    
    def local_search(current_masks, used_masks, cur_score, time_limit, seed_val):
        """Local search with efficient delta calculation."""
        rng = random.Random(seed_val)
        stuck = 0
        
        while time.time() < time_limit:
            # Pick random index
            idx = rng.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate neighbor
            new_mask = old_mask
            op = rng.random()
            if op < 0.35:
                zeros = [i for i in range(n) if not ((old_mask >> i) & 1)]
                if zeros:
                    new_mask = old_mask | (1 << rng.choice(zeros))
            elif op < 0.70:
                ones = [i for i in range(n) if (old_mask >> i) & 1]
                if ones:
                    new_mask = old_mask ^ (1 << rng.choice(ones))
            elif op < 0.85:
                ones = [i for i in range(n) if (old_mask >> i) & 1]
                zeros = [i for i in range(n) if not ((old_mask >> i) & 1)]
                if ones and zeros:
                    bo = rng.choice(ones)
                    bz = rng.choice(zeros)
                    new_mask = (old_mask ^ (1 << bo)) | (1 << bz)
            else:
                new_mask = rng.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue
            
            # Delta calculation
            base_old = old_mask * total_masks
            base_new = new_mask * total_masks
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mj = current_masks[j]
                delta += comp[base_new + mj] - comp[base_old + mj]
            
            if delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                cur_score += delta
                stuck = 0
            else:
                stuck += 1
                if stuck > 150:
                    # Kick
                    for _ in range(rng.randint(2, 3)):
                        idx2 = rng.randint(0, m - 1)
                        old2 = current_masks[idx2]
                        while True:
                            new2 = rng.randint(0, full_mask)
                            if new2 not in used_masks:
                                break
                        current_masks[idx2] = new2
                    used_masks = set(current_masks)
                    cur_score = calc_score(current_masks)
                    stuck = 0
        
        return current_masks, used_masks, cur_score
    
    def polish(current_masks, used_masks, cur_score, time_limit):
        """Exhaustive single-bit polish."""
        polished = True
        while polished:
            if time.time() > time_limit:
                break
            polished = False
            for idx in range(m):
                old_mask = current_masks[idx]
                base_old = old_mask * total_masks
                for bit in range(n):
                    if not ((old_mask >> bit) & 1):
                        new_mask = old_mask | (1 << bit)
                        if new_mask not in used_masks:
                            base_new = new_mask * total_masks
                            delta = 0
                            for j in range(m):
                                if j == idx:
                                    continue
                                mj = current_masks[j]
                                delta += comp[base_new + mj] - comp[base_old + mj]
                            if delta > 0:
                                used_masks.remove(old_mask)
                                used_masks.add(new_mask)
                                current_masks[idx] = new_mask
                                cur_score += delta
                                polished = True
                                base_old = new_mask * total_masks  # Update for next iteration
                    if (old_mask >> bit) & 1:
                        new_mask = old_mask ^ (1 << bit)
                        if new_mask not in used_masks:
                            base_new = new_mask * total_masks
                            delta = 0
                            for j in range(m):
                                if j == idx:
                                    continue
                                mj = current_masks[j]
                                delta += comp[base_new + mj] - comp[base_old + mj]
                            if delta > 0:
                                used_masks.remove(old_mask)
                                used_masks.add(new_mask)
                                current_masks[idx] = new_mask
                                cur_score += delta
                                polished = True
                                base_old = new_mask * total_masks
        return current_masks, used_masks, cur_score
    
    random.seed(42)
    
    best_global_score = -1
    best_global_solution = None
    
    num_restarts = 12
    
    for r in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        current_masks = greedy_init(42 + r)
        used_masks = set(current_masks)
        cur_score = calc_score(current_masks)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
        
        # Local search with time budget
        ls_limit = deadline - 10.0
        current_masks, used_masks, cur_score = local_search(current_masks, used_masks, cur_score, ls_limit, 1000 + r)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
        
        # Polish
        polish_limit = deadline - 2.0
        current_masks, used_masks, cur_score = polish(current_masks, used_masks, cur_score, polish_limit)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
    
    # Convert to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
