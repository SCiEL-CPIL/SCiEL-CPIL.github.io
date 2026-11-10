import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute popcount for all masks
    popcount = [0] * num_masks
    for i in range(1, num_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Comparability check: a is subset of b iff (a & ~b) == 0
    def comparable(a, b):
        return (a & ~b) == 0 or (b & ~a) == 0
    
    def calc_score(masks):
        s = 0
        for i in range(m):
            for j in range(i + 1, m):
                if comparable(masks[i], masks[j]):
                    s += 1
        return s
    
    # Build a maximal chain: empty set, then add one element at a time up to full set
    # This gives 13 elements all mutually comparable
    def build_chain():
        chain = [0]
        cur = 0
        for i in range(n):
            cur |= (1 << i)
            chain.append(cur)
        return chain
    
    # Greedy init: start with a chain, then greedily add most-comparable elements
    def greedy_init(seed):
        random.seed(seed)
        current = build_chain()
        used = set(current)
        
        for _ in range(m - len(current)):
            best_mask = -1
            best_score = -1
            for mask in range(num_masks):
                if mask in used:
                    continue
                score = 0
                for cm in current:
                    if comparable(mask, cm):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_mask != -1:
                current.append(best_mask)
                used.add(best_mask)
            else:
                mask = random.randint(0, full_mask)
                while mask in used:
                    mask = random.randint(0, full_mask)
                current.append(mask)
                used.add(mask)
        
        return current
    
    # Try several greedy initializations and pick the best
    best_init = None
    best_init_score = -1
    for seed in [42, 123, 456, 789, 1024, 2048, 7, 13, 99, 314]:
        init = greedy_init(seed)
        score = calc_score(init)
        if score > best_init_score:
            best_init_score = score
            best_init = init
    
    best_score = best_init_score
    best_solution = list(best_init)
    
    # Simulated Annealing
    random.seed(42)
    
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 20.0
        cooling_rate = 0.9995
        min_T = 0.001
        no_improve_steps = 0
        max_no_improve = 20000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask
            r = random.random()
            if r < 0.35:
                # Flip one bit
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            elif r < 0.55:
                # Flip two bits
                all_bits = list(range(n))
                random.shuffle(all_bits)
                new_mask = old_mask ^ (1 << all_bits[0]) ^ (1 << all_bits[1])
            elif r < 0.75:
                # Flip three bits
                all_bits = list(range(n))
                random.shuffle(all_bits)
                new_mask = old_mask
                for i in range(3):
                    new_mask ^= (1 << all_bits[i])
            elif r < 0.85:
                # Move along chain: add or remove one element from a random position
                pc = popcount[old_mask]
                if pc > 0 and random.random() < 0.5:
                    # Remove a random set bit
                    bits = [i for i in range(n) if (old_mask >> i) & 1]
                    bit = random.choice(bits)
                    new_mask = old_mask ^ (1 << bit)
                elif pc < n:
                    # Add a random unset bit
                    bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                    bit = random.choice(bits)
                    new_mask = old_mask | (1 << bit)
                else:
                    new_mask = random.randint(0, full_mask)
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                other = current_masks[j]
                old_comp = 1 if comparable(old_mask, other) else 0
                new_comp = 1 if comparable(new_mask, other) else 0
                delta += new_comp - old_comp
            
            # Acceptance
            if delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                no_improve_steps = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    used_masks.remove(old_mask)
                    used_masks.add(new_mask)
                    current_masks[idx] = new_mask
                    current_score += delta
                    no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break

        restarts += 1
    
    # Final polish: try replacing each element with a better one
    polish_rounds = 3
    for round_num in range(polish_rounds):
        improved = False
        for idx in range(m):
            old_mask = best_solution[idx]
            best_replace = -1
            best_delta = 0
            
            old_contrib = 0
            for j in range(m):
                if j == idx:
                    continue
                if comparable(old_mask, best_solution[j]):
                    old_contrib += 1
            
            for mask in range(num_masks):
                if mask == old_mask or mask in best_solution:
                    continue
                
                new_contrib = 0
                for j in range(m):
                    if j == idx:
                        continue
                    if comparable(mask, best_solution[j]):
                        new_contrib += 1
                
                delta = new_contrib - old_contrib
                if delta > best_delta:
                    best_delta = delta
                    best_replace = mask
            
            if best_replace != -1 and best_delta > 0:
                best_solution[idx] = best_replace
                best_score += best_delta
                improved = True
        
        if not improved:
            break
    
    # Convert to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
