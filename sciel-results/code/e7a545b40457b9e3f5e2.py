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
    
    # Precompute comparability matrix for speed
    # comp[a][b] = 1 if a and b are comparable (one is strict subset of other)
    comp = [0] * (num_masks * num_masks)
    for a in range(num_masks):
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[a * num_masks + b] = 1
    
    def is_comparable(a, b):
        return comp[a * num_masks + b]
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if comp[masks[i] * num_masks + masks[j]]:
                    s += 1
        return s
    
    # 1. Better Greedy Initialization - try multiple seeds
    def greedy_init(seed):
        random.seed(seed)
        current = [0, full_mask]
        used = {0, full_mask}
        
        for _ in range(m - 2):
            best_mask = -1
            best_score = -1
            for mask in range(num_masks):
                if mask in used:
                    continue
                score = 0
                base = mask * num_masks
                for cm in current:
                    if comp[base + cm]:
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
    for seed in [42, 123, 456, 789, 1024, 2048]:
        init = greedy_init(seed)
        score = calc_score(init)
        if score > best_init_score:
            best_init_score = score
            best_init = init
    
    current_masks = list(best_init)
    best_score = best_init_score
    best_solution = list(current_masks)
    
    # 2. Simulated Annealing with precomputed comparability
    random.seed(42)
    
    restarts = 0
    max_restarts = 30
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 15.0
        cooling_rate = 0.9997
        min_T = 0.005
        no_improve_steps = 0
        max_no_improve = 15000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask with more diverse moves
            r = random.random()
            if r < 0.4:
                # Flip one bit (small perturbation)
                if old_mask == 0:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask | (1 << bit)
                elif old_mask == full_mask:
                    bit = random.randint(0, n - 1)
                    new_mask = old_mask ^ (1 << bit)
                else:
                    if random.random() < 0.5:
                        # Remove a bit
                        bits = [i for i in range(n) if (old_mask >> i) & 1]
                        bit = random.choice(bits)
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        # Add a bit
                        bits = [i for i in range(n) if not ((old_mask >> i) & 1)]
                        bit = random.choice(bits)
                        new_mask = old_mask | (1 << bit)
            elif r < 0.65:
                # Flip two bits
                all_bits = list(range(n))
                random.shuffle(all_bits)
                bit1 = all_bits[0]
                bit2 = all_bits[1]
                new_mask = old_mask ^ (1 << bit1) ^ (1 << bit2)
            elif r < 0.85:
                # Flip three bits
                all_bits = list(range(n))
                random.shuffle(all_bits)
                new_mask = old_mask
                for i in range(3):
                    new_mask ^= (1 << all_bits[i])
            else:
                # Random mask or swap with another element
                r2 = random.random()
                if r2 < 0.5:
                    new_mask = random.randint(0, full_mask)
                else:
                    # Swap with another random element
                    idx2 = random.randint(0, m - 1)
                    if idx2 == idx:
                        idx2 = (idx2 + 1) % m
                    new_mask = current_masks[idx2]
                    if new_mask in used_masks and new_mask != old_mask:
                        # This would create duplicate, skip
                        continue
                    # We'll handle the swap specially
                    # For now, just use random
                    new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta using precomputed matrix
            delta = 0
            base_old = old_mask * num_masks
            base_new = new_mask * num_masks
            for j in range(m):
                if j == idx:
                    continue
                other = current_masks[j]
                old_comp = comp[base_old + other]
                new_comp = comp[base_new + other]
                delta += new_comp - old_comp
            
            # Acceptance probability
            if delta > 0:
                # Accept
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                no_improve_steps = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                # Reject or accept with probability exp(delta/T)
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
    
    # 3. Final Greedy Polish Pass
    # Try to improve by replacing each element with a better one
    polish_rounds = 5
    for round_num in range(polish_rounds):
        improved = False
        for idx in range(m):
            old_mask = best_solution[idx]
            best_replace = -1
            best_delta = 0
            
            # Calculate current contribution of old_mask
            old_contrib = 0
            base_old = old_mask * num_masks
            for j in range(m):
                if j == idx:
                    continue
                if comp[base_old + best_solution[j]]:
                    old_contrib += 1
            
            # Try to find a better replacement
            for mask in range(num_masks):
                if mask == old_mask or mask in best_solution:
                    continue
                
                new_contrib = 0
                base_new = mask * num_masks
                for j in range(m):
                    if j == idx:
                        continue
                    if comp[base_new + best_solution[j]]:
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
