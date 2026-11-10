import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability matrix: comp[a][b] = 1 if a is strict subset of b or vice versa
    # Use a flat array for speed
    comp = [0] * (num_masks * num_masks)
    # Also precompute popcount for tie-breaking
    popcount = [0] * num_masks
    for mask in range(num_masks):
        popcount[mask] = bin(mask).count('1')
        
    for a in range(num_masks):
        for b in range(num_masks):
            if a != b and ((a & b) == a or (a & b) == b):
                comp[a * num_masks + b] = 1

    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if comp[masks[i] * num_masks + masks[j]]:
                    s += 1
        return s
    
    # 1. Improved Greedy Initialization using Dynamic Degrees
    # degrees[mask] = number of selected masks that are comparable to 'mask'
    degrees = [0] * num_masks
    current_masks = []
    used_masks = set()
    
    # Initial seed: Pick a mask with popcount 6 (middle of lattice)
    # This ensures we start in a "dense" region
    seed_mask = 0
    for mask in range(num_masks):
        if popcount[mask] == 6:
            seed_mask = mask
            break
            
    current_masks.append(seed_mask)
    used_masks.add(seed_mask)
    
    # Update degrees for the first mask
    base = seed_mask * num_masks
    for k in range(num_masks):
        if comp[base + k]:
            degrees[k] += 1
            
    for _ in range(m - 1):
        best_mask = -1
        best_degree = -1
        best_pop = -1
        
        for mask in range(num_masks):
            if mask in used_masks:
                continue
            deg = degrees[mask]
            if deg > best_degree or (deg == best_degree and popcount[mask] > best_pop):
                best_degree = deg
                best_pop = popcount[mask]
                best_mask = mask
        
        if best_mask == -1:
            # Fallback, shouldn't happen
            best_mask = random.randint(0, full_mask)
            while best_mask in used_masks:
                best_mask = random.randint(0, full_mask)
        
        current_masks.append(best_mask)
        used_masks.add(best_mask)
        
        # Update degrees
        base = best_mask * num_masks
        for k in range(num_masks):
            if comp[base + k]:
                degrees[k] += 1
                
    best_score = calc_score(current_masks)
    best_solution = list(current_masks)
    
    # 2. Simulated Annealing with precomputed comparability
    random.seed(42)
    
    restarts = 0
    max_restarts = 30
    
    while time.time() < deadline and restarts < max_restarts:
        current_masks = list(best_solution)
        used_masks = set(current_masks)
        current_score = best_score
        
        T = 8.0
        cooling_rate = 0.99998
        min_T = 0.0005
        no_improve_steps = 0
        max_no_improve = 10000
        
        while time.time() < deadline and T > min_T:
            # Pick a random index to modify
            idx = random.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate new mask: mostly small perturbations
            r = random.random()
            if r < 0.65:
                # Flip one bit
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
            elif r < 0.85:
                # Flip two bits
                if old_mask == 0:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = (1 << b1) | (1 << b2)
                elif old_mask == full_mask:
                    b1 = random.randint(0, n - 1)
                    b2 = random.randint(0, n - 1)
                    while b2 == b1:
                        b2 = random.randint(0, n - 1)
                    new_mask = full_mask ^ (1 << b1) ^ (1 << b2)
                else:
                    all_bits = list(range(n))
                    random.shuffle(all_bits)
                    b1 = all_bits[0]
                    b2 = all_bits[1]
                    new_mask = old_mask ^ (1 << b1) ^ (1 << b2)
            else:
                # Random mask
                new_mask = random.randint(0, full_mask)
            
            if new_mask == old_mask or new_mask in used_masks:
                continue

            # Calculate delta using precomputed matrix
            delta = 0
            old_base = old_mask * num_masks
            new_base = new_mask * num_masks
            for j in range(m):
                if j == idx:
                    continue
                cm = current_masks[j]
                old_c = comp[old_base + cm]
                new_c = comp[new_base + cm]
                delta += new_c - old_c
            
            # Acceptance
            if delta >= 0 or random.random() < math.exp(delta / T):
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                current_score += delta
                if delta > 0:
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
            else:
                no_improve_steps += 1
            
            T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                break
        
        restarts += 1
    
    # Final polish: local search with only positive delta moves
    current_masks = list(best_solution)
    used_masks = set(current_masks)
    current_score = best_score
    
    for _ in range(5000):
        if time.time() > deadline:
            break
        improved = False
        for idx in range(m):
            if time.time() > deadline:
                break
            old_mask = current_masks[idx]
            best_new_mask = -1
            best_delta = 0
            old_base = old_mask * num_masks
            
            for mask in range(num_masks):
                if mask == old_mask or mask in used_masks:
                    continue
                new_base = mask * num_masks
                delta = 0
                for j in range(m):
                    if j == idx:
                        continue
                    cm = current_masks[j]
                    old_c = comp[old_base + cm]
                    new_c = comp[new_base + cm]
                    delta += new_c - old_c
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = mask
            
            if best_new_mask != -1 and best_delta > 0:
                used_masks.remove(old_mask)
                used_masks.add(best_new_mask)
                current_masks[idx] = best_new_mask
                current_score += best_delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_masks)
        
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
