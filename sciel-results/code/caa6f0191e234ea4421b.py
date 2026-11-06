import time
import random

def construct():
    n = 512
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # a >= 1, a+3d <= n
    ap_data = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            x1 = a
            x2 = a + d
            x3 = a + 2 * d
            x4 = a + 3 * d
            # Mask for these 4 elements
            # Elements are 1-indexed, so bit index is x-1
            mask = (1 << (x1 - 1)) | (1 << (x2 - 1)) | (1 << (x3 - 1)) | (1 << (x4 - 1))
            ap_data.append((mask, (x1, x2, x3, x4)))
    
    num_aps = len(ap_data)
    
    def get_violation(mask):
        for i in range(num_aps):
            m, _ = ap_data[i]
            if (mask & m) == m:
                return i
        return -1

    def repair(mask):
        # Repair mask to be 4-AP free by removing elements from violated APs
        # This function is called after mutation/crossover
        # We limit repair steps to avoid infinite loops or excessive cost
        steps = 0
        while steps < 1000:
            idx = get_violation(mask)
            if idx == -1:
                return mask
            _, tup = ap_data[idx]
            # Pick a random element from the violated AP to remove
            # This is a heuristic; removing any one breaks the AP
            x = random.choice(tup)
            mask &= ~(1 << (x - 1))
            steps += 1
        # If we couldn't repair in 1000 steps, return current mask (likely still invalid, but we will penalize or handle)
        # For robustness, if it fails, we might just return a small valid set or keep trying
        # But usually, removing 1 element per AP is sufficient for small mutations.
        return mask

    def popcount(mask):
        return bin(mask).count('1')

    def greedy_init(order):
        S = 0
        for x in order:
            # Check if adding x creates a 4-AP
            # We can check if (S | (1 << (x-1))) is valid?
            # That's expensive (44k checks).
            # Instead, check if x is part of any AP with 3 other elements in S.
            # This is the same logic as the original program's creates_4ap_with_x.
            # Let's reuse that logic but return boolean.
            
            # Case 1: x is 1st
            max_d = (n - x) // 3
            valid = True
            for d in range(1, max_d + 1):
                if ((1 << (x + d - 1)) & S) and ((1 << (x + 2 * d - 1)) & S) and ((1 << (x + 3 * d - 1)) & S):
                    valid = False
                    break
            if not valid:
                continue
                
            # Case 2: x is 2nd
            max_d = min(x - 1, (n - x) // 2)
            for d in range(1, max_d + 1):
                if ((1 << (x - d - 1)) & S) and ((1 << (x + d - 1)) & S) and ((1 << (x + 2 * d - 1)) & S):
                    valid = False
                    break
            if not valid:
                continue
                
            # Case 3: x is 3rd
            max_d = min((x - 1) // 2, n - x)
            for d in range(1, max_d + 1):
                if ((1 << (x - 2 * d - 1)) & S) and ((1 << (x - d - 1)) & S) and ((1 << (x + d - 1)) & S):
                    valid = False
                    break
            if not valid:
                continue
                
            # Case 4: x is 4th
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if ((1 << (x - 3 * d - 1)) & S) and ((1 << (x - 2 * d - 1)) & S) and ((1 << (x - d - 1)) & S):
                    valid = False
                    break
            if not valid:
                continue
                
            S |= (1 << (x - 1))
        return S

    # Initialize Population
    pop_size = 50
    population = []
    
    # Create diverse greedy sets
    for i in range(pop_size):
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_init(order)
        population.append(S)
        
    # Ensure at least one valid set (greedy always produces valid)
    # Evaluate initial fitness
    def evaluate(mask):
        # Verify validity just in case, though greedy is valid
        # If invalid, repair it (shouldn't happen for initial, but good for safety)
        if get_violation(mask) != -1:
            mask = repair(mask)
        return popcount(mask)
    
    best_mask = max(population, key=evaluate)
    best_size = evaluate(best_mask)
    
    start_time = time.time()
    time_limit = 55.0
    
    # GA Loop
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        new_population = []
        
        # Elitism: keep top 10%
        population.sort(key=evaluate, reverse=True)
        elite_count = max(1, pop_size // 10)
        new_population.extend(population[:elite_count])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            p1 = random.choice(population)
            p2 = random.choice(population)
            if evaluate(p1) > evaluate(p2):
                parent1 = p1
            else:
                parent1 = p2
            
            p3 = random.choice(population)
            p4 = random.choice(population)
            if evaluate(p3) > evaluate(p4):
                parent2 = p3
            else:
                parent2 = p4
                
            # Crossover: Uniform
            # Combine bits
            # We can do: child = (parent1 & random_mask) | (parent2 & ~random_mask)
            # But we need to ensure we don't lose too many bits.
            # Simple uniform crossover:
            child = 0
            for i in range(n):
                if random.random() < 0.5:
                    if (parent1 >> i) & 1:
                        child |= (1 << i)
                else:
                    if (parent2 >> i) & 1:
                        child |= (1 << i)
                        
            # Mutation: Flip 1-5 bits
            num_mut = random.randint(1, 5)
            for _ in range(num_mut):
                bit_pos = random.randint(0, n - 1)
                child ^= (1 << bit_pos)
                
            # Repair
            child = repair(child)
            
            new_population.append(child)
            
        population = new_population
        
        # Update best
        current_best = max(population, key=evaluate)
        current_best_size = evaluate(current_best)
        if current_best_size > best_size:
            best_size = current_best_size
            best_mask = current_best
            
    # Convert best_mask to sorted list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)
