import time
import random

def construct():
    N = 256
    deadline = time.time() + 85.0  # Safety margin before 90s limit

    # Precompute 4-AP conflicts using bitmasks for O(1) validity checks
    # conflicts[x] is a list of masks where each mask has bits set for the other 3 elements of a 4-AP containing x
    # If all those bits are set in the current mask, adding x creates a 4-AP
    conflicts = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: (a, b, c, e)
            mask_a = (1 << b) | (1 << c) | (1 << e)
            conflicts[a].append(mask_a)
            mask_b = (1 << a) | (1 << c) | (1 << e)
            conflicts[b].append(mask_b)
            mask_c = (1 << a) | (1 << b) | (1 << e)
            conflicts[c].append(mask_c)
            mask_e = (1 << a) | (1 << b) | (1 << c)
            conflicts[e].append(mask_e)

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for cmask in conflicts[x]:
            if (mask & cmask) == cmask:
                return False
        return True

    def count_3aps(mask):
        """Count 3-term APs in the set represented by mask."""
        count = 0
        for i in range(N):
            if not ((mask >> i) & 1):
                continue
            max_d = (N - 1 - i) // 2
            for d in range(1, max_d + 1):
                j = i + d
                k = i + 2 * d
                if ((mask >> j) & 1) and ((mask >> k) & 1):
                    count += 1
        return count

    def delta_3aps_add(mask, x):
        """Calculate increase in 3-AP count if x is added to mask."""
        delta = 0
        # Case 1: x is the start: x, x+d, x+2d
        max_d = (N - 1 - x) // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x + d)) & 1) and ((mask >> (x + 2 * d)) & 1):
                delta += 1
        # Case 2: x is the middle: x-d, x, x+d
        max_d = min(x, N - 1 - x)
        for d in range(1, max_d + 1):
            if ((mask >> (x - d)) & 1) and ((mask >> (x + d)) & 1):
                delta += 1
        # Case 3: x is the end: x-2d, x-d, x
        max_d = x // 2
        for d in range(1, max_d + 1):
            if ((mask >> (x - 2 * d)) & 1) and ((mask >> (x - d)) & 1):
                delta += 1
        return delta

    def delta_3aps_remove(mask, x):
        """Calculate decrease in 3-AP count if x is removed from mask."""
        return delta_3aps_add(mask, x)

    def greedy_init():
        """Build a valid set quickly using greedy approach."""
        mask = 0
        elems = list(range(N))
        random.shuffle(elems)
        
        for x in elems:
            if is_valid_add(mask, x):
                if delta_3aps_add(mask, x) > 0 or random.random() < 0.3:
                    mask |= (1 << x)
        return mask

    def crossover(parent1, parent2):
        """Uniform crossover: for each bit, pick from parent1 or parent2."""
        child = 0
        for i in range(N):
            if random.random() < 0.5:
                if (parent1 >> i) & 1:
                    child |= (1 << i)
            else:
                if (parent2 >> i) & 1:
                    child |= (1 << i)
        return child

    def mutate(mask, mutation_rate=0.05):
        """Random mutation: flip bits with given probability."""
        new_mask = mask
        for i in range(N):
            if random.random() < mutation_rate:
                new_mask ^= (1 << i)
        return new_mask

    def repair(mask):
        """Remove elements that create 4-APs."""
        # Check each element in order, remove if it creates a 4-AP
        new_mask = 0
        for i in range(N):
            if (mask >> i) & 1:
                if is_valid_add(new_mask, i):
                    new_mask |= (1 << i)
        return new_mask

    # Initialize population
    pop_size = 20
    population = []
    scores = []
    
    for _ in range(pop_size):
        mask = greedy_init()
        population.append(mask)
        scores.append(count_3aps(mask))
    
    best_mask = population[0]
    best_score = scores[0]
    for i in range(1, pop_size):
        if scores[i] > best_score:
            best_score = scores[i]
            best_mask = population[i]

    # Genetic Algorithm
    elite_size = 3
    tournament_size = 5
    mutation_rate = 0.03
    repair_prob = 0.3

    while time.time() < deadline:
        # Select parents using tournament selection
        parent1_idx = random.randint(0, pop_size - 1)
        parent2_idx = random.randint(0, pop_size - 1)
        for _ in range(tournament_size - 1):
            c1 = random.randint(0, pop_size - 1)
            if scores[c1] > scores[parent1_idx]:
                parent1_idx = c1
            c2 = random.randint(0, pop_size - 1)
            if scores[c2] > scores[parent2_idx]:
                parent2_idx = c2
        
        # Crossover
        if random.random() < 0.7:
            child = crossover(population[parent1_idx], population[parent2_idx])
        else:
            child = population[parent1_idx]
        
        # Mutation
        child = mutate(child, mutation_rate)
        
        # Repair if invalid
        if random.random() < repair_prob:
            child = repair(child)
        
        # Evaluate child
        child_score = count_3aps(child)
        
        # Check if child is valid (no 4-APs)
        valid = True
        for i in range(N):
            if (child >> i) & 1:
                if not is_valid_add(child & ~(1 << i), i):
                    valid = False
                    break
        
        if not valid:
            # Try to repair
            child = repair(child)
            child_score = count_3aps(child)
        
        # Update population
        # Find worst individual
        worst_idx = 0
        for i in range(1, pop_size):
            if scores[i] < scores[worst_idx]:
                worst_idx = i
        
        # Replace worst if child is better
        if child_score > scores[worst_idx]:
            population[worst_idx] = child
            scores[worst_idx] = child_score
        
        # Update best
        if child_score > best_score:
            best_score = child_score
            best_mask = child

    # Convert best_mask back to sorted list (1-indexed)
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    return sorted(result)
