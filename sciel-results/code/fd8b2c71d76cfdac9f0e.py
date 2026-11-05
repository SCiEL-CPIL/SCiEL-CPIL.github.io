import time
import random

N = 256

# Precompute 3-AP pair masks for each element
# For element x, store masks of the two other elements needed to form a 3-AP
pair_masks = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    # x first: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        mask = (1 << (x + d - 1)) | (1 << (x + 2 * d - 1))
        pair_masks[x].append(mask)
    # x middle: x-d, x, x+d
    for d in range(1, min(x - 1, N - x) + 1):
        mask = (1 << (x - d - 1)) | (1 << (x + d - 1))
        pair_masks[x].append(mask)
    # x last: x-2d, x-d, x
    for d in range(1, (x - 1) // 2 + 1):
        mask = (1 << (x - 2 * d - 1)) | (1 << (x - d - 1))
        pair_masks[x].append(mask)

# Precompute 4-AP conflict masks: for each x, list of masks of the other 3 elements
f4_masks = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        f4_masks[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        f4_masks[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        f4_masks[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        f4_masks[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def creates_4AP(A, x):
    for mask in f4_masks[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_add(A, x):
    """Number of 3-APs formed by adding x to A."""
    g = 0
    for mask in pair_masks[x]:
        if (A & mask) == mask:
            g += 1
    return g

def loss_remove(A, x):
    """Number of 3-APs lost by removing x from A."""
    l = 0
    for mask in pair_masks[x]:
        if (A & mask) == mask:
            l += 1
    return l

def count_3AP_total(A):
    """Count total 3-APs in set A."""
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        for mask in pair_masks[b]:
            if (A & mask) == mask:
                count += 1
    return count

def greedy_seed(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not creates_4AP(A, x):
            A |= (1 << (x - 1))
    return A

def local_search(A, deadline):
    best_score = count_3AP_total(A)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try single additions
        best_gain = 0
        best_x = -1
        for x in range(1, N + 1):
            if is_in_set(A, x):
                continue
            if creates_4AP(A, x):
                continue
            g = gain_add(A, x)
            if g > best_gain:
                best_gain = g
                best_x = x
        
        if best_x != -1:
            A |= (1 << (best_x - 1))
            best_score += best_gain
            improved = True
            continue
        
        # 2. Try single swaps
        best_net = 0
        best_r = -1
        best_x = -1
        
        for r in range(1, N + 1):
            if not is_in_set(A, r):
                continue
            
            loss = loss_remove(A, r)
            A_temp = A & ~(1 << (r - 1))
            
            for x in range(1, N + 1):
                if x == r or is_in_set(A, x):
                    continue
                if creates_4AP(A_temp, x):
                    continue
                g = gain_add(A_temp, x)
                net = g - loss
                if net > best_net:
                    best_net = net
                    best_r = r
                    best_x = x
        
        if best_r != -1 and best_x != -1:
            A = A & ~(1 << (best_r - 1))
            A |= (1 << (best_x - 1))
            best_score += best_net
            improved = True
            continue
        
        if not improved:
            break
             
    return best_score, A

def genetic_algorithm(deadline, rng):
    """
    Genetic algorithm with bitmask-based fitness evaluation.
    Evolves populations of 4-AP-free subsets via crossover and mutation.
    """
    pop_size = 20
    elite_count = 2
    mutation_rate = 0.1
    crossover_rate = 0.7
    
    # Initialize population with diverse seeds
    population = []
    for _ in range(pop_size):
        A = greedy_seed(rng)
        population.append(A)
    
    best_A = max(population, key=count_3AP_total)
    best_score = count_3AP_total(best_A)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate fitness
        fitnesses = [(count_3AP_total(A), A) for A in population]
        fitnesses.sort(key=lambda x: -x[0])
        
        # Update best
        if fitnesses[0][0] > best_score:
            best_score = fitnesses[0][0]
            best_A = fitnesses[0][1]
        
        # Selection: keep elites
        new_population = [A for _, A in fitnesses[:elite_count]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection for parents
            def tournament_select():
                idx1 = rng.randint(0, len(fitnesses) - 1)
                idx2 = rng.randint(0, len(fitnesses) - 1)
                return fitnesses[idx1][1] if fitnesses[idx1][0] >= fitnesses[idx2][0] else fitnesses[idx2][1]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            if rng.random() < crossover_rate:
                # Crossover: uniform crossover on bitmask
                child = (parent1 & parent2)  # Intersection first
                # Randomly add elements from parent1 not in parent2
                diff1 = parent1 & ~parent2
                diff2 = parent2 & ~parent1
                for bit in range(N):
                    if (diff1 >> bit) & 1:
                        if rng.random() < 0.5:
                            child |= (1 << bit)
                    elif (diff2 >> bit) & 1:
                        if rng.random() < 0.5:
                            child |= (1 << bit)
            else:
                child = parent1
            
            # Mutation: random bit flips
            for bit in range(N):
                if rng.random() < mutation_rate / N:
                    child ^= (1 << bit)
            
            # Repair: ensure no 4-APs by removing conflicting elements
            # Check each element and remove if it creates a 4-AP
            repaired = 0
            for x in range(1, N + 1):
                if (child >> (x - 1)) & 1:
                    # Check if x creates a 4-AP with current set minus x
                    A_without_x = child & ~(1 << (x - 1))
                    if creates_4AP(A_without_x, x):
                        child &= ~(1 << (x - 1))
                        repaired += 1
            
            # If repair removed too many, try a simpler repair
            if repaired > 5:
                # Just keep the intersection of parents, which is safer
                child = parent1 & parent2
            
            new_population.append(child)
        
        population = new_population
    
    return best_score, best_A

def construct():
    start = time.time()
    deadline = start + 55.0
    rng = random.Random(12345)

    best_A = 0
    best_score = -1

    # First, try local search from multiple seeds
    while time.time() < deadline - 20.0:
        A = greedy_seed(rng)
        score, A = local_search(A, min(deadline - 15.0, time.time() + 5.0))
        if score > best_score:
            best_score = score
            best_A = A

    # Then, try genetic algorithm with remaining time
    if time.time() < deadline - 5.0:
        ga_score, ga_A = genetic_algorithm(deadline - 1.0, rng)
        if ga_score > best_score:
            best_score = ga_score
            best_A = ga_A

    # Final local search polish
    if time.time() < deadline - 0.5 and best_A != 0:
        score, best_A = local_search(best_A, deadline)
        best_score = score

    if best_A == 0:
        best_A = 1

    A_list = [x for x in range(1, N + 1) if is_in_set(best_A, x)]
    
    if not A_list:
        A_list = [1]
        
    return A_list

if __name__ == "__main__":
    res = construct()
    print(len(res))
