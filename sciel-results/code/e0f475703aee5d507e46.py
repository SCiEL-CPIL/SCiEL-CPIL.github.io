import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512
    POP_SIZE = 50
    ELITE_SIZE = 5

    # Precompute bit masks for each value 1..512
    bit_masks = [1 << i for i in range(MAX_VAL + 1)]

    def score_mask(mask):
        count = 0
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= bit_masks[x]
        return m

    def mask_to_list(mask):
        lst = []
        for x in range(1, MAX_VAL + 1):
            if mask & bit_masks[x]:
                lst.append(x)
        return lst

    def greedy_construct():
        current_mask = 0
        current_set = set()
        for _ in range(N):
            best_val = -1
            best_score = -1
            for x in range(1, MAX_VAL + 1):
                if x in current_set:
                    continue
                new_mask = current_mask | bit_masks[x]
                s = score_mask(new_mask)
                if s > best_score:
                    best_score = s
                    best_val = x
            if best_val != -1:
                current_set.add(best_val)
                current_mask |= bit_masks[best_val]
        return current_mask

    # Initialize population
    population = []

    # Greedy start
    g_mask = greedy_construct()
    g_score = score_mask(g_mask)
    population.append((g_mask, g_score))

    # Random starts
    for _ in range(POP_SIZE - 1):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        m = list_to_mask(lst)
        s = score_mask(m)
        population.append((m, s))

    # Sort by score descending
    population.sort(key=lambda x: x[1], reverse=True)

    best_mask = population[0][0]
    best_score = population[0][1]

    # Genetic Algorithm
    while time.time() < deadline:
        # Selection and reproduction
        new_population = []

        # Elitism: keep top ELITE_SIZE
        for i in range(ELITE_SIZE):
            new_population.append((population[i][0], population[i][1]))

        # Fill rest of population with offspring
        while len(new_population) < POP_SIZE:
            # Tournament selection
            def tournament_select():
                idx1 = random.randint(0, POP_SIZE - 1)
                idx2 = random.randint(0, POP_SIZE - 1)
                if population[idx1][1] >= population[idx2][1]:
                    return population[idx1][0]
                else:
                    return population[idx2][0]

            parent1_mask = tournament_select()
            parent2_mask = tournament_select()

            # Crossover: uniform crossover
            child_mask = 0
            for i in range(1, MAX_VAL + 1):
                if random.random() < 0.5:
                    if parent1_mask & bit_masks[i]:
                        child_mask |= bit_masks[i]
                else:
                    if parent2_mask & bit_masks[i]:
                        child_mask |= bit_masks[i]

            # Ensure exactly N elements
            child_list = mask_to_list(child_mask)
            if len(child_list) > N:
                # Remove random excess
                excess = len(child_list) - N
                for _ in range(excess):
                    idx = random.randint(0, len(child_list) - 1)
                    child_list.pop(idx)
            elif len(child_list) < N:
                # Add random missing
                missing_needed = N - len(child_list)
                child_set = set(child_list)
                for _ in range(missing_needed):
                    while True:
                        x = random.randint(1, MAX_VAL)
                        if x not in child_set:
                            child_list.append(x)
                            child_set.add(x)
                            break

            child_mask = list_to_mask(child_list)

            # Mutation: flip random bits
            if random.random() < 0.3:
                # Remove a random element
                if child_list:
                    idx = random.randint(0, len(child_list) - 1)
                    child_mask ^= bit_masks[child_list[idx]]
                    child_list.pop(idx)
                    # Add a new random element
                    while True:
                        x = random.randint(1, MAX_VAL)
                        if x not in child_list:
                            child_mask |= bit_masks[x]
                            child_list.append(x)
                            break

            child_score = score_mask(child_mask)
            new_population.append((child_mask, child_score))

        # Update population
        population = new_population
        population.sort(key=lambda x: x[1], reverse=True)

        # Track best
        if population[0][1] > best_score:
            best_score = population[0][1]
            best_mask = population[0][0]

    return mask_to_list(best_mask)
