import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]

    def score_mask(mask):
        valid_diffs = 0
        # Max d is such that a + 2d <= 512, so 2d <= 511 -> d <= 255
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
        return valid_diffs

    def score_list(lst):
        if len(lst) != N:
            return -1
        if len(set(lst)) != N:
            return -1
        mask = list_to_mask(lst)
        return score_mask(mask)

    # Generate initial population with diverse structures
    def generate_initial_population(size=60):
        pop = []
        
        # 1. Base-4 Cantor-like set (digits 0,1)
        base4_powers = [1, 4, 16, 64, 256]
        base4_combos = []
        for mask_bits in range(32):
            val = 0
            for k in range(5):
                if (mask_bits >> k) & 1:
                    val += base4_powers[k]
            base4_combos.append(val)
        
        # Try a few offsets for base-4
        for offset in [1, 5, 10, 50, 100]:
            cand = [offset + v for v in base4_combos]
            if all(1 <= x <= MAX_VAL for x in cand):
                pop.append(cand)
                
        # 2. Base-3 Cantor-like set (digits 0,1)
        base3_powers = [1, 3, 9, 27, 81]
        base3_combos = []
        for mask_bits in range(32):
            val = 0
            for k in range(5):
                if (mask_bits >> k) & 1:
                    val += base3_powers[k]
            base3_combos.append(val)
            
        for scale in [1, 2, 3]:
            for offset in [1, 5, 10]:
                cand = [offset + scale * v for v in base3_combos]
                if all(1 <= x <= MAX_VAL for x in cand):
                    pop.append(cand)

        # 3. Dense intervals
        pop.append(list(range(1, N + 1)))
        pop.append(list(range(MAX_VAL - N + 1, MAX_VAL + 1)))
        
        # 4. Random sets
        while len(pop) < size:
            pop.append(random.sample(range(1, MAX_VAL + 1), N))
            
        return pop

    def crossover(p1, p2):
        # Uniform crossover with repair
        child = []
        available = set(range(1, MAX_VAL + 1))
        for x in p1:
            available.discard(x)
        for x in p2:
            available.discard(x)
            
        # Pick N elements from the union of parents, preferring those in both
        # Simple approach: shuffle parents and pick N unique
        candidates = list(set(p1 + p2))
        random.shuffle(candidates)
        
        # If we don't have enough unique candidates from parents, fill with random
        if len(candidates) < N:
            missing = N - len(candidates)
            random_fill = random.sample(list(available), missing)
            candidates.extend(random_fill)
            
        child = candidates[:N]
        return child

    def mutate(individual, rate=0.1):
        new_ind = individual[:]
        for _ in range(int(N * rate)):
            idx = random.randint(0, N - 1)
            # Replace with a random number not in the set
            while True:
                v = random.randint(1, MAX_VAL)
                if v not in new_ind:
                    break
            new_ind[idx] = v
        return new_ind

    def tournament_select(pop, scores, k=3):
        selected = random.sample(range(len(pop)), k)
        best_idx = max(selected, key=lambda i: scores[i])
        return pop[best_idx][:]

    # Initialize
    population = generate_initial_population(60)
    scores = [score_list(ind) for ind in population]
    
    best_score = max(scores) if scores else -1
    best_list = population[scores.index(best_score)][:] if scores else []
    
    # Ensure best_list is valid
    if not best_list:
        best_list = random.sample(range(1, MAX_VAL + 1), N)
        best_score = score_list(best_list)

    # GA Loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        # Sort population by score
        sorted_indices = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        
        # Elitism: keep top 2
        new_population = []
        for i in range(min(2, len(sorted_indices))):
            new_population.append(population[sorted_indices[i]][:])
            
        # Generate rest of population
        while len(new_population) < len(population):
            p1 = tournament_select(population, scores)
            p2 = tournament_select(population, scores)
            
            if random.random() < 0.5:
                child = crossover(p1, p2)
            else:
                child = p1[:]
                
            child = mutate(child)
            
            # Ensure validity (distinct, in range)
            if len(set(child)) == N and all(1 <= x <= MAX_VAL for x in child):
                new_population.append(child)
            else:
                # Fallback to random if mutation broke it
                new_population.append(random.sample(range(1, MAX_VAL + 1), N))
        
        population = new_population
        scores = [score_list(ind) for ind in population]
        
        if max(scores) > best_score:
            best_score = max(scores)
            best_list = population[scores.index(best_score)][:]

    return best_list
