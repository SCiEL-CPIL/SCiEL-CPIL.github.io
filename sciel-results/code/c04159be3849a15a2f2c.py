def construct():
    import math
    import random
    import time

    def score_of(U):
        U = sorted(set(U))
        if 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        diff_set = set()
        sum_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
                sum_set.add(a + b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0 or n_diff == 0:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1
        return 1 + math.log(ratio) / denom

    random.seed(42)
    
    # Initial population
    pop_size = 50
    max_val = 100
    min_size = 4
    max_size = 8
    
    def random_set():
        n = random.randint(min_size, max_size)
        elems = random.sample(range(1, max_val + 1), n - 1)
        return sorted([0] + elems)
    
    # Seed with some known good structures and randoms
    population = []
    # Add some structured seeds
    structured_seeds = [
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 10, 31],
        [0, 1, 4, 13, 40],
        [0, 2, 5, 12, 29],
        [0, 1, 3, 6, 12],
        [0, 1, 4, 10, 25],
        [0, 1, 3, 7, 19],
    ]
    for s in structured_seeds:
        population.append(list(s))
        
    while len(population) < pop_size:
        population.append(random_set())
        
    best_score = 0
    best_set = population[0]
    
    # Evaluate initial population
    for ind in population:
        sc = score_of(ind)
        if sc > best_score:
            best_score = sc
            best_set = list(ind)
            
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    while time.time() - start_time < time_limit:
        new_population = []
        
        # Elitism: keep the best few
        scored_pop = [(score_of(ind), ind) for ind in population]
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        for i in range(min(5, len(scored_pop))):
            new_population.append(list(scored_pop[i][1]))
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = random.choice(scored_pop[:10])[1]
            parent2 = random.choice(scored_pop[:10])[1]
            
            # Crossover: mix elements
            child = list(set(parent1 + parent2))
            if 0 not in child:
                child.append(0)
            child = sorted(child)
            
            # Enforce size constraints
            if len(child) > max_size:
                # Remove largest elements to keep size down and max_val low
                child = child[:max_size]
            elif len(child) < min_size:
                # Add random small elements
                while len(child) < min_size:
                    if random.random() < 0.5:
                        new_elem = random.randint(1, max_val)
                    else:
                        new_elem = random.randint(1, max(1, child[-1]//2))
                    if new_elem not in child:
                        child.append(new_elem)
                child = sorted(child)
            
            # Mutation
            if random.random() < 0.7:
                # Change one element
                idx = random.randint(1, len(child)-1) # Don't change 0
                if random.random() < 0.5:
                    # Replace
                    new_val = random.randint(1, max_val)
                    child[idx] = new_val
                else:
                    # Add new element
                    if len(child) < max_size:
                        new_val = random.randint(1, max_val)
                        child.append(new_val)
            if random.random() < 0.3:
                # Remove one element (not 0)
                if len(child) > min_size:
                    idx = random.randint(1, len(child)-1)
                    child.pop(idx)
            
            child = sorted(set(child))
            if 0 not in child:
                child.insert(0, 0)
            
            new_population.append(child)
            
        population = new_population
        
        # Update best
        for ind in population:
            sc = score_of(ind)
            if sc > best_score:
                best_score = sc
                best_set = list(ind)
                
    return best_set
