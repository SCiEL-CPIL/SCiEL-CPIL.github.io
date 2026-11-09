import time
import random
import numpy as np

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    # Precompute mask indices for evaluation
    # We represent a set as a boolean numpy array of size 513 (indices 0..512)
    
    def evaluate(S):
        """Count distinct d > 0 such that a, a+d, a+2d are all in S."""
        mask = np.zeros(513, dtype=bool)
        for x in S:
            mask[x] = True
        
        valid_diffs = 0
        # Max d is such that 2*d < 512, so d <= 255
        # Actually if a=1, a+2d <= 512 => 2d <= 511 => d <= 255
        for d in range(1, 256):
            # Check if there exists a such that mask[a] and mask[a+d] and mask[a+2d] are True
            # a ranges from 1 to 512-2d
            # mask[1:513-2d] corresponds to a
            # mask[1+d:513-d] corresponds to a+d
            # mask[1+2d:513] corresponds to a+2d
            
            # We can just check the whole array with shifts
            # Let's use slices to avoid allocation if possible, or just rely on numpy speed
            # mask[0:513-2d] & mask[d:513-d] & mask[2d:513]
            # Note: index 0 is unused, but including it is fine as long as we don't pick 0 as 'a'.
            # However, if a=0 is in set, it's invalid. But our sets are in 1..512.
            # So mask[0] is always False.
            
            # Optimization: Only check if there are enough elements? No, numpy is fast.
            if np.any(mask[:513-2*d] & mask[d:513-d] & mask[2*d:513]):
                valid_diffs += 1
                
        return valid_diffs

    def random_set():
        return set(random.sample(range(1, 513), 32))

    def greedy_init():
        """Greedy construction: add elements that maximize new progressions."""
        s = set()
        # Start with a random element
        s.add(random.randint(1, 512))
        
        for _ in range(31):
            best_elem = None
            best_score = -1
            # Sample candidates to speed up greedy step
            candidates = random.sample(range(1, 513), min(100, 512))
            for cand in candidates:
                if cand in s:
                    continue
                new_s = s | {cand}
                sc = evaluate(new_s)
                if sc > best_score:
                    best_score = sc
                    best_elem = cand
            if best_elem is not None:
                s.add(best_elem)
            else:
                while True:
                    val = random.randint(1, 512)
                    if val not in s:
                        s.add(val)
                        break
        return s

    # Initialize population
    pop_size = 40
    population = []
    
    # Seed with some known good structures and randoms
    # 1..32 is a known good start
    population.append(set(range(1, 33)))
    
    for _ in range(pop_size - 1):
        if random.random() < 0.3:
            population.append(greedy_init())
        else:
            population.append(random_set())
    
    # Evaluate initial population
    scored_pop = [(evaluate(ind), ind) for ind in population]
    scored_pop.sort(key=lambda x: -x[0])
    
    best_score = scored_pop[0][0]
    best_set = scored_pop[0][1].copy()
    
    # Genetic Algorithm Loop
    # Parameters
    mutation_rate = 0.1
    crossover_rate = 0.8
    tournament_size = 5
    elite_size = 2
    
    generation = 0
    while time.time() - start_time < timeout - 5.0:
        generation += 1
        
        # Selection: Tournament
        def select():
            tournament = random.sample(scored_pop, tournament_size)
            return max(tournament, key=lambda x: x[0])[1]
        
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(scored_pop[i][1])
        
        # Fill rest of population
        while len(new_population) < pop_size:
            parent1 = select()
            parent2 = select()
            
            if random.random() < crossover_rate:
                # Uniform Crossover
                child = set()
                p1_list = list(parent1)
                p2_list = list(parent2)
                # Ensure we pick 32 unique elements
                # Simple strategy: iterate through union, pick from p1 if possible, else p2?
                # Better: Create a list of 32 slots. For each slot, pick from p1 or p2.
                # But we need to ensure no duplicates and size 32.
                # Since parents have size 32, we can do:
                # Create a boolean mask for which parent each slot comes from?
                # No, elements are values.
                
                # Robust Crossover:
                # Take random subset of parent1 and random subset of parent2, merge, then fill gaps.
                # Or: 
                # 1. List all 64 elements (with duplicates if any? No, sets are unique).
                # 2. Shuffle the 64 elements? No, that mixes them.
                
                # Let's do: 
                # child = set()
                # for x in parent1:
                #     if random.random() < 0.5:
                #         child.add(x)
                # for x in parent2:
                #     if random.random() < 0.5 and x not in child:
                #         child.add(x)
                # while len(child) < 32:
                #     # Fill from remaining or random
                #     ...
                
                # This might not preserve structure well.
                # Alternative: "Ordered Crossover" style but for sets is tricky.
                
                # Let's use a simple mutation-heavy approach for "crossover" if needed, 
                # or just use mutation on a single parent (cloning).
                # Given the complexity, let's use:
                # Child starts as copy of parent1.
                # Then, for each element in parent2, if random < 0.5, replace a random element in child with it?
                
                # Let's try a different crossover:
                # Pick 16 elements from parent1 and 16 from parent2.
                # If duplicates, replace with random or from other parent.
                c1 = random.sample(list(parent1), 16)
                c2 = random.sample(list(parent2), 16)
                child = set(c1 + c2)
                # If duplicates, child size < 32
                if len(child) < 32:
                    # Fill missing from remaining of parents or random
                    remaining = (parent1 | parent2) - child
                    if len(remaining) >= 32 - len(child):
                        fill = random.sample(list(remaining), 32 - len(child))
                    else:
                        # Need randoms
                        fill = []
                        while len(fill) < 32 - len(child):
                            r = random.randint(1, 512)
                            if r not in child and r not in fill:
                                fill.append(r)
                    child.update(fill)
                
                new_population.append(child)
            else:
                # Clone
                new_population.append(set(parent1))
        
        # Mutation
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                ind = new_population[i]
                # Remove 1, Add 1
                to_remove = random.choice(list(ind))
                ind.discard(to_remove)
                while True:
                    to_add = random.randint(1, 512)
                    if to_add not in ind:
                        ind.add(to_add)
                        break
                new_population[i] = ind
        
        # Evaluate new population
        scored_new = [(evaluate(ind), ind) for ind in new_population]
        scored_new.sort(key=lambda x: -x[0])
        
        # Update best
        if scored_new[0][0] > best_score:
            best_score = scored_new[0][0]
            best_set = scored_new[0][1].copy()
        
        # Next generation is the new population (sorted)
        scored_pop = scored_new
        
        # Optional: Add some random individuals to maintain diversity
        if generation % 10 == 0:
            for _ in range(2):
                scored_pop.append((evaluate(random_set()), random_set()))
            scored_pop.sort(key=lambda x: -x[0])
            scored_pop = scored_pop[:pop_size]

    return list(best_set)
