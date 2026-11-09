import random
import time

N = 256

def construct():
    best_set = None
    best_size = N + 1
    start_time = time.time()
    time_limit = 25.0  # Leave 5s buffer
    
    # Helper: Greedy construction given a specific order of candidates
    # Returns the maximal Sidon set constructed.
    def greedy_build(order):
        selected = []
        sums = set()
        for x in order:
            new_sums = []
            valid = True
            for s in selected:
                val = x + s
                if val in sums or val in new_sums:
                    valid = False
                    break
                new_sums.append(val)
            
            if valid:
                val = 2 * x
                if val in sums or val in new_sums:
                    valid = False
                else:
                    new_sums.append(val)
            
            if valid:
                selected.append(x)
                sums.update(new_sums)
        return selected

    # Helper: Generate a random maximal set
    def random_maximal():
        order = list(range(1, N + 1))
        random.shuffle(order)
        return greedy_build(order)

    # Initialize with a few random maximal sets
    population = []
    for _ in range(50):
        s = random_maximal()
        population.append(s)
        if len(s) < best_size:
            best_size = len(s)
            best_set = s.copy()

    # Main Loop: Seeded Greedy Exploration
    # We take a good set, try to preserve its structure by prioritizing its elements
    # in a new greedy pass, but with randomization to escape local optima.
    
    while time.time() - start_time < time_limit:
        # Pick a parent from the best few sets
        # Sort population by size to pick better parents
        population.sort(key=lambda x: len(x))
        parent = random.choice(population[:10]) if len(population) >= 10 else random.choice(population)
        
        # Create a new order
        # 1. Take elements from parent
        parent_elements = set(parent)
        missing_elements = [x for x in range(1, N + 1) if x not in parent_elements]
        
        # Strategy: 
        # - Keep a random subset of parent elements (e.g., 60-90%)
        # - Randomize their order
        # - Randomize missing elements
        # - Concatenate: [Kept Parent Elements] + [Missing Elements]
        # This biases the greedy builder to keep the "good" parts of the parent.
        
        keep_prob = random.uniform(0.6, 0.9)
        kept = [x for x in parent if random.random() < keep_prob]
        random.shuffle(kept)
        
        random.shuffle(missing_elements)
        
        new_order = kept + missing_elements
        
        new_set = greedy_build(new_order)
        
        if len(new_set) < best_size:
            best_size = len(new_set)
            best_set = new_set.copy()
            # Reset population to focus on the new best? Or just add it.
            # Adding it is sufficient.
            population.append(new_set)
            
            # If we find a very small set, we can restart population around it
            if best_size <= 11: # Threshold for aggressive restart
                 population = [new_set.copy() for _ in range(20)]
                 # Add some random ones to maintain diversity
                 for _ in range(10):
                     population.append(random_maximal())
        
        # Maintain population size
        if len(population) > 100:
            # Keep the smallest 50 and some random ones
            population.sort(key=lambda x: len(x))
            population = population[:50] + [random_maximal() for _ in range(50)]

    return best_set
