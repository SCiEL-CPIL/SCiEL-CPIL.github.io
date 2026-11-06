import random
import time

def construct():
    n = 512
    
    def creates_4ap_with_x(S, x):
        # S is a set
        # Check if adding x to S creates a 4-AP involving x.
        
        # Case 1: x is the first term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return True
        # Case 2: x is the second term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                return True
        # Case 3: x is the third term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                return True
        # Case 4: x is the fourth term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                return True
        return False

    def greedy_fill(candidates, existing_set):
        """
        Greedily add elements from candidates to existing_set
        such that no 4-AP is formed.
        Returns the new set.
        """
        S = set(existing_set)
        # Randomize order of candidates to avoid bias
        cand_list = list(candidates)
        random.shuffle(cand_list)
        for x in cand_list:
            if x not in S:
                if not creates_4ap_with_x(S, x):
                    S.add(x)
        return S

    # Initialize Population
    # Use different random seeds for initial greedy sets
    population = []
    for i in range(20):
        random.seed(i)
        init_set = set()
        all_elements = list(range(1, n + 1))
        random.shuffle(all_elements)
        for x in all_elements:
            if not creates_4ap_with_x(init_set, x):
                init_set.add(x)
        population.append(init_set)
        
    best_set = max(population, key=len)
    best_size = len(best_set)
    
    start_time = time.time()
    time_limit = 55.0
    
    # Genetic Algorithm Loop
    while time.time() - start_time < time_limit:
        # Selection: Keep top 5
        population.sort(key=len, reverse=True)
        population = population[:5]
        
        # Generate Offspring
        new_population = []
        # Elitism: Keep best 2
        for i in range(2):
            new_population.append(set(population[i]))
            
        # Crossover and Mutation
        while len(new_population) < 10:
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover: Union
            child = p1.union(p2)
            
            # Repair: Greedy fill on the union? 
            # No, the union might have 4-APs.
            # We need to remove elements to make it valid.
            # Greedy Repair:
            # Take elements of child in random order, add if valid.
            # This might discard good elements from p1 and p2.
            # Better: Start with p1, try to add elements from p2.
            # Or: Start with p2, try to add elements from p1.
            # Let's try: Start with p1, add valid elements from p2.
            child_set = set(p1)
            p2_elements = list(p2)
            random.shuffle(p2_elements)
            for x in p2_elements:
                if x not in child_set:
                    if not creates_4ap_with_x(child_set, x):
                        child_set.add(x)
                        
            # Mutation: Try to add random elements not in child
            # Or remove one and add another (swap)
            if random.random() < 0.5:
                # Add mutation
                candidates = [x for x in range(1, n+1) if x not in child_set]
                if candidates:
                    x = random.choice(candidates)
                    if not creates_4ap_with_x(child_set, x):
                        child_set.add(x)
            else:
                # Swap mutation
                if child_set:
                    x_out = random.choice(list(child_set))
                    candidates = [x for x in range(1, n+1) if x not in child_set]
                    if candidates:
                        x_in = random.choice(candidates)
                        S_prime = child_set - {x_out}
                        if not creates_4ap_with_x(S_prime, x_in):
                            child_set.remove(x_out)
                            child_set.add(x_in)
                            
            new_population.append(child_set)
            
        # Update Population
        population = new_population
        
        # Update Best
        for s in population:
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)

    return sorted(best_set)
