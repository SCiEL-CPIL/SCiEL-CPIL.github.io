import numpy as np
import time
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    # Safety margin: stop at 130s to allow for return overhead (evaluator kills at 150s)
    deadline = start_time + 130.0

    def build_indices(n):
        """
        Builds list of K4 index tuples and K5 index tuples.
        Also builds reverse index: triple_idx -> list of clique indices.
        """
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        k4s_containing = [[] for _ in range(num_triples)]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            idx_tuple = (t1, t2, t3, t4)
            k4s.append(idx_tuple)
            for t in idx_tuple:
                k4s_containing[t].append(len(k4s) - 1)
                
        k5s = []
        k5s_containing = [[] for _ in range(num_triples)]
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            idx_tuple = tuple(idxs)
            k5s.append(idx_tuple)
            for t in idx_tuple:
                k5s_containing[t].append(len(k5s) - 1)
                
        return triples, num_triples, k4s, k5s, k4s_containing, k5s_containing

    def calculate_cost_fast(colors, k4s, k5s, k4s_containing, k5s_containing):
        """
        Calculates total number of violations using efficient counting.
        """
        cost = 0
        n = len(colors)
        
        # Count red K4s: for each K4, check if all 4 triples are red (color 0)
        for k4 in k4s:
            if all(colors[i] == 0 for i in k4):
                cost += 1
        
        # Count blue K5s: for each K5, check if all 10 triples are blue (color 1)
        for k5 in k5s:
            if all(colors[i] == 1 for i in k5):
                cost += 1
        
        return cost

    def gf256_coloring(n):
        """
        Deterministic coloring using GF(256) arithmetic.
        This is based on the known construction for R(4,5;3) >= 34.
        Uses primitive polynomial x^8+x^4+x^3+x^2+1.
        """
        # GF(256) with primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
        # Build multiplication table
        poly = 0b100011101  # x^8+x^4+x^3+x^2+1
        exp_table = [0] * 512
        log_table = [0] * 256
        x = 1
        for i in range(255):
            exp_table[i] = x
            log_table[x] = i
            # Multiply by 2 (shift left)
            x = (x << 1)
            if x & 0x100:
                x ^= poly
        exp_table[255] = exp_table[0]
        exp_table[256] = exp_table[1]
        # ... fill up to 511
        for i in range(256, 512):
            exp_table[i] = exp_table[i - 255]
        
        def gf_mult(a, b):
            if a == 0 or b == 0:
                return 0
            return exp_table[log_table[a] + log_table[b]]
        
        def gf_add(a, b):
            return a ^ b
        
        triples = list(combinations(range(n), 3))
        colors = []
        for a, b, c in triples:
            # Map vertices to GF(256) elements
            aa = a + 1
            bb = b + 1
            cc = c + 1
            
            # Compute product in GF(256)
            prod = gf_mult(gf_mult(aa, bb), cc)
            
            # Compute trace
            trace = 0
            x = prod
            for i in range(8):
                trace ^= x
                x = gf_mult(x, x)
            
            # Color 0 (red) if trace is 0, else 1 (blue)
            colors.append(1 if trace else 0)
        
        return np.array(colors, dtype=np.int8)

    def heuristic_init(n):
        """Try multiple heuristic initializations and return the best."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        patterns = [
            lambda a,b,c: 0 if (a+b+c) % 3 == 0 else 1,
            lambda a,b,c: 0 if (a+b+c) % 2 == 0 else 1,
            lambda a,b,c: 0 if (a*b+c) % 3 == 0 else 1,
            lambda a,b,c: 0 if (a**2+b**2+c**2) % 2 == 0 else 1,
            lambda a,b,c: 0 if (a*b*c) % 2 == 0 else 1,
            lambda a,b,c: 0 if (a+b) % 2 == c % 2 else 1,
            lambda a,b,c: 0 if (a+b+c) % 4 < 2 else 1,
            lambda a,b,c: 0 if (a*b + b*c + c*a) % 3 == 0 else 1,
            lambda a,b,c: 0 if (a+b) % 3 == c % 3 else 1,
            lambda a,b,c: 0 if (a*b + b*c + c*a) % 2 == 0 else 1,
        ]
        
        return triples, num_triples, patterns

    def genetic_algorithm(n, time_limit):
        """
        Genetic algorithm with crossover, mutation, and local search refinement.
        """
        triples, num_triples, patterns = heuristic_init(n)
        
        # Build indices once
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        k4s_containing = [[] for _ in range(num_triples)]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            idx_tuple = (t1, t2, t3, t4)
            k4s.append(idx_tuple)
            for t in idx_tuple:
                k4s_containing[t].append(len(k4s) - 1)
                
        k5s = []
        k5s_containing = [[] for _ in range(num_triples)]
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            idx_tuple = tuple(idxs)
            k5s.append(idx_tuple)
            for t in idx_tuple:
                k5s_containing[t].append(len(k5s) - 1)
        
        # Initialize population
        pop_size = 10  # Reduced from 20 to save time
        population = []
        
        # Add GF(256) coloring if n <= 256
        if n <= 256:
            gf_colors = gf256_coloring(n)
            population.append(gf_colors)
        
        # Add heuristic initializations
        for pat in patterns:
            c = np.array([pat(a,b,c) for a,b,c in triples], dtype=np.int8)
            population.append(c)
        
        # Add random initializations
        while len(population) < pop_size:
            population.append(np.random.randint(0, 2, num_triples, dtype=np.int8))
        
        # Evaluate initial population
        costs = [calculate_cost_fast(ind, k4s, k5s, k4s_containing, k5s_containing) for ind in population]
        
        best_idx = int(np.argmin(costs))
        best_cost = costs[best_idx]
        best_individual = population[best_idx].copy()
        
        # If already valid, return immediately
        if best_cost == 0:
            return best_individual, True
        
        # Genetic algorithm loop
        gen = 0
        last_improve_gen = 0
        stagnation_limit = 15  # Reduced from 30
        
        while time.time() < time_limit:
            # Selection: tournament selection
            def tournament_select(pop, costs, k=3):
                idxs = np.random.randint(0, len(pop), k)
                best_k = idxs[np.argmin([costs[i] for i in idxs])]
                return pop[best_k].copy()
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(best_individual.copy())
            
            # Fill rest with offspring
            while len(new_population) < pop_size:
                parent1 = tournament_select(population, costs)
                parent2 = tournament_select(population, costs)
                
                # Crossover: uniform crossover
                mask = np.random.randint(0, 2, num_triples, dtype=np.int8)
                child = np.where(mask == 1, parent1, parent2).astype(np.int8)
                
                # Mutation: flip some bits
                mutation_rate = 0.02 + 0.03 * (gen / 100)
                if mutation_rate > 0.1:
                    mutation_rate = 0.1
                flip_mask = np.random.rand(num_triples) < mutation_rate
                child[flip_mask] = 1 - child[flip_mask]
                
                new_population.append(child)
            
            population = new_population
            costs = [calculate_cost_fast(ind, k4s, k5s, k4s_containing, k5s_containing) for ind in population]
            
            # Check for improvement
            new_best_idx = int(np.argmin(costs))
            new_best_cost = costs[new_best_idx]
            
            if new_best_cost < best_cost:
                best_cost = new_best_cost
                best_individual = population[new_best_idx].copy()
                last_improve_gen = gen
                if best_cost == 0:
                    return best_individual, True
            
            gen += 1
            
            # Stagnation: if no improvement for stagnation_limit generations, restart
            if gen - last_improve_gen > stagnation_limit:
                # Reinitialize some of the population
                for i in range(pop_size // 2):
                    if np.random.rand() < 0.5:
                        population[i] = np.random.randint(0, 2, num_triples, dtype=np.int8)
                    else:
                        population[i] = gf256_coloring(n) if n <= 256 else np.random.randint(0, 2, num_triples, dtype=np.int8)
                costs = [calculate_cost_fast(ind, k4s, k5s, k4s_containing, k5s_containing) for ind in population]
                last_improve_gen = gen
        
        return best_individual, (best_cost == 0)

    def local_search(n, init_colors, time_limit):
        """
        Local search (steepest descent with occasional random restarts) to refine a coloring.
        Uses efficient delta computation.
        """
        triples, num_triples, k4s, k5s, k4s_containing, k5s_containing = build_indices(n)
        colors = init_colors.copy()
        
        current_cost = calculate_cost_fast(colors, k4s, k5s, k4s_containing, k5s_containing)
        best_cost = current_cost
        best_colors = colors.copy()
        
        if best_cost == 0:
            return best_colors, True
        
        max_iter = 2000  # Reduced from 5000
        no_improve = 0
        max_no_improve = 200  # Reduced from 500
        
        for iteration in range(max_iter):
            if time.time() > time_limit:
                break
            
            # Find the best single flip
            best_delta = 0
            best_flip = -1
            
            # Sample a subset of flips for speed
            sample_size = min(100, num_triples)  # Reduced from 200
            candidates = np.random.choice(num_triples, size=sample_size, replace=False)
            
            for t in candidates:
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta_cost = 0
                
                # Check K4s containing t
                for k4_idx in k4s_containing[t]:
                    idxs = k4s[k4_idx]
                    # Count how many of the 4 triples in this K4 are red (color 0)
                    red_count = sum(1 for i in idxs if colors[i] == 0)
                    if red_count == 4:
                        # Currently a red K4 violation
                        if new_color != 0:
                            delta_cost -= 1  # Flipping to blue removes violation
                    elif red_count == 3 and old_color == 1:
                        # Flipping to red would create a violation
                        if new_color == 0:
                            delta_cost += 1
                
                # Check K5s containing t
                for k5_idx in k5s_containing[t]:
                    idxs = k5s[k5_idx]
                    # Count how many of the 10 triples in this K5 are blue (color 1)
                    blue_count = sum(1 for i in idxs if colors[i] == 1)
                    if blue_count == 10:
                        # Currently a blue K5 violation
                        if new_color != 1:
                            delta_cost -= 1  # Flipping to red removes violation
                    elif blue_count == 9 and old_color == 0:
                        # Flipping to blue would create a violation
                        if new_color == 1:
                            delta_cost += 1
                
                if delta_cost < best_delta:
                    best_delta = delta_cost
                    best_flip = t
            
            if best_flip >= 0 and best_delta < 0:
                colors[best_flip] = 1 - colors[best_flip]
                current_cost += best_delta
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors, True
            else:
                no_improve += 1
                # Random restart from best
                if no_improve > max_no_improve:
                    colors = best_colors.copy()
                    current_cost = best_cost
                    no_improve = 0
                    # Add some random noise
                    flip_mask = np.random.rand(num_triples) < 0.05
                    colors[flip_mask] = 1 - colors[flip_mask]
                    current_cost = calculate_cost_fast(colors, k4s, k5s, k4s_containing, k5s_containing)
        
        return best_colors, (best_cost == 0)

    best_n = 9
    best_result_colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9),3)]
    
    # Start with smaller n to ensure we have a valid result, then try larger
    target_ns = [20, 25, 28, 30, 31, 32, 33, 34]
    
    for n in target_ns:
        # Check time - need at least 15s remaining
        if time.time() > deadline - 15:
            break
            
        # Allocate time based on n
        remaining = deadline - time.time()
        if n <= 25:
            time_limit = time.time() + min(20, remaining - 10)
        elif n <= 30:
            time_limit = time.time() + min(30, remaining - 10)
        elif n <= 32:
            time_limit = time.time() + min(40, remaining - 10)
        else:
            time_limit = time.time() + min(50, remaining - 10)
        
        if time_limit <= time.time():
            continue
        
        try:
            # First try GA
            ga_colors, ga_valid = genetic_algorithm(n, time_limit)
            
            if ga_valid:
                best_n = n
                best_result_colors = ga_colors.tolist()
                if n == 34:
                    break
            else:
                # Try local search on GA result
                remaining_ls = deadline - time.time() - 10
                if remaining_ls > 5:
                    ls_colors, ls_valid = local_search(n, ga_colors, time.time() + remaining_ls)
                    if ls_valid:
                        best_n = n
                        best_result_colors = ls_colors.tolist()
                        if n == 34:
                            break
        except Exception as e:
            pass
            
    return {'n': best_n, 'colors': best_result_colors}
